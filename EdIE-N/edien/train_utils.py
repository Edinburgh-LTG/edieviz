import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from edien.utils import get_current_git_hash


def patch_param_grid(bp):
    # sklearn annoyingly checks types
    if hasattr(bp, 'keys'):
        for k in bp.keys():
            if k == 'param_grid':
                print('Found gridsearch CV and patched')
                setattr(bp, k, bp[k].as_dict())
            else:
                patch_param_grid(bp[k])


def only_pseudorandomness_please(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def report_eval(true, preds, labels, verbose=False):
    cm = confusion_matrix(true, preds, labels=labels)
    cm = np.array(cm)

    PAD = 30
    # Smooth out dividing zeros
    epsilon = 1e-6
    macro_recall = np.diag(cm) / (cm.sum(axis=1) + epsilon)
    macro_precision = np.diag(cm) / (cm.sum(axis=0) + epsilon)
    macro_f1 = (2 * macro_precision * macro_recall)/(macro_precision + macro_recall + epsilon)

    # Micro f1 is the same as accuracy for multi-class
    micro_f1 = np.diag(cm).sum() / cm.sum()

    # micro_recall = np.diag(cm).sum() / cm.sum()
    # micro_precision = np.diag(cm).sum() / cm.sum()

    counts = cm.sum(axis=1)
    if verbose:
        print('%s\t%s\t%s\t%s\t%s' % ('Score'.ljust(PAD), 'F1', 'Prec', 'Rec', 'Counts'))
        for l, f, p, r, c in zip(labels, macro_f1, macro_precision, macro_recall, counts):
            print('%s\t%.2f\t%.2f\t%.2f\t%d' % (l.ljust(PAD),
                                                f * 100,
                                                p * 100,
                                                r * 100,
                                                c))
    print('='*60)
    print('%s\t%.2f\t%.2f\t%.2f\t%d' % ('Total Macro:'.ljust(PAD),
                                        macro_f1.mean() * 100,
                                        macro_precision.mean() * 100,
                                        macro_recall.mean() * 100,
                                        counts.sum()))
    print('%s\t%.2f' % ('Total Micro:'.ljust(PAD), micro_f1 * 100))
    # print(cm)
    print()
    # norm_cm = cm / cm.sum(axis=1, keepdims=True)
    # plot_matrix(norm_cm, labels, normalize=True)
    # plot_matrix(cm, labels, normalize=False)


def eval_loop(y_true, y_pred, verbose=False):
    # targets are the outputs we want to predict
    # we evaluate each output separately
    all_labels = []
    for target, preds in zip(y_pred._fields, y_pred):
        print(target)
        gold = getattr(y_true, target)
        labels = set(gold)
        labels = tuple(sorted(labels, key=lambda x: (x[2:], x)))
        assert len(gold) == len(preds)
        if target == 'negation':
            # We want to do entity level stuff - so let's only take B- prediction
            ent_preds, ent_gold = [], []
            # We purposely double count tokens that are tagged both
            # as modifiers and entities
            for g, p, ent, mod in zip(gold, preds, y_true.ner_tags, y_true.mod_tags):
                if ent[:2] == 'B-':
                    ent_gold.append(g)
                    ent_preds.append(p)
                if mod[:2] == 'B-':
                    ent_gold.append(g)
                    ent_preds.append(p)
            gold, preds = ent_gold, ent_preds
        report_eval(gold, preds, labels=labels, verbose=verbose)
        all_labels.append(labels)

    return all_labels


def train_loop(conf):

    only_pseudorandomness_please(conf.seed)
    # Patch blueprint inconsistencies
    patch_param_grid(conf)
    conf.paths.experiment_name = conf.name
    bp = conf.build()

    # ======================   LOAD DATASET   =================================

    print('Loading dataset...')

    train = bp.data.train_vars(bp.paths)

    dev = bp.data.dev_vars(bp.paths)

    # ==========================   TRAIN   ====================================

    print('Fitting model...')

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #     else:
    #         print('No grad: %s' % name)
    if conf.get('continue_training', None):
        model_path = bp.paths.model_path
        print('Loading model from %s' % model_path)
        model = bp.model.load(model_path)
    else:
        model = bp.model

    if conf.device.startswith('cuda'):
        # Set default gpu
        model.to(conf.device)

    # NOTE: This needs to be done after moving to gpu
    model.setup_optimizer()
    # NOTE: Hack to access vocab encoder during training
    # To be able to use BIOAccuracy
    model.metrics = bp.data.metrics
    model.vocabs = bp.data.vocab_encoder.vocabs

    for task in bp.model.model.tasks:
        bp.model.model.tasks[task].label_vocab = bp.data.vocab_encoder.vocabs[task]
    # print(bp)
    model.fit(train, dev=dev)

    # ==========================   EVAL   =====================================

    if bp.verbose:
        # Train eval
        train_preds = model.predict(train)

        dec_train_preds = bp.data.decode(train_preds)
        dec_train = bp.data.decode(train)

        print('=== Eval Train ===')
        train_labels = eval_loop(dec_train, dec_train_preds)

        # Dev eval
        dev_preds = model.predict(dev)

        dec_dev_preds = bp.data.decode(dev_preds)
        dec_dev = bp.data.decode(dev)

        print('=== Eval Dev ===')
        dev_labels = eval_loop(dec_dev, dec_dev_preds)

        # assert train_labels == dev_labels

    # ======================   SAVE MODEL TO FILE   ===========================

    # Update conf details
    conf.edien_git_hash = get_current_git_hash()
    train_metrics = getattr(model, 'train_metrics', None)
    dev_metrics = getattr(model, 'dev_metrics', None)
    best_dev_metrics = getattr(model, 'best_dev_metrics', None)
    best_dev_metrics_time = getattr(model, 'best_dev_metrics_time', None)
    conf.results = dict(train=train_metrics,
                        dev=dev_metrics,
                        best_dev=best_dev_metrics,
                        best_dev_time=best_dev_metrics_time)

    if model.persist:
        save_path = bp.paths.model_path
        print('Saving model to %s' % save_path)
        model.save(save_path)

        save_path = bp.paths.blueprint_path
        print('Saving blueprint to %s' % save_path)
        # Save unbuilt config as the blueprint used for this experiment
        conf.to_file(save_path)
    else:
        print('Not saving - running in non-persist mode')

    return conf
