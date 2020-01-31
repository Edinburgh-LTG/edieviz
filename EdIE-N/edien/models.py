import os
import joblib
import numpy as np
import torch
from itertools import chain
from collections import namedtuple, defaultdict
from tqdm import tqdm

from edien.vocab import Vocab
from edien.preprocess import PaddedVariabliser


# Probably a good idea if the general API for returned types is namedtuple.
# For the time being, namedtuple returned will include (entities, modifiers)


def batch_to_X_y(batch, inputs, targets):
    Xnt = namedtuple('X', inputs)
    ynt = namedtuple('y', targets)

    X = Xnt(*(getattr(batch, attr) for attr in inputs))
    # We replace labels with None if they don't exist
    # since when evaluating on new data we won't have labels
    y = ynt(*(getattr(batch, attr, None) for attr in targets))

    return X, y


class EdFactorN(torch.nn.Module):
    """Multitasking feedforward net"""
    def __init__(self, model, optimizer, **kwargs):
        super(EdFactorN, self).__init__()

        self.optimizer = optimizer

        # What eval metric to look at to check if we should renew patience
        self.persist = True
        self.patience_renewer = None
        self.max_epochs = np.inf
        self.max_steps = np.inf
        self.scheduler = None
        self.min_lr = 1e-5
        self.do_eval = True

        self.model = model

        for k, v in sorted(kwargs.items(), key=lambda x: x[0]):
            setattr(self, k, v)
        self.stats = None
        self.targets = self.model.targets
        self.inputs = self.model.inputs

    def encode(self, X):
        # Convert to namedtuple of numpy array
        # useful for shuffling data among other things
        # X dims: (input_type, num_examples, seq_len)
        X = X.__class__(*[np.array(each)
                          if each is not None
                          else None
                          for each in X])
        return X

    def shuffle_inplace(self, data):
        # Shuffles in place
        # We assume number of examples is on index 0
        num_examples = data[0].shape[0]
        rng_state = np.random.get_state()
        # Shuffle in place using rng_state to make sure we get same
        # permutation for all "variables"
        for col in data:
            assert col.shape[0] == num_examples
            np.random.set_state(rng_state)
            np.random.shuffle(col)
        np.random.set_state(rng_state)

    def setup_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight', '.crf.', 'log_gamma', 'beta']
        decay_parameters = [(n, p) for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)]
        no_decay_parameters = [(n, p) for n, p in self.model.named_parameters()
                               if any(nd in n for nd in no_decay)]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_parameters]
             },
            {'params': [p for n, p in no_decay_parameters],
             'weight_decay': 0.0}
        ]
        self.optimizer = self.optimizer(optimizer_grouped_parameters)
        self.scheduler = self.scheduler(self.optimizer)

    def fresh_metrics(self):
        # Metrics
        fields = tuple(self.metrics.keys())
        computation = (m.__class__(f)
                       for m, f in zip(self.metrics.values(), fields))
        metrics = namedtuple('Metrics', self.metrics.keys())(*computation)
        return metrics

    def _run_epoch(self, train, dev):
        # We assume number of examples is on index 0
        num_examples = train[0].shape[0]
        self.shuffle_inplace(train)

        pbar = tqdm(total=num_examples,
                    leave=False,
                    bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
                    )
        for i in range(0, num_examples, self.batch_size):
            # Do not use "incomplete" batches
            if i + self.batch_size > num_examples:
                pbar.update(batch_len)
                break
            self.train()
            # Create batches and wrap arrays into chainer Variables
            batch = train.__class__(*(each[i:i+self.batch_size]
                                      for each in train))

            self.model.zero_loss()
            self.scheduler.optimizer.zero_grad()

            preds = self.model(batch)

            self.model.loss.backward()
            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.scheduler.optimizer.step()

            self.running_metrics.loss(float(self.model.loss))
            for field in self.targets:
                vocab = getattr(self.vocabs, field)
                field_gold = vocab.decode(getattr(batch, field))
                field_preds = vocab.decode(getattr(preds, field))
                getattr(self.running_metrics, field)(field_gold, field_preds)
            self.model.zero_loss()

            batch_len = batch[0].shape[0]
            desc = ' '.join('%s: %.5f' % (m.label.replace('_tags', ''), m.score)
                            if m.label.endswith('loss')
                            else
                            '%s: %.2f' % (m.label.replace('_tags', ''), m.score)
                            for m in self.running_metrics)
            desc = 'train: %s' % desc
            pbar.set_description(desc, refresh=False)
            current_learning_rate = tuple(self.scheduler.optimizer.param_groups)[0]['lr']
            pbar.set_postfix(epoch=self.num_epochs,
                             patience=self.patience_left,
                             lr=current_learning_rate)
            pbar.update(batch_len)
            self.batches_processed += 1
            self.scheduler.step()

            if self.batches_processed % self.checkpoint_every == 0:
                if self.do_eval:

                    dev_metrics = self.fresh_metrics()
                    dev_preds = self.predict(dev)

                    dev_loss = float(self.model.loss)
                    # use negative loss since for all other metrics we want to
                    # are are seeking to maximise the value: loss breaks interface
                    dev_metrics.loss(-dev_loss)

                    for field in self.targets:
                        vocab = getattr(self.vocabs, field)
                        dev_field_gold = vocab.decode(getattr(dev, field))
                        dev_field_preds = vocab.decode(getattr(dev_preds, field))
                        getattr(dev_metrics, field)(dev_field_gold, dev_field_preds)

                    dev_metric_labels = [m.label for m in dev_metrics]
                    dev_nt = namedtuple('TRes', dev_metric_labels)
                    values = (float(round(m.score, 4)) for m in dev_metrics)
                    dev_metric_row = dev_nt(*values)
                    self.dev_metrics.append(dev_metric_row)

                    # Update best scores and patience
                    for ml in dev_metric_labels:
                        assert(self.patience_renewer in dev_metric_labels)
                        cur_val = getattr(dev_metric_row, ml)
                        cur_best = self.best_dev_metrics[ml]
                        if cur_val > cur_best:
                            self.best_dev_metrics[ml] = cur_val
                            cur_ckp = int(self.batches_processed / self.checkpoint_every)
                            self.best_dev_metrics_time[ml] = cur_ckp
                            if ml == self.patience_renewer:
                                self.patience_left = self.patience
                                if self.persist:
                                    self.save('/tmp/model.ckpt')
                        else:
                            if ml == self.patience_renewer:
                                self.patience_left -= 1

                    # Update dev progress bar
                    dev_pbar = tqdm(total=1,
                                    position=1,
                                    bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}{postfix}')
                    dev_desc = ' '.join('%s: %.5f' % (m.label.replace('_tags', ''), abs(m.score))
                                        if m.label.endswith('loss')
                                        else
                                        '%s: %.2f' % (m.label.replace('_tags', ''), m.score)
                                        for m in dev_metrics)
                    dev_desc = ' dev : %s' % dev_desc
                    dev_pbar.set_description(dev_desc)
                    dev_pbar.update(1)
                    dev_pbar.close()

                    # Update dev progress bar
                    best_pbar = tqdm(total=1,
                                     position=2,
                                     bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}{postfix}')
                    best_desc = ' '.join('%s: %.5f' % (k.replace('_tags', ''), abs(v))
                                         if k.endswith('loss')
                                         else
                                         '%s: %.2f' % (k.replace('_tags', ''), v)
                                         for k, v in self.best_dev_metrics.items())
                    best_desc = 'best : %s' % best_desc
                    best_pbar.set_description(best_desc)
                    best_pbar.update(1)
                    best_pbar.close()

                train_nt = namedtuple('TRes', [m.label for m in self.running_metrics])
                values = (float(round(m.score, 4)) for m in self.running_metrics)
                self.train_metrics.append(train_nt(*values))

                self.running_metrics = self.fresh_metrics()
                self.num_steps += 1

                if self.patience_left == 0 \
                        or self.num_epochs >= self.max_epochs \
                        or self.num_steps >= self.max_steps \
                        or current_learning_rate == 0.:
                    self.patience_left = 0
                    # No improvement / let's lower learning rate
                    if self.persist and self.do_eval:
                        self.load('/tmp/model.ckpt')
                        os.remove('/tmp/model.ckpt')
                    break

        epoch_time = pbar._time() - pbar.start_t
        self.num_epochs += 1
        self.train_time += epoch_time
        pbar.close()

    def fit(self, train, dev):
        """X: namedtuple

        :x: namedtuple of inputs
        :y: namedtuple of targets (we do multiple tasks)
        :returns: None - trains the models in place

        """
        self.patience_left = self.patience
        self.train_metrics = []
        self.dev_metrics = []
        self.best_dev_metrics = defaultdict(lambda: -np.inf)
        self.best_dev_metrics_time = defaultdict(lambda: -np.inf)
        if self.patience_renewer is None:
            self.patience_renewer = 'avg_loss'
        self.num_epochs = 0
        self.num_steps = 0
        self.batches_processed = 0
        self.train_time = 0
        self.running_metrics = self.fresh_metrics()

        train = self.encode(train)
        dev = self.encode(dev)

        while(self.patience_left > 0):
            self._run_epoch(train, dev)
        print('\n' * 3)
        print('Took %.2f seconds to train for %d epochs, %d num_steps'
              '(can use as max_steps)' %
              (self.train_time, self.num_epochs, self.num_steps))

        fields = self.train_metrics[0]._fields
        self.train_metrics = dict(zip(fields, zip(*self.train_metrics)))
        if self.do_eval:
            self.dev_metrics = dict(zip(fields, zip(*self.dev_metrics)))

        return self

    def predict(self, X):
        """X: namedtuple

        :x: namedtuple of inputs
        :returns: namedtuple of targets

        """
        # Nothing complex - just concat features together
        # TODO: potential
        X = self.encode(X)
        all_nts = []
        self.eval()

        with torch.no_grad():
            num_examples = X[0].shape[0]
            batch_size = min(num_examples, self.batch_size)
            num_batches = 0
            for i in range(0, num_examples, batch_size):
                num_batches += 1
                # Create batches
                batch = X.__class__(*(each[i:i+batch_size]
                                      for each in X))
                ys = self.model(batch)
                all_nts.append(ys)
            # Average out effect of computing loss over multiple batches
            self.model.loss /= num_batches
        # We have already moved to cpu in predict
        # ys = [np.hstack(stat_rows) for stat_rows in zip(*all_nts)]
        ys = [tuple(chain(*stat_rows)) for stat_rows in zip(*all_nts)]
        ys = self.model.return_nt(*ys)
        return ys

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
        return self


class EdFFMT(torch.nn.Module):

    def __init__(self,
                 encoder,
                 tasks,
                 num_samples=0,
                 device='cpu'):
        super(EdFFMT, self).__init__()
        assert num_samples >= 0
        # We assume below models have sklearn style interfaces
        self.encoder = encoder
        self.tasks = tasks
        for k, v in self.tasks.items():
            self.set_task_model(k, v)

        self.targets = tuple(tasks.keys())
        self.inputs = self.encoder.inputs
        self.return_nt = namedtuple('MTPreds', self.targets)
        # if num_samples is zero, standard dev eval is used
        # if num_samples is > zero, MC dropout is used with
        # num_samples samples #obvious
        self.num_samples = num_samples
        self.device = device
        self.loss = 0.

    def get_task_model(self, task):
        return getattr(self, '%s_model' % task)

    def set_task_model(self, task, model):
        setattr(self, '%s_model' % task, model)

    def zero_loss(self):
        self.loss = 0.
        for task in self.targets:
            self.get_task_model(task).zero_loss()

    def forward(self, batch):

        data_encoder = PaddedVariabliser(device=self.device)

        # Pad and Variablise
        batch = data_encoder.encode(batch)

        # NOTE: we assume sentence lengths of all outputs are the same
        sentence_lengths = data_encoder.sequence_lengths[self.targets[0]]

        if self.training:
            preds = self.compute(batch, sentence_lengths)
        else:
            # If num_samples is greater than zero we use MC DROPOUT
            # else we use default eval.
            if self.num_samples > 0:
                preds = self.mc_dropout(batch, sentence_lengths)
            else:
                preds = self.compute(batch, sentence_lengths)

        preds = self.return_nt(*preds)

        # Unpad and Unvariablise the predictions
        preds = data_encoder.decode(preds)

        return preds

    def compute(self, batch, sentence_lengths, predict_proba=False):

        X, y = batch_to_X_y(batch, self.inputs, self.targets)

        act = self.encoder(X, sentence_lengths)

        batch_data = batch._asdict()

        preds = []
        for task in self.targets:

            task_model = self.get_task_model(task)
            # Pick out the correct data to be predicted
            yy = getattr(batch, task, None)
            pred = task_model(act,
                              yy,
                              sentence_lengths,
                              batch_data=batch_data,
                              predict_proba=predict_proba)

            pred_field = task_model.get_piped_preds_name(task)
            batch_data[pred_field] = pred
            preds.append(pred)

            self.loss += task_model.loss
            task_model.zero_loss()

        return preds

    def mc_dropout(self, batch, sentence_lengths):
        # Remember RNG state
        TORCH_RNG_STATE = torch.get_rng_state()
        TORCH_GPU_STATES_GPU = torch.cuda.get_rng_state_all()
        torch.manual_seed(42)

        # Make dropout modules work in sampling mode in
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.training = True
            elif m.__class__.__name__.startswith('AugmentedLSTM'):
                m.training = True
            elif m.__class__.__name__.startswith('StackedBidirectional'):
                m.training = True

        run_samples = []
        for i in range(self.num_samples):
            preds = self.compute(batch, sentence_lengths, predict_proba=True)
            run_samples.append(preds)
            # TODO: average probability predictions across batches

        # self.loss /= self.num_samples

        preds = []
        self.mc_mean, self.mc_std, self.mc_samples = dict(), dict(), dict()
        for task_pred, task in zip(zip(*run_samples), self.targets):
            # concatenate predictions along a new first axis.
            # dim: num_samples x batch_size x seq_length x stuff..
            stacked_samples = torch.stack(task_pred)
            mean_samples = torch.mean(stacked_samples, dim=0)
            std_samples = torch.std(stacked_samples, dim=0)
            if self.tasks[task].num_labels > 1:
                pred = torch.argmax(mean_samples, dim=2)
            else:
                pred = (mean_samples > .5).squeeze().long()
            # print(pred.shape)
            preds.append(pred)

            self.mc_samples[task] = stacked_samples.cpu().numpy()
            self.mc_mean[task] = mean_samples.cpu().numpy()
            self.mc_std[task] = std_samples.cpu().numpy()

        # NOTE sets RNG state from before
        torch.cuda.set_rng_state_all(TORCH_GPU_STATES_GPU)
        torch.set_rng_state(TORCH_RNG_STATE)

        return preds

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        try:
            self.load_state_dict(torch.load(path))
        except RuntimeError as e:
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
