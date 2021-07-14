import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import namedtuple, Counter
from matplotlib import collections as mc

from mlconf import YAMLLoaderAction, ArgumentParser
from edien.components import BertSentenceEncoder
from edien.preprocess import PaddedVariabliser
from edien.train_utils import only_pseudorandomness_please
from edien.vocab import BertCoder
from utils import FaissIndex, k_nearest_interpolation, entropy, argmax, pprint


def plot_uncertainty(ax, axb, tokens, targets, probs):

    assert len(tokens) == len(targets)
    assert len(targets) == len(probs)

    vocab = {'O': 0}
    all_keys = set()
    for p in probs:
        all_keys.update([k.replace('_', ' ') for k in p.keys()])
    if 'O' in all_keys:
        all_keys.remove('O')
    vocab.update(zip(sorted(all_keys, key=lambda x: x.split('-')[::-1]),
                     range(1, len(all_keys) + 1)))
    idx_to_vocab = {v: k for k, v in vocab.items()}

    num_labels = len(vocab)
    seq_len = len(tokens)
    entropies = [entropy(p) for p in probs]

    x_ax = np.arange(seq_len)
    y_ax = np.arange(num_labels)
    ax.set_ylabel('kNN Findings')

    ax.set_ylim([-.5, num_labels])
    ax.set_xlim([-.5, seq_len])
    axb.set_xlim([-.5, seq_len])
    axb.set_ylabel('kNN Entropy')
    axb.set_xlabel('Test Sentence')

    ax.set_xticks([])
    ax.set_yticks(y_ax)
    ax.set_yticklabels(tuple(vocab.keys()))
    axb.set_xticks(x_ax)
    axb.set_xticklabels(tokens, fontsize=14, ha='right', rotation=50)

    lines = []
    colors = []
    for i in x_ax[1:]:
        for j in y_ax:
            for k in y_ax:
                p_ij = probs[i-1].get(idx_to_vocab[j].replace(' ', '_'), 0.0)
                p_ik = probs[i].get(idx_to_vocab[k].replace(' ', '_'), 0.0)
                pp = p_ij * p_ik
                if pp > 0.:
                    color = (.4, 0, .6, pp)
                    lines.append(((i-1, j), (i, k)))
                    colors.append(color)
                    ax.text(i-1, j + .1, idx_to_vocab[j].replace(' ', '\n'),
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            color=color,
                            alpha=pp)
    lc = mc.LineCollection(lines, colors=colors, linewidths=.5)
    ax.add_collection(lc)

    axb.plot(entropies, lw=.5)

    xx, yy = np.meshgrid(x_ax, y_ax)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    return ax, axb


class Sentence(object):

    def __init__(self, tokens, pos, gold, preds):
        super(Sentence, self).__init__()
        self.tokens = tokens
        self.pos = pos
        self.gold = gold
        self.preds = preds


def type_(token):
    return token[2:] or 'O'


def changed(prv, nxt):
    return type_(prv) != type_(nxt) or nxt[:1] == 'B'


def count_ent_spans(tokens):
    count = 0
    prev = 'O'
    for t in tokens:
        if t != 'O' and changed(prev, t):
            count += 1
        prev = t
    return count


def has_label_errors(pred, gold):
    err = False
    for p, g in zip(pred, gold):
        if p == 'O' or g == 'O':
            continue
        if p != g:
            err = True
            break
    return err


def classify_segment(pred, gold):
    assert len(pred) == len(gold)
    assert len(pred) > 0
    pt = [type_(p) for p in pred]
    gt = [type_(g) for g in gold]

    num_pred_ents = count_ent_spans(pred)
    num_gold_ents = count_ent_spans(gold)

    pl = set(pt)
    gl = set(gt)

    cls, TP, TN, FP, FN = None, 0, 0, 0, 0
    if pl == {'O'} and gl == {'O'}:
        cls = 'TN'
        TN += 1
    elif pl == {'O'} and gl != {'O'}:
        cls = 'FN'
        FN += 1
    elif pl != {'O'} and gl == {'O'}:
        cls = 'FP'
        FP += 1
    # Correct if there is:
    # i)   a single type shared across gold and pred
    # ii)  there is a single entity in gold and pred
    elif pl == gl and 'O' not in pl and num_gold_ents == 1 and num_pred_ents == 1:
        cls = 'TP'
        TP += 1
    # label error
    # i)  there is a single entity in gold and pred
    # ii) there is no O prediction
    elif 'O' not in pl and 'O' not in gl and pl != gl and num_gold_ents == 1 and num_pred_ents == 1:
        cls = 'LABEL_ERROR'
        FP += 1
        FN += 1
    elif (pl - {'O'}) == (gl - {'O'}) and not has_label_errors(pt, gt):
        cls = 'BOUNDARY_ERROR'
        FP += num_pred_ents
        FN += num_gold_ents
    else:
        # This can be broken down into multiple FP and FN
        cls = 'LABEL+BOUNDARY_ERROR'
        FP += num_pred_ents
        FN += num_gold_ents
    return cls, TP, TN, FP, FN


if __name__ == "__main__":
    parser = ArgumentParser(description='Visualise label entropy and print out '
                            'K most similar sentences to those in the conll '
                            'file using KNN with faiss index.')
    parser.add_argument('--K', default=10, type=str,
                        help='The number of nearest neighbours to use')
    parser.add_argument('--load_index', required=True, type=str,
                        help='The folder to load the FAISS index from')
    parser.add_argument('--conll_file',
                        type=str,
                        required=True,
                        help='Path to conll file')
    parser.add_argument('--load_blueprint', action=YAMLLoaderAction)

    conf = parser.parse_args()
    only_pseudorandomness_please(conf.seed)

    # Make sure we aren't trying to create vocab on test
    conf.data.vocab_encoder.load_if_exists = True

    print('Loading model specified in "%s"' % conf.load_blueprint)
    bp = conf.build()
    tokenizer = bp.data.vocab_encoder.vocabs.tokens.fit([])
    word_masker = bp.data.vocab_encoder.vocabs.word_mask.fit([])

    # Load faiss index.
    print('Loading faiss index from "%s"' % conf.load_index)
    index = FaissIndex.load(conf.load_index)
    print('Using %d nearest neighbours' % conf.K)
    NUM_EXAMPLES = 5

    Entry = namedtuple('Entry', ['tokens', 'word_mask', 'output'])
    Output = namedtuple('Output', ['output'])

    # Load CONLL file
    with open(conf.conll_file, 'r') as f:
        sentences, parts = [], []
        for line in f.readlines():
            line = line.rstrip()
            if line:
                parts.append(line.split(' '))
            else:
                sent = Sentence(*zip(*parts))
                sentences.append(sent)
                parts = []
        if parts:
            sent = Sentence(*zip(*parts))
            sentences.append(sent)

    TPs, TNs, FPs, FNs = 0, 0, 0, 0
    segment_classes = []
    for s in sentences:

        # Split into chunks following Manning's clean explanation
        # https://nlpers.blogspot.com/2006/08/doing-named-entity-recognition-dont.html
        pred_chunks, gold_chunks, token_chunks = [], [], []
        prev_pred, prev_gold = s.preds[0], s.gold[0]
        # I. beginning of sentences
        prev_i = 0
        assert len(s.preds) == len(s.gold)
        sstr = '***   Sentence: ' + ' '.join(s.tokens) + '   ***'
        print('*' * len(sstr))
        print(sstr)
        print('*' * len(sstr))
        print()
        for i, (p, g) in enumerate(zip(s.preds[1:], s.gold[1:]), 1):
            # BIO typing
            pred_type = type_(p)
            gold_type = type_(g)
            # II.
            # anywhere there is a change to or from a word/O/O token from
            # or to a token where either guess or gold is not
            if ((prev_pred, prev_gold) == ('O', 'O')) \
                    and (((p == 'O') and (g != 'O')) or ((p != 'O') and (g == 'O'))):
                pred_chunks.append(s.preds[prev_i:i])
                gold_chunks.append(s.gold[prev_i:i])
                token_chunks.append(s.tokens[prev_i:i])
                prev_i = i
            elif ((p, g) == ('O', 'O')) \
                    and (((prev_pred == 'O') and (prev_gold != 'O')) or ((prev_pred != 'O') and (prev_gold == 'O'))):
                pred_chunks.append(s.preds[prev_i:i])
                gold_chunks.append(s.gold[prev_i:i])
                token_chunks.append(s.tokens[prev_i:i])
                prev_i = i
            # III. anywhere that both systems change their class assignment
            # simultaneously, regardless of whether they agree.
            elif changed(prev_gold, g) and changed(prev_pred, p):
                pred_chunks.append(s.preds[prev_i:i])
                gold_chunks.append(s.gold[prev_i:i])
                token_chunks.append(s.tokens[prev_i:i])
                prev_i = i
            # IV. This case is not mentioned in the post, we also need to split
            # when we have consecutive O's in pred or gold but change
            # predictions in the other sequence. NOTE that the case of
            # O, O and O, *  is already covered in ii)
            elif (prev_gold, g) == ('O', 'O') and changed(prev_pred, p):
                pred_chunks.append(s.preds[prev_i:i])
                gold_chunks.append(s.gold[prev_i:i])
                token_chunks.append(s.tokens[prev_i:i])
                prev_i = i
            elif (prev_pred, p) == ('O', 'O') and changed(prev_gold, g):
                pred_chunks.append(s.preds[prev_i:i])
                gold_chunks.append(s.gold[prev_i:i])
                token_chunks.append(s.tokens[prev_i:i])
                prev_i = i
            prev_pred = p
            prev_gold = g
        # I. continued. end of sentences
        pred_chunks.append(s.preds[prev_i:])
        gold_chunks.append(s.gold[prev_i:])
        token_chunks.append(s.tokens[prev_i:])

        cc = 0
        has_error = False
        for p, g, t in zip(pred_chunks, gold_chunks, token_chunks):
            cls, TP, TN, FP, FN = classify_segment(p, g)
            if cls[0] != 'T':
                has_error = True
                print('\t%s %s %s%r%r' % (' '.join(t), '->', cls, g, p))
            segment_classes.append(cls)
            TPs += TP
            TNs += TN
            FPs += FP
            FNs += FN
            cc += len(p)
        print()

        tokens = tokenizer.encode([s.tokens])
        mask = word_masker.encode([s.tokens])
        entry = Entry(tokens, mask, (tuple(range(len(s.tokens))),))

        data_encoder = PaddedVariabliser()
        data = data_encoder.encode(entry)
        sent_lens = data_encoder.sequence_lengths['output']

        outputs = bp.model.model.encoder(data, sent_lens=sent_lens)
        outputs = outputs.detach()

        output = Output(outputs)
        result = data_encoder.decode(output).output[0]

        # Get nearest neighbours and print entropies
        all_probs = []
        dists, idxs = index.index.search(result, k=conf.K)
        print('-----------  Per token Entropy calculation   -----------\n')
        for i, (dist, idx) in enumerate(zip(dists, idxs)):
            ner_labels = [index.sentence_lookup[match].ner_tags[index.label_lookup[match]] for match in idx]
            probs = k_nearest_interpolation(dist, ner_labels)
            all_probs.append(probs)
            # print(entropy(probs), probs)
            print('\t', s.tokens[i], s.gold[i], s.preds[i], '%.4f' % entropy(probs))
        print('\n--------------------------------------------------------\n')

        if has_error:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5), gridspec_kw=dict(height_ratios=[.7, .3]))
            plot_uncertainty(ax1, ax2, s.tokens, s.gold, all_probs)
            plt.tight_layout()
            plt.show()

        for i, (dist, idx) in enumerate(zip(dists, idxs)):
            if s.preds[i] != s.gold[i]:
                print()
                header = '###   %s: "%s" mispredicted as "%s"   ###' % (s.tokens[i], s.gold[i], s.preds[i])
                print('#' * len(header))
                print(header)
                print('#' * len(header))
                print()
                j = 1
                for d, match in zip(dist[:NUM_EXAMPLES], idx[:NUM_EXAMPLES]):
                    context = index.sentence_lookup[match]
                    print("### Nearest Neighbour %d: sent_id %s with distance %.2f ###" % (j, context.sent_id, d))
                    pprint(context.tokens, context.ner_tags)
                    print()
                    j += 1
        print()
        print('\n')
    prec = float(TPs) / (TPs + FPs) if TPs + FPs > 0 else 0.
    rec = float(TPs) / (TPs + FNs) if TPs + FNs > 0 else 0.
    F1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0. else 0.
    print('ALL: TP: %d; FP: %d; FN: %d' % (TPs, FPs, FNs))
    print('Prec : %.2f' % (100 * prec))
    print('Rec  : %.2f' % (100 * rec))
    print('F1   : %.2f' % (100 * F1))
    error_counts = Counter(segment_classes)
    print(json.dumps(error_counts))
