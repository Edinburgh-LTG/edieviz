import json
from collections import defaultdict, Counter
from argparse import ArgumentParser


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
    parser = ArgumentParser(description='Calculate and plot loose score')

    parser.add_argument('--conll_file',
                        type=str,
                        required=True,
                        help='Path to conll file')
    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help='Name results')

    args = parser.parse_args()

    # Load CONLL file
    with open(args.conll_file, 'r') as f:
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
        pred_chunks, gold_chunks = [], []
        prev_pred, prev_gold = s.preds[0], s.gold[0]
        # I. beginning of sentences
        prev_i = 0
        assert len(s.preds) == len(s.gold)
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
                prev_i = i
            elif ((p, g) == ('O', 'O')) \
                    and (((prev_pred == 'O') and (prev_gold != 'O')) or ((prev_pred != 'O') and (prev_gold == 'O'))):
                pred_chunks.append(s.preds[prev_i:i])
                gold_chunks.append(s.gold[prev_i:i])
                prev_i = i
            # III. anywhere that both systems change their class assignment
            # simultaneously, regardless of whether they agree.
            elif changed(prev_gold, g) and changed(prev_pred, p):
                pred_chunks.append(s.preds[prev_i:i])
                gold_chunks.append(s.gold[prev_i:i])
                prev_i = i
            # IV. This case is not mentioned in the post, we also need to split
            # when we have consecutive O's in pred or gold but change
            # predictions in the other sequence. NOTE that the case of
            # O, O and O, *  is already covered in ii)
            elif (prev_gold, g) == ('O', 'O') and changed(prev_pred, p):
                pred_chunks.append(s.preds[prev_i:i])
                gold_chunks.append(s.gold[prev_i:i])
                prev_i = i
            elif (prev_pred, p) == ('O', 'O') and changed(prev_gold, g):
                pred_chunks.append(s.preds[prev_i:i])
                gold_chunks.append(s.gold[prev_i:i])
                prev_i = i
            prev_pred = p
            prev_gold = g
        # I. continued. end of sentences
        pred_chunks.append(s.preds[prev_i:])
        gold_chunks.append(s.gold[prev_i:])

        cc = 0
        for p, g in zip(pred_chunks, gold_chunks):
            cls, TP, TN, FP, FN = classify_segment(p, g)
            segment_classes.append(cls)
            # print(g, p, cls, s.tokens[cc: cc + len(p)])
            TPs += TP
            TNs += TN
            FPs += FP
            FNs += FN
            cc += len(p)
    prec = float(TPs) / (TPs + FPs) if TPs + FPs > 0 else 0.
    rec = float(TPs) / (TPs + FNs) if TPs + FNs > 0 else 0.
    F1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0. else 0.
    # The numbers below should match CONLL eval
    print('ALL: TP: %d; FP: %d; FN: %d' % (TPs, FPs, FNs))
    print('Prec : %.2f' % (100 * prec))
    print('Rec  : %.2f' % (100 * rec))
    print('F1   : %.2f' % (100 * F1))
    error_counts = Counter(segment_classes)
    # The numbers below are the breakdown of errors
    # each label error counts as 2 errors (a FP + FN)
    # and each boundary and boundary + label error counts as at least 2 errors
    # in the CONLL score
    print(json.dumps([args.name, error_counts]))
