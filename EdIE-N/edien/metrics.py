import re
import numpy as np
from collections import defaultdict
from itertools import chain
from edien.vocab import Vocab


class Average(object):

    def __init__(self, label=None):
        self.count = 0
        self.cumsum = 0.
        self._label = label

    @property
    def label(self):
        return ['avg_%s' % self._label, 'avg'][self._label is None]

    def __call__(self, value):
        self.count += 1
        self.cumsum += value
        return self.score

    def reset(self):
        self.count = 0
        self.cumsum = 0.

    @property
    def score(self):
        return self.cumsum / self.count if self.count else 0.0


class Accuracy(Average):

    '''Accuracy Score - Scorer'''

    def __init__(self, label=None):
        super(Accuracy, self).__init__(label)

    @property
    def min_val(self):
        return 0.

    @property
    def label(self):
        return ['%s_acc' % self._label, 'acc'][self._label is None]

    def __call__(self, true_tags, pred_tags):
        # truth = np.array(tuple(chain.from_iterable(true_tags)))
        # preds = np.array(tuple(chain.from_iterable(pred_tags)))
        truth = np.array(true_tags)
        preds = np.array(pred_tags)

        correct = float(np.sum(preds == truth))
        self.count += len(preds)
        self.cumsum += correct
        return self.cumsum / self.count if self.count else 0.0

    @property
    def score(self):
        return super(Accuracy, self).score * 100


class F1(object):

    '''F1 Score - Scorer'''

    def __init__(self, label=None):
        self._label = label
        self.confusion_matrix = defaultdict(lambda: defaultdict(lambda: 0))
        self.max_label_seen = 0
        super(F1, self).__init__()

    @property
    def min_val(self):
        return 0.

    @property
    def label(self):
        return ['%s_f1' % self._label, 'f1'][self._label is None]

    def __call__(self, true_tags, pred_tags):
        # truth = np.array(tuple(chain.from_iterable(true_tags)))
        # preds = np.array(tuple(chain.from_iterable(pred_tags)))
        truth = np.array(true_tags)
        preds = np.array(pred_tags)

        for t, p in zip(truth, preds):
            self.confusion_matrix[p][t] += 1

            if t > self.max_label_seen:
                self.max_label_seen = t

            if p > self.max_label_seen:
                self.max_label_seen = p

        return self.score

    @property
    def cm(self):
        # True positives for each class are on the diagonal
        return np.array([[self.confusion_matrix[i][j]
                          for j in range(self.max_label_seen + 1)]
                         for i in range(self.max_label_seen + 1)],
                        dtype=np.int32)

    @property
    def tp(self):
        # True positives for each class are on the diagonal
        return np.array([self.confusion_matrix[i][i]
                         for i in range(self.max_label_seen + 1)])

    @property
    def fn(self):
        sum_rows = np.array([np.sum([self.confusion_matrix[i][j]
                                     for j in range(self.max_label_seen + 1)])
                             for i in range(self.max_label_seen + 1)])
        # subtract diagonal from rows
        return sum_rows - self.tp

    @property
    def fp(self):
        sum_cols = np.array([np.sum([self.confusion_matrix[i][j]
                                     for i in range(self.max_label_seen + 1)])
                             for j in range(self.max_label_seen + 1)])
        # subtract diagonal from rows
        return sum_cols - self.tp

    @property
    def precision(self):
        tp = self.tp
        # Follow sklearn code base and set infs to zero
        denom = tp + self.fp
        prec = tp/denom
        prec[denom == 0.] = 0.
        return prec

    @property
    def recall(self):
        tp = self.tp
        denom = tp + self.fn
        rec = tp/denom
        rec[denom == 0.] = 0.
        return rec

    @property
    def score(self):
        prec = self.precision
        rec = self.recall
        denom = prec + rec
        f1 = 2 * (prec * rec) / denom
        f1[denom == 0.] = 0.
        return f1.mean() * 100

    def reset(self):
        self.confusion_matrix = defaultdict(lambda: defaultdict(lambda: 0))


# Below code stolen and adapted from: https://github.com/spyysalo/conlleval.py/blob/master/conlleval.py
class BIOF1(object):

    '''Accuracy Score - Scorer'''

    def __init__(self, label=None):
        super(BIOF1, self).__init__()
        self.tp = 0
        self.pred = 0.
        self.true = 0.
        self._label = label

    @property
    def label(self):
        return ['%s_f1' % self._label, 'f1'][self._label is None]

    @property
    def min_val(self):
        return 0.

    def reset(self):
        self.tp = 0
        self.pred = 0.
        self.true = 0.

    def __call__(self, true_tags, pred_tags):
        # NOTE: ignore values such as padding are negative.
        # We remove them to avoid taking them into account

        correct_chunk, found_correct, found_guessed = 0, 0, 0
        for t_sent, p_sent in zip(true_tags, pred_tags):
            assert(len(t_sent) == len(p_sent))

            in_correct = False        # currently processed chunks is correct until now
            last_correct = 'O'        # previous chunk tag in corpus
            last_correct_type = ''    # type of previously identified chunk tag
            last_guessed = 'O'        # previously identified chunk tag
            last_guessed_type = ''    # type of previous chunk tag in corpus
            for t_label, p_label in zip(t_sent, p_sent):
                if p_label is None:
                    raise ValueError("Predicting a label that doesn't exist")
                if t_label is None:
                    raise ValueError("Vocabulary not covering all possible labels")
                correct, correct_type = parse_tag(t_label)
                guessed, guessed_type = parse_tag(p_label)

                end_correct = end_of_chunk(last_correct, correct, last_correct_type, correct_type)
                end_guessed = end_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)
                start_correct = start_of_chunk(last_correct, correct, last_correct_type, correct_type)
                start_guessed = start_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)

                if in_correct:
                    if (end_correct and end_guessed and
                        last_guessed_type == last_correct_type):
                        in_correct = False
                        correct_chunk += 1
                    elif (end_correct != end_guessed or guessed_type != correct_type):
                        in_correct = False

                if start_correct and start_guessed and guessed_type == correct_type:
                    in_correct = True

                if start_correct:
                    found_correct += 1
                if start_guessed:
                    found_guessed += 1
                # if first_item != options.boundary:
                #     if correct == guessed and guessed_type == correct_type:
                #         counts.correct_tags += 1
                #     counts.token_counter += 1

                last_guessed = guessed
                last_correct = correct
                last_guessed_type = guessed_type
                last_correct_type = correct_type

        if in_correct:
            correct_chunk += 1

        self.tp += correct_chunk
        self.pred += found_guessed
        self.true += found_correct

        return self.score

    @property
    def score(self):
        prec = self.tp / self.pred if self.pred > 0. else 0.
        rec = self.tp / self.true if self.true > 0. else 0.
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.
        return f1 * 100


def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start
