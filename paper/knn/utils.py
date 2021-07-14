import os
import faiss
import pickle
import numpy as np

from collections import namedtuple, defaultdict
from scipy.special import logsumexp


# https://arxiv.org/pdf/1911.00172.pdf
# Equation 2 - P_{knn}
def k_nearest_interpolation(dists, labels):
    assert len(dists) == len(labels)
    scores = defaultdict(list)
    # We use logsumexp to avoid overflow
    # http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

    # Softmax defn for label y_i and scores s_i (here neg dist):
    # P(y_i) = e^{s_i} / (\sum_j e^{s_j})

    # However in our case, the score for a single label i
    # may be a sum of terms if multiple returned results have this label
    # e.g. may be P(y_a) = e^{s_1} + e^{s_2} / (e^{s_1} + e^{s_2} + e^{s_3})

    # To ameliorate overflow issues, use logs + logsumexp
    # log(P(y_i)) = log(numerator / denom)
    #             = log(numerator) - log(denom)
    #             = logsumexp(numerator_scores) - logsumexp(all_scores)

    # So to get probs we need only exp the result above.
    all_scores = []
    for d, l in zip(dists, labels):
        # Parse to float64 for increased accuracy
        d = np.float64(d)
        scores[l].append(-d)
        all_scores.append(-d)
    # Compute logsumexp of denominator (sum of all class activations)
    denom = logsumexp(all_scores)
    # NOTE: as mentioned above,
    # we use logsumexp for the numerator as well, since we are
    # aggregating across labels. E.g. if label: B-Tumour occurs
    # multiple times, we need to sum across all activations for that label.
    # As in Equation 2 of the KNN Language Models.
    probs = {l: np.exp(logsumexp(scores[l]) - denom) for l in set(labels)}
    return probs


def entropy(probs):
    return max(0., -sum(p * np.log(p) for p in probs.values() if p > 0.))


def argmax(probs):
    argmax = None
    best = -np.inf
    for k, v in probs.items():
        if v > best:
            best = v
            argmax = k
    return argmax


def pprint(*seqs):
    max_seq_len = []
    for step in zip(*seqs):
        max_len = max(map(len, step))
        max_seq_len.append(max_len)

    for seq in seqs:
        print(' '.join(w.center(max(8, max_seq_len[i]))
                       for i, w in enumerate(seq)))


class FaissIndex(object):
    """Interface for knn neighbours"""

    FAISS_FILE = 'index.bin'
    LABELS_FILE = 'labels.pkl'
    SENTS_FILE = 'senteces.pkl'

    def __init__(self, dims):
        super(FaissIndex, self).__init__()
        # Init constructs the object
        self.index = faiss.IndexFlatL2(dims)
        self.label_lookup = dict()
        self.sentence_lookup = dict()
        self.index_size = 0

    def __len__(self):
        return self.index_size

    def add(self, embeds, labels):
        num_labels = len(labels)
        ids = list(range(self.index_size, self.index_size + num_labels))
        self.index.add(embeds)
        self.label_lookup.update(dict(zip(ids, range(len(labels.tokens)))))
        self.sentence_lookup.update(dict(zip(ids, [labels for i in ids])))
        self.index_size += num_labels

    def save(self, folderpath):
        if not os.path.isdir(folderpath):
            print('Creating faiss index directory %s' % folderpath)
            os.makedirs(folderpath)
        faiss_file = os.path.join(folderpath, FaissIndex.FAISS_FILE)
        faiss.write_index(self.index, faiss_file)

        labels_file = os.path.join(folderpath, FaissIndex.LABELS_FILE)
        with open(labels_file, 'wb') as f:
            pickle.dump(self.label_lookup, f)

        sents_file = os.path.join(folderpath, FaissIndex.SENTS_FILE)
        with open(sents_file, 'wb') as f:
            pickle.dump(self.sentence_lookup, f)

    @classmethod
    def load(cl, folderpath):

        faiss_file = os.path.join(folderpath, FaissIndex.FAISS_FILE)
        index = faiss.read_index(faiss_file)

        labels_file = os.path.join(folderpath, FaissIndex.LABELS_FILE)
        with open(labels_file, 'rb') as f:
            label_lookup = pickle.load(f)

        sents_file = os.path.join(folderpath, FaissIndex.SENTS_FILE)
        with open(sents_file, 'rb') as f:
            sentence_lookup = pickle.load(f)

        obj = cl.__new__(cl)
        obj.index = index
        obj.label_lookup = label_lookup
        obj.sentence_lookup = sentence_lookup
        obj.index_size = len(label_lookup)
        return obj
