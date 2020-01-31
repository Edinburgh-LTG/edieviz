import os
import six
import yaml
import numpy as np
from shutil import copyfile
from collections import Counter, defaultdict
from edien.utils import deep_unpack, deep_apply


# Below classes are assumed to have:
# fit, encode, decode, save and load
class Vocab(object):
    """The tokens we know. Class defines a way to create the vocabulary
    and assign each known token to an index. All other tokens are replaced
    with the token UNK, which of course is UNK following the definition
    of Dr. UNK UNK from UNK.
    UNK is assigned the token 0 - because we like being arbitrary.
    The rest of the known tokens are sorted by frequency and assigned indices
    in such a manner.

    We keep the number of counts in order to be able to update our
    vocabulary later on. However, we throw away counts below or
    equal to threshold counts - because zipf's law and we don't
    have stocks in any companies producing ram chips.
    """

    FILE_RESERVED_PREFIX = '<ReSeRveD>:'
    FILE_SEP = '\t'
    PAD = '⚽'
    UNK = '∄'
    START_SENT = '⒮'
    END_SENT = 'ⓢ'
    START_WORD = '⒲'
    END_WORD = 'ⓦ'
    PAD_IDX = -1

    def __init__(self,
                 size=None,
                 threshold=0,
                 reserved=None,
                 handle_unk=True,
                 axis=1,
                 **kwargs):
        """
            size: int - the number of tokens we can represent.
            We always represent UNK, START and END but we don't count
            them in len.
            threshold: int - we throw away tokens with up to and including
            this many counts.
            axis: When fitting and encoding what axis the data is on.
            Eg. if 2d array use axis=1 1d axis=0 etc.
        """
        super(Vocab, self).__init__()
        self.size = size
        self.threshold = threshold
        self.reserved = reserved or dict()
        self.handle_unk = handle_unk
        assert axis >= 0
        self.axis = axis
        self.reserved[self.PAD] = -1
        if handle_unk is True:
            self.reserved[self.UNK] = 0
            self.reserved[self.START_SENT] = 1
            self.reserved[self.END_SENT] = 2
            self.reserved[self.START_WORD] = 3
            self.reserved[self.END_WORD] = 4
            # We replace PAD with zero so don't need an embedding for it
        self.num_reserved = len(self.reserved) - 1
        if self.size is not None:
            assert self.size > self.num_reserved
        self.index = dict()
        self.rev_index = dict()
        self.dataset_index = dict()
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def dataset_indices(key):
        return tuple(self.dataset_index[key].values())

    def __repr__(self):
        return ('Vocab object\nactual size: %d\nthreshold: %d\baxis: %d'
                % (len(self), self.threshold, self.axis))

    def __len__(self):
        return len(self.index) + self.num_reserved

    def __getitem__(self, key):
        return self.index[key]

    def __contains__(self, item):
        return item in self.index

    @staticmethod
    def _sorted_k_v(key_value_pairs, reverse=True):
        return sorted(key_value_pairs,
                      # Sort first according to value and then key
                      # NOTE IMPORTANT: if we don't order by second value
                      # as well - can be non-deterministic for equal values
                      key=lambda x: (x[1], x[0]),
                      reverse=reverse)

    @property
    def labels(self):
        # Values are an index from most frequent to most infrequent
        # So need to reverse order
        keys, _ = zip(*self._sorted_k_v(self.index.items(),
                                        reverse=False))
        return keys

    def _build_index(self):
        # we sort because in python 3 most_common is not guaranteed
        # to return the same order for elements with same count
        # when the code runs again. #fun_debugging
        candidates = tuple(self._sorted_k_v(self.counts.most_common()))
        offset = self.num_reserved
        if self.size is not None:
            # size includes space we need for reserved
            limit = self.size - offset
            keep = candidates[:limit]
        else:
            keep = candidates
        # we leave reserved indices to represent the UNK and the rest
        if keep:
            keys, _ = zip(*keep)
            values = tuple(range(offset, len(keys)+offset))
            self.index = dict(zip(keys, values))
            self.index.update(self.reserved)
            self.rev_index = dict(zip(self.index.values(), self.index.keys()))
        else:
            self.index = dict()
            self.rev_index = dict()

    def _threshold_counts(self):
        remove = set()
        for key, c in six.iteritems(self.counts):
            if key in self.reserved:
                remove.add(key)
            if c <= self.threshold:
                remove.add(key)
        for key in remove:
            self.counts.pop(key)

    def fit(self, tokens, axis=None):
        """Populate the vocabulary using the tokens as input.
        Tokens are expected to be a iterable of tokens."""
        axis = self.axis if axis is None else axis
        assert axis >= 0
        unpacked = deep_unpack(tokens, axis)
        self.counts = Counter(unpacked)
        self._threshold_counts()
        self._build_index()
        return self

    def fit_by_dataset(self, tokens, dataset, axis=None):
        """In addition stores information about which tokens were valid
        in which dataset."""
        axis = self.axis if axis is None else axis
        assert axis >= 0
        assert len(tokens) == len(dataset)
        assert len(self.index) > 0
        encoded = self.encode(tokens, axis=axis)

        for row, dset in zip(encoded, dataset):
            unpacked = deep_unpack(row, axis - 1)
            if dset not in self.dataset_index:
                self.dataset_index[dset] = set()
            for each in unpacked:
                self.dataset_index[dset].add(each)
        # convert to tuples
        for dset in self.dataset_index:
            self.dataset_index[dset] = tuple(self.dataset_index[dset])

    def encode(self, tokens, axis=None):
        axis = self.axis if axis is None else axis
        assert axis >= 0
        if self.handle_unk:
            return deep_apply(lambda x: self.index.get(x, self.reserved[self.UNK]),
                              tokens,
                              axis)
        else:
            return deep_apply(lambda x: self.index.get(x), tokens, axis)

    def decode(self, tokens, axis=None):
        axis = self.axis if axis is None else axis
        assert axis >= 0
        if self.handle_unk:
            return deep_apply(lambda x: self.rev_index.get(x, self.UNK),
                              tokens,
                              axis)
        else:
            return deep_apply(lambda x: self.rev_index.get(x), tokens, axis)

    def save(self, filepath):
        # with open(filepath, 'w') as f:
        entries = {'reserved': self.reserved,
                   'index': self.index,
                   'dataset_index': self.dataset_index if self.dataset_index else dict(),
                   'handle_unk': self.handle_unk,
                   'threshold': self.threshold,
                   'axis': self.axis
                   }
        # lines = []
        # for k, v in self.reserved.items():
        #     k = '%s%s' % (self.FILE_RESERVED_PREFIX, k)
        #     lines.append('%s%s%d\n' % (k, self.FILE_SEP, v))
        # # Values are an index from most frequent to most infrequent
        # # So need to reverse order
        # for k, v in self._sorted_k_v(self.index.items(), reverse=False):
        #     # If we use a list with append we reverse order.
        #     if k not in self.reserved.keys():
        #         lines.append('%s%s%d\n' % (k, self.FILE_SEP, v))
        # all_lines = ''.join(lines)
        with open(filepath, 'w') as f:
            f.write(yaml.safe_dump(entries, default_flow_style=False, sort_keys=False))

    def load(self, filepath):

        if not os.path.isfile(filepath):
            raise ValueError('%s does not exist' % filepath)
        with open(filepath, 'r') as f:
            entries = yaml.safe_load(f)

        for k, v in entries.items():
            setattr(self, k, v)

        self.rev_index = {v: k for k, v in self.index.items()}

        return self
