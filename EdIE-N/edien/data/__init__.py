# coding: utf-8
import os
import yaml
from collections import namedtuple
from itertools import chain
from edien.vocab import Vocab
from edien.utils import deep_unpack

from edien.data.ess import EdIELoader, EdIEDataset


class DatasetEncoder(object):

    """"""

    def __init__(self,
                 name,
                 dataset,
                 inputs,
                 targets,
                 metrics,
                 formatter,
                 vocab_encoder=None):
        """Limit returned attributes to inputs and targets.
        Optionally encode vocabs. Potentially add transforms."""
        self.name = name
        self.dataset = dataset
        self.inputs = inputs
        self.targets = targets
        # This is a dictionary mapping between targets and metrics
        self.metrics = metrics
        # TODO: might need many transforms
        self.formatter = formatter
        self.vocab_encoder = vocab_encoder

    def encode(self, sents, paths, train=False):
        sents = self.formatter.encode(sents, self.inputs, self.targets)
        if self.vocab_encoder is not None:
            if not self.vocab_encoder.vocabs_are_loaded:
                if train:
                    self.vocab_encoder.load_vocabs(paths.vocab_folder,
                                                   data=sents)
                else:
                    self.vocab_encoder.load_vocabs(paths.vocab_folder)
            sents = self.vocab_encoder.encode(sents)
        return sents

    @property
    def train_sents(self):
        return self.dataset.train_sents

    @property
    def dev_sents(self):
        return self.dataset.dev_sents

    @property
    def test_sents(self):
        return self.dataset.test_sents

    def train_vars(self, paths):
        """Return variables for train as named tuple"""
        train = self.encode(self.dataset.train_sents, paths, train=True)
        return train

    def dev_vars(self, paths):
        """Return variables for dev as named tuple"""
        dev = self.encode(self.dataset.dev_sents, paths)
        return dev

    def test_vars(self, paths):
        """Return variables for test as named tuple"""
        test = self.encode(self.dataset.test_sents, paths)
        return test

    def decode(self, preds, axis=1):
        """Inverse vocab dict lookup."""
        decoded_preds = self.vocab_encoder.decode(preds)
        decoded_preds = self.formatter.decode(decoded_preds, axis=axis)
        return decoded_preds


# TODO:
# For multitask / transfer learning allow loading multiple datasets
# X and y attributes are prefixed by dataset name in namedtuple
class DatasetFuser(object):
    """docstring for DatasetFuser"""
    def __init__(self, datasets):
        assert len(datasets) >= 1
        self.datasets = datasets
        self.dev_dataset = datasets[0]
        self.test_dataset = datasets[0]
        self.dev_path = self.dev_dataset.dev_path
        self.test_path = self.test_dataset.test_path

    @property
    def train_sents(self):
        return tuple(chain(*[ds.train_sents for ds in self.datasets]))

    @property
    def dev_sents(self):
        return self.dev_dataset.dev_sents
        # return tuple(chain(*[ds.dev_sents for ds in self.datasets]))

    @property
    def test_sents(self):
        return self.test_dataset.test_sents
        # return tuple(chain(*[ds.test_sents for ds in self.datasets]))


class VocabEncoder(object):
    """Encodes data to int using vocab lookups"""

    def __init__(self, vocabs, load_if_exists=True):
        self.vocabs = vocabs
        # Whether to load vocab from file if it exists
        # False means we always refit the vocabulary and overwrite
        # previous vocab files.
        self.load_if_exists = load_if_exists
        self.vocabs_are_loaded = False

    def encode(self, sents):

        if not self.vocabs_are_loaded:
            raise ValueError('Need to call: load_vocabs first')

        fields = sents._fields
        valid_fields = list(fields)

        encoded_attr = []
        for attr in fields:
            vocab = self.vocabs[attr]
            data = getattr(sents, attr)
            if data[0] is None:
                # print('%s is None, discarding' % attr)
                valid_fields.pop(valid_fields.index(attr))
            else:
                encoded = vocab.encode(data)
                encoded_attr.append(encoded)
                index = 0
                # print(attr,
                #       data[index:index+1],
                #       '->',
                #       encoded[index:index+1],
                #       '->',
                #       vocab.decode(encoded[index:index+1]))

        nt = namedtuple('VocabEncoded', valid_fields)
        return nt(*encoded_attr)

    def decode(self, preds):

        if not self.vocabs_are_loaded:
            raise ValueError('Need to call: load_vocabs first')

        fields = preds._fields
        nt = namedtuple('VocabDecoded', fields)

        decoded_attr = []
        for attr in fields:
            vocab = self.vocabs[attr]
            data = getattr(preds, attr)
            decoded_attr.append(vocab.decode(data))
        return nt(*decoded_attr)

    def load_vocabs(self, vocab_folder, data=None):
        # If we pass in sents we fallback to fitting the vocabs
        # on that dataset

        for k in self.vocabs.keys():

            v = self.vocabs[k]
            vocab_path = os.path.join(vocab_folder, v.filename)

            if self.load_if_exists and os.path.isfile(vocab_path):
                vocab = v.load(vocab_path)
                # print('Found vocab for %s in file %s' % (k, vocab_path))
            else:
                if data is not None:
                    # Try to find attribute this belongs to
                    sents = getattr(data, k)

                    vocab = v.fit(sents)

                    print('Saving (%s, size=%d) vocab to %s' %
                          (k, len(vocab), vocab_path))
                    vocab.save(vocab_path)
                else:
                    msg = "Attempted to fit vocab on test/dev data"
                    raise ValueError('%s' % msg)
            self.vocabs[k] = vocab

        # If we have a dataset attribute, after loading the dataset we compute
        # which indices are used for which datasets.
        dataset = getattr(data, 'dataset', None)
        if dataset is not None:
            dataset = self.vocabs['dataset'].encode(dataset)
            for k in self.vocabs.keys():
                v = self.vocabs[k]
                vocab_path = os.path.join(vocab_folder, v.filename)
                if k != 'dataset':
                    sents = getattr(data, k)
                    self.vocabs[k].fit_by_dataset(sents, dataset)
                    # Overwrite vocab with version that has per dataset
                    # index as well
                    self.vocabs[k].save(vocab_path)
        self.vocabs_are_loaded = True


class WindowedNGrams(object):
    """WindowedNGrams formatter"""
    def __init__(self, ngram, target_position):
        super(WindowedNGrams, self).__init__()
        assert ngram > 0
        assert target_position >= 0
        assert target_position < ngram
        self.ngram = ngram
        self.target_position = target_position

    def encode(self, sentences, inputs, targets):
        right_pad = self.ngram - (self.target_position + 1)
        left_pad = self.ngram - right_pad - 1
        padding = (left_pad, right_pad)

        # NOTE: we ignore sentence boundaries for now
        ngrams = tuple(chain.from_iterable(s.ngrams(self.ngram,
                                                    padding=padding)
                                           for s in sentences))

        attrs = tuple(chain(inputs, targets))
        nt = namedtuple('windowed', attrs)

        attr_vals = []
        for attr in attrs:
            if attr in targets:
                attr_vals.append(tuple(getattr(ng, attr)[self.target_position]
                                 for ng in ngrams))
            else:
                attr_vals.append(tuple(getattr(ng, attr) for ng in ngrams))

        return nt(*attr_vals)


class Sentences(object):
    """Sentences formatter"""
    def __init__(self):
        super(Sentences, self).__init__()

    def encode(self, sentences, inputs, targets):

        attrs = tuple(chain(inputs, targets))
        nt = namedtuple('sentences', attrs)

        attr_vals = []
        for attr in attrs:
            doc_vals = []
            for s in sentences:
                vals = getattr(s, attr)
                # assert(len(vals) <= self.max_len)
                # diff = self.max_len - len(vals)
                # if diff > 0 :
                    # vals = vals + ((Vocab.PAD,) * diff)
                doc_vals.append(vals)
            attr_vals.append(tuple(doc_vals))

        return nt(*attr_vals)

    def decode(self, sentences, axis=1):

        attrs = sentences._fields
        nt = namedtuple('sentences', attrs)

        attr_vals = []
        for attr in attrs:
            # vals = tuple(p for s in getattr(sentences, attr) for p in s)
            vals = tuple(deep_unpack((s for s in getattr(sentences, attr)),
                               axis=axis))
            attr_vals.append(vals)

        return nt(*attr_vals)
