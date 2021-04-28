import os
import math
import importlib
import numpy as np
from dataclasses import dataclass
from itertools import chain
from edien.vocab import Vocab
from edien import EdIENPath


class Dataset(object):
    """Base representation of a Dataset. For each <prefix>Dataset class that
    subclasses this class, we assume a <prefix>Loader class with the same
    <prefix> will be defined in the module. This parent class apart from
    defining some utilities and useful fundamentals, also implements the
    barebone functionality for loading sentences, which may actually
    suffice and not need overriding."""

    def __init__(self,
                 train_path,
                 dev_path=None,
                 test_path=None,
                 train_perc=1.):
        super(Dataset, self).__init__()
        if isinstance(train_path, list) or isinstance(train_path, tuple):
            self.train_path = [Dataset.get_path(p) for p in train_path]
        else:
            self.train_path = Dataset.get_path(train_path)
        self.dev_path = Dataset.get_path(dev_path)
        self.test_path = Dataset.get_path(test_path)
        # Percentage of the training set to use
        assert 0. < train_perc <= 1.
        self.train_perc = train_perc
        module = importlib.import_module(self.__module__)
        # Assumes a __name__Loader is defined in the same module
        loader_name = self.name.replace('Dataset', 'Loader')
        cls = getattr(module, loader_name)
        self.loader = cls

    @classmethod
    def get_path(cl, ending):
        if ending is not None:
            path = EdIENPath()
            return os.path.join(path.datasets_folder, ending)
        return ending

    @property
    def name(self):
        return self.__class__.__name__

    def load_sentences(self, path, section):
        possible_sections = ('train', 'dev', 'test')
        if section not in possible_sections:
            raise ValueError('Section not in %s' % possible_sections)
        sentences = tuple(self.loader.load(path).sentences)
        return sentences

    @property
    def train_sents(self):
        train_sentences = self.load_sentences(self.train_path, 'train')
        if self.train_perc != 1.:
            total_sents = len(train_sentences)
            subset_sents = int(math.ceil(self.train_perc * total_sents))
            train_sentences = list(train_sentences)
            np.random.shuffle(train_sentences)
            train_sentences = tuple(train_sentences[:subset_sents])
            print('Loaded %d out of %d train sentences' %
                  (len(train_sentences), total_sents))
        else:
            print('Loaded %d train sentences' % len(train_sentences))
        return train_sentences

    @property
    def dev_sents(self):
        dev_sentences = self.load_sentences(self.dev_path, 'dev')
        print('Loaded %d dev sentences' % len(dev_sentences))
        return dev_sentences

    @property
    def test_sents(self):
        test_sentences = self.load_sentences(self.test_path, 'test')
        print('Loaded %d test sentences' % len(test_sentences))
        return test_sentences


@dataclass(frozen=True)
class Document:
    sentences: tuple

    def __len__(self):
        return len(self.sentences)

    @property
    def text(self):
        return '\n'.join(s.text for s in self.sentences)


@dataclass(frozen=True)
class Sentence:
    tokens: tuple

    def __len__(self):
        return len(self.tokens)

    @property
    def dataset(self):
        class_name = self.__class__.__name__.rstrip('Sent')
        class_name = '%sDataset' % class_name
        package_name = self.__class__.__module__
        return '%s.%s' % (package_name, class_name)

    @property
    def word_mask(self):
        return self.tokens

    @property
    def words(self):
        print('Warning: using deprecated "words" property')
        return self.tokens

    @property
    def text(self):
        return ' '.join(w for w in self.tokens)

    def to_conll(self, preds, attr):
        lines = []
        gold = getattr(self, attr)
        assert len(gold) == len(preds)
        assert len(preds) == len(self.tokens)
        for fields in zip(self.tokens,
                          self.tokens,  # We don't care about POS
                          gold,
                          preds):
            lines.append('%s\n' % ' '.join(fields))
        return ''.join(lines)

    @property
    def chars(self):
        return tuple(tuple(chain(Vocab.START_WORD, *w, Vocab.END_WORD))
                     if w is not Vocab.PAD else (Vocab.PAD,)
                     for w in self.tokens)

    @property
    def lowercase_chars(self):
        return tuple(tuple(chain(Vocab.START_WORD, *w.lower(), Vocab.END_WORD))
                     if w is not Vocab.PAD else (Vocab.PAD,)
                     for w in self.tokens)

    @property
    def ner_tags_cui_no_bio(self):
        return tuple(valid_labels.lookup_cui(t[2:])
                     if t.startswith('B-') or t.startswith('I-')
                     else t
                     for t in self.ner_tags_cui)


def tag_to_cui(tag, cui_lookup):
    if tag[:2] in ('B-', 'I-'):
        tag = '%s%s' % (tag[:2], cui_lookup[tag[2:]])
    else:
        # We only expect tag to be 'O' if it doesn't have B- or I-
        if len(tag) > 1:
            raise ValueError('Unexpected tag type: %s' % tag)
    return tag
