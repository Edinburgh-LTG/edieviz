import os
import six
import yaml
import numpy as np
from shutil import copyfile
from collections import Counter, defaultdict
from edien.utils import deep_unpack, deep_apply
from pytorch_transformers import BertTokenizer
from transformers import AutoTokenizer


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


class BertCoder(object):

    def __init__(self,
                 filename,
                 bert_filename,
                 do_lower_case=False,
                 word_boundaries=False):
        self.filename = filename
        self.bert_filename = bert_filename
        self.do_lower_case = do_lower_case
        self.do_basic_tokenize = False
        # Hack around the fact that we need to know the word boundaries
        self.word_boundaries = word_boundaries

    def __len__(self):
        return self.tokenizer.vocab_size

    def fit(self, tokens):
        # NOTE: We allow the model to use default: do_basic_tokenize.
        # This potentially splits tokens into more tokens apart from subtokens:
        # eg. Mr.Doe -> Mr . D ##oe  (Note that . is not preceded by ##)
        # We take this into account when creating the token_flags in
        # function text_to_token_flags
        self.tokenizer = BertTokenizer(self.bert_filename,
                                       # do_basic_tokenize=self.do_basic_tokenize,
                                       do_lower_case=self.do_lower_case)
        return self

    def text_to_token_flags(self, text):
        """Return a tuple representing which subtokens are the beginning of a
        token. This is needed for NER using BERT:

        https://arxiv.org/pdf/1810.04805.pdf:

        "We use the representation of the first sub-token as the input to the
        token-level classifier over the NER label set."

        """
        text = self.tokenizer.basic_tokenizer._run_strip_accents(text)
        token_flags = []
        if self.do_lower_case:
            actual_split = text.lower().split()
        else:
            actual_split = text.split()

        bert_tokens = []
        for token in actual_split:
            local_bert_tokens = self.tokenizer.tokenize(token) or ['[UNK]']
            token_flags.append(1)
            for more in local_bert_tokens[1:]:
                token_flags.append(0)
            bert_tokens.extend(local_bert_tokens)
        # assert len(actual_tokens) == 0, [actual_tokens, actual_split, bert_tokens]
        assert len(token_flags) == len(bert_tokens), [actual_split, bert_tokens]
        assert sum(token_flags) == len(actual_split)
        return tuple(token_flags)

    def encode(self, tokens):
        # Sometimes tokens include whitespace!
        # for sent_tokens in tokens:
        #     for token in sent_tokens:
        #         if ' ' in token:
        #             print(token)
        # The AIS dataset has a token ". .", for example.
        sent_tokens_no_ws = [[token.replace(' ', '') for token in sent_tokens]
                             for sent_tokens in tokens]
        texts = (' '.join(sent_tokens) for sent_tokens in sent_tokens_no_ws)
        if self.word_boundaries:
            encoded = tuple(self.text_to_token_flags(text)
                            for text in texts)
            # encoded = tuple(tuple(0 if token.startswith('##') else 1
            #                       for token in self.tokenizer.tokenize(text))
            #                 for text in texts)
        else:
            # Adds CLS and SEP
            encoded = tuple(tuple(self.tokenizer.encode(text, add_special_tokens=True))
                            for text in texts)
        return encoded

    def decode(self, ids):
        if self.word_boundaries:
            return []
        else:
            # NOTE: we only encode a single sentence, so use [0]
            return tuple(tuple(self.tokenizer.decode(sent_ids, clean_up_tokenization_spaces=False)[0].split())
                         for sent_ids in ids)

    def load(self, filename):
        self.tokenizer = BertTokenizer(filename,
                                       # do_basic_tokenize=self.do_basic_tokenize,
                                       do_lower_case=self.do_lower_case)
        return self

    def save(self, filename):
        copyfile(self.bert_filename, filename)


# Below is an adaptation of BERTCoder to updated hugging-face API
# Assumes you request what tokenizer to download
class TransformerCoder(object):

    def __init__(self,
                 filename,
                 do_lower_case=False,
                 word_boundaries=False):
        self.filename = filename
        self.do_lower_case = do_lower_case
        self.do_basic_tokenize = False
        # Hack around the fact that we need to know the word boundaries
        self.word_boundaries = word_boundaries

    def __len__(self):
        return self.tokenizer.vocab_size

    def fit(self, tokens):
        # NOTE: We allow the model to use default: do_basic_tokenize.
        # This potentially splits tokens into more tokens apart from subtokens:
        # eg. Mr.Doe -> Mr . D ##oe  (Note that . is not preceded by ##)
        # We take this into account when creating the token_flags in
        # function text_to_token_flags
        self.tokenizer = AutoTokenizer.from_pretrained(self.filename,
                                                       use_fast=False,
                                                       # do_basic_tokenize=self.do_basic_tokenize,
                                                       do_lower_case=self.do_lower_case)
        return self

    def text_to_token_flags(self, text):
        """Return a tuple representing which subtokens are the beginning of a
        token. This is needed for NER using BERT:

        https://arxiv.org/pdf/1810.04805.pdf:

        "We use the representation of the first sub-token as the input to the
        token-level classifier over the NER label set."

        """
        text = self.tokenizer.basic_tokenizer._run_strip_accents(text)
        token_flags = []
        if self.do_lower_case:
            actual_split = text.lower().split()
        else:
            actual_split = text.split()

        bert_tokens = []
        for token in actual_split:
            local_bert_tokens = self.tokenizer.tokenize(token) or ['[UNK]']
            token_flags.append(1)
            for more in local_bert_tokens[1:]:
                token_flags.append(0)
            bert_tokens.extend(local_bert_tokens)
        # assert len(actual_tokens) == 0, [actual_tokens, actual_split, bert_tokens]
        assert len(token_flags) == len(bert_tokens), [actual_split, bert_tokens]
        assert sum(token_flags) == len(actual_split)
        return tuple(token_flags)

    def encode(self, tokens):
        # Sometimes tokens include whitespace!
        # for sent_tokens in tokens:
        #     for token in sent_tokens:
        #         if ' ' in token:
        #             print(token)
        # The AIS dataset has a token ". .", for example.
        sent_tokens_no_ws = [[token.replace(' ', '') for token in sent_tokens]
                             for sent_tokens in tokens]
        texts = (' '.join(sent_tokens) for sent_tokens in sent_tokens_no_ws)
        if self.word_boundaries:
            encoded = tuple(self.text_to_token_flags(text)
                            for text in texts)
            # encoded = tuple(tuple(0 if token.startswith('##') else 1
            #                       for token in self.tokenizer.tokenize(text))
            #                 for text in texts)
        else:
            # Adds CLS and SEP
            encoded = tuple(tuple(self.tokenizer.encode(text, add_special_tokens=True))
                            for text in texts)
        return encoded

    def decode(self, ids):
        if self.word_boundaries:
            return []
        else:
            # NOTE: we only encode a single sentence, so use [0]
            return tuple(tuple(self.tokenizer.decode(sent_ids, clean_up_tokenization_spaces=False).split())
                         for sent_ids in ids)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.filename,
                                                       use_fast=False,
                                                       # do_basic_tokenize=self.do_basic_tokenize,
                                                       do_lower_case=self.do_lower_case)
        return self

    def save(self, filename):
        pass
