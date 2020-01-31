import os
import re
import numpy as np
import dataclasses

from dataclasses import dataclass
from collections import Counter, defaultdict
from itertools import chain
from xml.etree import ElementTree as ET
from edien.data import base
from edien.vocab import Vocab


class EdIELoader(object):

    def __init__(self, folder, docs):
        super(EdIELoader, self).__init__()
        self.folder = folder
        self.docs = docs

    def __repr__(self):
        template = '<EdIELoader %s : %d docs>'
        parts = (self.folder, len(self.docs))
        return template % parts

    @property
    def sentences(self):
        return tuple(chain.from_iterable(d.sentences for d in self.docs))

    @staticmethod
    def get_id_from_filename(filename):
        return filename.split('-')[0]

    @staticmethod
    def collect_xmls(folder, ignore_ids=None):
        filenames = []
        ignore_ids = set(ignore_ids or [])
        for _, _, files in os.walk(folder):
            for f in files:
                if f.endswith('.xml'):
                    f_id = EdIELoader.get_id_from_filename(f)
                    if f_id not in ignore_ids:
                        fname = os.path.join(folder, f)
                        filenames.append(fname)
        return filenames

    @staticmethod
    def get_ids_from_file(filename):
        with open(filename, 'r') as f:
            ids = tuple(f.read().split('\n'))
        return ids

    @classmethod
    def load(cl, folder, ignore_ids=None):
        ignore_ids = ignore_ids or []
        # Below documents have overlapping modifier tags
        ignore_ids.extend(['18001', '1914', '21928', '26146'])
        # NOTE: not sure if we will always be able to filter out by id
        # without the actual need to open the file.
        # NOTE 2: We use ignore_ids to load a train and dev set
        filenames = EdIELoader.collect_xmls(folder, ignore_ids)
        docs = []
        for fn in filenames:
            root = ET.parse(fn)
            for doc_tag in root.iter('document'):
                parsed_doc = EdIEDoc.from_xml(doc_tag)
                docs.append(parsed_doc)
        if docs:
            return cl(folder, docs)
        else:
            raise ValueError('No documents found in %s' % folder)

    # def find_train_dev_split(self,
    #                          y_label,
    #                          train_split=.9,
    #                          split_tolerance=.02,
    #                          attempts=100):
    #     """
    #     Heuristic approach - no guarantees.
    #     Shuffle dataset, impose constraints
    #
    #     Assumptions:
    #         1. Don't mix sentences between documents
    #         2. Dev set should have at least one example of each class
    #         3. Percentage of sentences in train should be:
    #            train_split +- split_tolerance
    #         4. Proportion of classes should be as close as possible
    #
    #     y_label: attribute over which we want the distribution to be
    #     approximately equal
    #
    #     train_split: percentage of sentences we want for training
    #
    #     split_tolerance: split won't be exact, accept if within +- margin
    #
    #     attempts: how many times to randomly shuffle docs and try
    #
    #     returns: list of document ids
    #     """
    #
    #     np.random.seed(5)
    #     assert(0 < train_split < 1)
    #     assert(attempts > 0)
    #     assert(train_split + split_tolerance < 1)
    #     assert(train_split - split_tolerance > 0)
    #     # acceptability range allows
    #     num_docs = len(self.docs)
    #     doc_indices = np.arange(num_docs)
    #
    #     lucky_seeds = np.arange(attempts)
    #     num_docs_train = int(num_docs * train_split)
    #
    #     best_class_distr_train = None
    #     best_class_distr_dev = None
    #     best_train_doc_ids = None
    #     best_sent_split = None
    #     best_kl = np.inf
    #
    #     PRINT_EVERY = 100
    #
    #     print('Running for a total of %d attempts' % attempts)
    #     for i, seed in enumerate(lucky_seeds):
    #         if i % PRINT_EVERY == (PRINT_EVERY - 1):
    #             print('Have run %d attempts. Best kl: %.4f' % ((i + 1), best_kl), end='\r')
    #         np.random.seed(seed)
    #         # Histograms for train/dev split
    #         candidate_train_indices = np.random.choice(doc_indices,
    #                                                    size=num_docs_train,
    #                                                    replace=False)
    #
    #         candidate_dev_indices = [i for i in doc_indices
    #                                  if i not in candidate_train_indices]
    #
    #         train_sents = tuple(chain.from_iterable(self.docs[i].sentences
    #                                           for i in candidate_train_indices))
    #         dev_sents = tuple(chain.from_iterable(self.docs[i].sentences
    #                                         for i in candidate_dev_indices))
    #
    #         train_keys, train_counts = zip(*Counter(
    #                                        chain.from_iterable(getattr(s, y_label)
    #                                                            for s in train_sents)
    #                                                 ).items())
    #         dev_counter = Counter(chain.from_iterable(getattr(s, y_label)
    #                                                   for s in dev_sents))
    #
    #         train_counts = np.array(train_counts)
    #         dev_counts = np.array(tuple(dev_counter[k] for k in train_keys))
    #
    #         class_distr_train = train_counts / train_counts.sum()
    #         class_distr_dev = dev_counts / dev_counts.sum()
    #
    #         # A doc split doesn't necessary lead to a reasonable sent split
    #         sent_split_perc = len(train_sents) / (len(train_sents) + len(dev_sents))
    #         lower_bound = (train_split - split_tolerance)
    #         upper_bound = (train_split + split_tolerance)
    #         good_sent_split = lower_bound < sent_split_perc < upper_bound
    #
    #         all_classes_in_both = (len(class_distr_train) == len(class_distr_dev))
    #
    #         # Check we still have all classes..
    #         if good_sent_split and all_classes_in_both:
    #             # Below "entropy" is KL-divergence.
    #             left_kl = entropy(class_distr_train, class_distr_dev)
    #             right_kl = entropy(class_distr_train, class_distr_dev)
    #             kl = (left_kl + right_kl)/2
    #             if kl < best_kl:
    #                 best_kl = kl
    #                 best_sent_split = sent_split_perc
    #                 best_train_doc_ids = [self.docs[i].doc_id
    #                                       for i in candidate_train_indices]
    #                 best_train_keys = train_keys
    #                 best_train_counts = train_counts
    #                 best_dev_counts = dev_counts
    #                 best_class_distr_train = class_distr_train
    #                 best_class_distr_dev = class_distr_dev
    #     if best_kl != np.inf:
    #         print()
    #         print('Best kl: %.4f' % best_kl)
    #         print('Best sent split perc: %.4f ' % best_sent_split)
    #         print('Class distribution')
    #         KEY_SIZE = 20
    #         print('Key%s\tTrain\t  Dev\t#Train\t#Dev' % (' ' *(KEY_SIZE - 3)))
    #         for key, tr_d, dev_d, tr_c, dev_c in zip(best_train_keys,
    #                                                  best_class_distr_train,
    #                                                  best_class_distr_dev,
    #                                                  best_train_counts,
    #                                                  best_dev_counts):
    #             print('%s\t%.4f\t%.4f\t%d\t%d' % (key[:KEY_SIZE].ljust(KEY_SIZE),
    #                                               tr_d,
    #                                               dev_d,
    #                                               tr_c,
    #                                               dev_c))
    #         return best_train_doc_ids
    #     else:
    #         raise ValueError('No splits found')


class EdIEDataset(base.Dataset):

    def __init__(self,
                 train_paths,
                 dev_path,
                 test_path,
                 split_folder=None):
        """Note this can be either ESS or Tayside"""
        assert isinstance(train_paths, list)
        super(EdIEDataset, self).__init__(train_paths, dev_path, test_path)
        self.split_folder = base.Dataset.get_path(split_folder)

    @property
    def train_sents(self):
        if self.split_folder:
            dev_filename = os.path.join(self.split_folder, 'dev_ids.txt')
            dev_ids = EdIELoader.get_ids_from_file(dev_filename)
        else:
            dev_ids = []

        ess_folder = self.train_paths[0]
        train = EdIELoader.load(ess_folder, ignore_ids=dev_ids)
        train_sentences = [t for t in train.sentences if len(t) > 2]

        for folder in self.train_paths[1:]:
            train = EdIELoader.load(folder, ignore_ids=[])
            train_sentences.extend(train.sentences)
        print('Loaded %d train sentences' % len(train_sentences))
        return train_sentences

    @property
    def dev_sents(self):
        if self.split_folder:
            train_filename = os.path.join(self.split_folder, 'train_ids.txt')
            train_ids = EdIELoader.get_ids_from_file(train_filename)
        else:
            train_ids = []

        dev = EdIELoader.load(self.dev_path, ignore_ids=train_ids)
        print('Loaded %d dev sentences' % len(dev.sentences))
        return [s for s in dev.sentences if len(s) > 1]

    @property
    def test_sents(self):
        test = EdIELoader.load(self.test_path)
        print('Loaded %d test sentences' % len(test.sentences))
        return test.sentences


@dataclass(frozen=True)
class EdIEDoc(base.Document):
    """"""
    doc_id: str
    labels: tuple

    SECTIONS = ('REPORT', 'CONCLUSION')
    MODIFIER_PREFIXES = ('loc_', 'time_')
    NUM_REPLACE = re.compile(r'[0-9]+')
    # There is a weird deletion character in the data
    DROP_TOKENS = ['\x7f']

    def __repr__(self):
        template = '<EdIEDoc %s : %d report, %d conclusion, tags: %r>'
        parts = (self.doc_id, len(self.reports), len(self.conclusions), self.labels)
        return template % parts

    def __len__(self):
        return len(self.sentences)

    @property
    def reports(self):
        return tuple(s for s in self.sentences
                     if s.section == EdIEDoc.SECTIONS[0])

    @property
    def conclusions(self):
        return tuple(s for s in self.sentences
                     if s.section == EdIEDoc.SECTIONS[1])

    @property
    def conclusions_text(self):
        return '\n'.join(s.text for s in self.conclusions)

    @property
    def reports_text(self):
        return '\n'.join(s.text for s in self.reports)

    @property
    def entity_labels(self):
        """Only return the entity without the modifiers"""
        return tuple(l.split(',')[0] for l in self.labels)

    @property
    def label_factors(self):
        """Get all unique modifiers and entities we have from gold dataset"""
        modifiers = set(chain.from_iterable([t[2:] for t in s.mod_tags if len(t) > 2]
                                            for s in self.sentences))
        entities = set(chain.from_iterable([t[2:] for t in s.ner_tags if len(t) > 2]
                                           for s in self.sentences))
        combine = tuple(sorted(modifiers.union(entities)))
        return combine

    @classmethod
    def get_document_labels(cl, parse_obj):
        labels = set()
        for ent_tag in parse_obj.iter('ent'):
            ent_type = ent_tag.get('type')
            if ent_type.startswith('label:'):
                labels.add(ent_type[6:])
        return tuple(labels)

    @classmethod
    def build_entity_dict(cl, parse_obj):
        """Create mapping from word ids to negated entities and entity types.

        returns: set, dict
        """
        # NOTE: We assume all entity/negation/modifier annotations are made
        # in the section delimited by <standoff>
        standoff = tuple(parse_obj.iter('standoff'))
        assert(len(standoff) == 1)
        standoff = standoff[0]
        # TODO: Add relations
        # Store which entity word ids are negated
        negated = set()
        # Store word id to tag lookup
        entities = dict()
        modifiers = dict()
        for ent_tag in standoff.iter('ent'):
            ent_type = ent_tag.get('type')
            negated_attr = ent_tag.get('neg', None)
            if not ent_type.startswith('label:'):
                label = ent_type.replace('neg_', '')
                # assert label != 'mod', 'Need to add mod stuff below'
                if label == 'mod':
                    label = ent_tag.get('stime')
                    if label is None:
                        label = 'loc_%s' % ent_tag.get('sloc')
                    else:
                        label = ('time_%s' % label)
                # Capture negation and remove prefix from type
                is_modifier = any(label.startswith(p)
                                  for p in EdIEDoc.MODIFIER_PREFIXES)
                parts = tuple(ent_tag.iter('part'))
                assert(len(parts) == 1)
                part = parts[0]

                start_word_id = part.get('sw')
                end_word_id = part.get('ew')
                if is_modifier:
                    assert start_word_id not in modifiers, 'Overlapping mods in doc %s' % parse_obj.get('id', 'unknown')
                    modifiers[start_word_id] = ('B-%s' % label)
                else:
                    assert start_word_id not in entities, 'Overlapping ents in doc %s' % parse_obj.get('id', 'unknown')
                    entities[start_word_id] = ('B-%s' % label)
                if ent_type.startswith('neg_') or negated_attr == 'yes':
                    negated.add(start_word_id)
                # If multi word token need to set the In tokens
                if start_word_id != end_word_id:
                    current_char_offset = int(start_word_id[1:])
                    end_char_offset = int(end_word_id[1:])
                    diff = end_char_offset - current_char_offset
                    assert diff > 0, 'Ids no longer expressing char offset'
                    # SPAM lookup with all inbetween (overkill)
                    for offset in range(diff):
                        current_char_offset += 1
                        span_token_id = ('w%d' % current_char_offset)
                        if is_modifier:
                            assert span_token_id not in modifiers, 'Overlapping mods'
                            modifiers[span_token_id] = ('I-%s' % label)
                        else:
                            assert span_token_id not in entities, 'Overlapping ents'
                            entities[span_token_id] = ('I-%s' % label)
                        if ent_type.startswith('neg_') or negated_attr == 'yes':
                            negated.add(span_token_id)

        return entities, modifiers, negated

    @classmethod
    def from_xml(cl, parse_obj, proc_all=False):
        """Parse a EdIE document from XML parser"""
        entities, modifiers, negated = EdIEDoc.build_entity_dict(parse_obj)
        doc_id = parse_obj.get('id', None)
        sentences = []
        sent_section = 'unknown'
        # Iterate over all children tags that are sentences
        for sent_tag in parse_obj.iter('s'):
            sent_id = sent_tag.get('id', None)
            # Some sentences are not to be processed - this is flagged by proc
            process = sent_tag.get('proc', None)
            # Unless we set proc_all, only process proc='yes' sentences.
            if process == 'yes' or proc_all:
                sent = defaultdict(lambda: [])
                for word_tag in sent_tag.iter('w'):
                    # Ignore tokens that are errors / dataset irregularities
                    if word_tag.text in cl.DROP_TOKENS:
                        continue
                    # TODO: Possibly consider stopword list
                    word_id = word_tag.get('id', '')
                    # word_string = EdIEDoc.NUM_REPLACE.sub('<d>', word_tag.text)
                    word_string = word_tag.text
                    lemma = word_tag.get('l', '')
                    pos_tag = word_tag.get('p', '')
                    # Lookup word id in entity dictionary and default to O
                    ner_tag = entities.get(word_id, 'O')
                    mod_tag = modifiers.get(word_id, 'O')
                    negation = 'neg' if word_id in negated else 'pos'
                    # word_type = word_tag.get('type', '')
                    # head = word_tag.get('headn', '')
                    # TODO: lookup label in dict to change
                    sent['word_ids'].append(word_id)
                    sent['tokens'].append(word_string)
                    sent['lemmas'].append(lemma)
                    sent['pos_tags'].append(pos_tag)
                    sent['ner_tags'].append(ner_tag)
                    sent['mod_tags'].append(mod_tag)
                    sent['negation'].append(negation)
                # Tuplify lists
                sent = {k: tuple(v) for k, v in sent.items()}
                sent['sent_id'] = sent_id
                sent['section'] = sent_section

                parsed_sent = EdIESent(**sent)
                sentences.append(parsed_sent)
            else:
                # Attempt to identify in what section we are in
                children = list(sent_tag)
                if len(children) == 1:
                    section = children[0].text
                    if section in EdIEDoc.SECTIONS:
                        sent_section = section
        labels = EdIEDoc.get_document_labels(parse_obj)
        return cl(doc_id=doc_id,
                  sentences=sentences,
                  labels=labels)


@dataclass(frozen=True)
class EdIESent(base.Sentence):
    sent_id: str
    section: str
    word_ids: tuple
    lemmas: tuple
    pos_tags: tuple
    mod_tags: tuple
    ner_tags: tuple
    negation: tuple

    @property
    def lemmas_word_fallback(self):
        return tuple(l.lower() if l else w.lower()
                     for l, w in zip(self.lemmas, self.tokens))

    @property
    def spannified_ner_tags(self):
        # TODO: potentially add DT (it adds negation as well)
        NP_TAGS = ['CD', 'JJ', 'NN', 'NNS']
        tags = list(self.ner_tags)
        pos_tags = list(self.pos_tags)
        i = 0
        while i < len(tags):
            ner_type = tags[i].split('-')[-1]
            # Find an entity span
            if ner_type not in ('O', Vocab.PAD):
                # Fix by finding left index to extend NP span backward
                left_idx = i
                right_idx = i
                # Scan sentence backwards
                j = i - 1
                while j >= 0:
                    if pos_tags[j] in NP_TAGS and tags[j] == 'O':
                        left_idx = j
                    else:
                        break
                    j -= 1
                # Fix by finding right index to extend NP span forward
                j = i + 1
                while j < len(tags):
                    if tags[j] == 'O':
                        if pos_tags[j] in NP_TAGS:
                            right_idx = j
                            j += 1
                        else:
                            j += 1
                            break
                    # Below retains number of entities - otherwise we merge neighbouring mentions of same type
                    # elif tags[j][2:] == ner_type and tags[j][:2] == 'I-':
                    elif tags[j][2:] == ner_type:
                        right_idx = j
                        j += 1
                    # If the tags don't match
                    else:
                        break
                i = j
                # Fix first token of "NP"
                tags[left_idx] = ('B-%s' % ner_type)
                # Fix rest tokens if they exist
                for j in range(left_idx + 1, right_idx + 1):
                    tags[j] = ('I-%s' % ner_type)
            else:
                i += 1
        return tuple(tags)

    @property
    def ner_tags_untyped(self):
        return tuple('%s%s' % (t[:2], 'entity') if t != 'O' else t
                     for t in self.ner_tags)

    @property
    def mod_tags_untyped(self):
        return tuple('%s%s' % (t[:2], 'modifier') if t != 'O' else t
                     for t in self.mod_tags)

    @property
    def ner_tags_no_bio(self):
        return tuple(tag[2:] if tag not in ('O', Vocab.PAD) else tag
                     for tag in self.ner_tags)

    @property
    def mod_tags_no_bio(self):
        return tuple(tag[2:] if tag not in ('O', Vocab.PAD) else tag
                     for tag in self.mod_tags)

    @property
    def ner_tags_no_outer(self):
        return tuple(tag for tag in self.ner_tags if tag != 'O')

    @property
    def mod_tags_no_outer(self):
        return tuple(tag for tag in self.mod_tags if tag != 'O')

    @property
    def ner_and_mod_tags(self):
        return tuple(tag for tag in chain(self.mod_tags, self.ner_tags)
                     if tag != 'O')

    @property
    def unprefixed_ner_and_mod_tags(self):
        return tuple(tag for tag in chain(self.mod_tags_no_bio,
                                          self.ner_tags_no_bio)
                     if tag != 'O')

    @property
    def ner_and_mod_tags_no_bio(self):
        return tuple(ner if ner != 'O' else mod
                     for mod, ner in zip(self.mod_tags_no_bio,
                                         self.ner_tags_no_bio))

    @property
    def is_entity(self):
        return tuple('ent' if ner[:2] in ('B-', 'I-')
                     or mod[:2] in ('B-', 'I-') else ner
                     for mod, ner in zip(self.mod_tags,
                                         self.ner_tags))

    @property
    def ner_and_mod_tags_bio(self):
        return tuple(ner if ner != 'O' else mod
                     for mod, ner in zip(self.mod_tags,
                                         self.ner_tags))

    @property
    def lowercase_tokens(self):
        return tuple(w.lower() for w in self.tokens)

    @property
    def non_entity_tokens(self):
        return tuple(w if ent != 'ent' else '<$ent$>'
                     for w, ent in zip(self.tokens, self.is_entity))

    def ngrams(self, n, padding=None):
        # padding is a 2 element tuple/list (left, right)
        # specifying how many elements we want to pad to the left
        # and to the right
        assert(n > 0)
        if padding is None:
            return tuple(self.span(i, i+n)
                         for i in range(len(self) - n + 1))
        else:
            pad_left, pad_right = padding
            assert(pad_left >= 0)
            assert(pad_right >= 0)
            return tuple(self.padded_span(-pad_left + i, -pad_left + i + n)
                         for i in range(len(self)))

    def padded_span(self, start, end):
        assert(start != end)
        k_start = max(0, start)
        k_end = min(len(self), end)
        iterable_fields = [field.name
                           for field in dataclasses.fields(EdIESent)
                           if field.type in (tuple, list)]
        sent_fields = tuple(getattr(self, field)[k_start:k_end]
                            for field in iterable_fields)
        num_fields_to_pad = len(iterable_fields)
        if start < 0 or end > len(self):
            parts = []
            if start < 0:
                num_pad_left = -start
                pad_col = (Vocab.PAD,) * num_pad_left
                left_padding = (pad_col,) * num_fields_to_pad
                parts.append(left_padding)
            parts.append(sent_fields)
            if end > len(self):
                num_pad_right = end - len(self)
                pad_col = (Vocab.PAD,) * num_pad_right
                right_padding = (pad_col,) * num_fields_to_pad
                parts.append(right_padding)

            sent_fields = tuple(tuple(chain.from_iterable(x))
                                for x in zip(*parts))
        sent_fields = dict(zip(iterable_fields, sent_fields))

        return EdIESent(sent_id=self.sent_id,
                        section=self.section,
                        **sent_fields)
