import os
import re
import copy
import numpy as np
import dataclasses

from dataclasses import dataclass
from collections import Counter, defaultdict
from itertools import chain
from intervaltree import IntervalTree
from xml.etree import ElementTree as ET
from edien.data import base
from edien.vocab import Vocab
from edien.utils import get_hash


class EdIELoader(object):

    def __init__(self, folder, docs):
        super().__init__()
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

    @staticmethod
    def fix_relation_source(parse_obj):
        standoff = parse_obj.find(EdIEDoc.XML_STANDOFF)
        rels = standoff.findall(EdIEDoc.XML_ST_RELS)
        for rel in rels:
            source = rel.get('source')
            if source is None:
                rel.set('source', 'gold')
        return parse_obj

    @classmethod
    def load(cl, folder, ignore_ids=None, **kwargs):
        ignore_ids = ignore_ids or []

        # Below documents have overlapping modifier tags
        # ignore_ids.extend(['18001', '1914', '21928', '26146'])

        # NOTE: not sure if we will always be able to filter out by id
        # without the actual need to open the file.
        # NOTE 2: We use ignore_ids to load a train and dev set
        filenames = EdIELoader.collect_xmls(folder, ignore_ids)
        docs = []
        for fn in filenames:
            root = ET.parse(fn)
            # Monkeypatch: Add gold source attribute to relations if non existent
            root = EdIELoader.fix_relation_source(root)
            # We are assuming each file has a single document
            parsed_doc = EdIEDoc.from_xml(root, **kwargs)
            docs.append(parsed_doc)
        if docs:
            return cl(folder, docs)
        else:
            raise ValueError('No documents found in %s' % folder)


class EdIEDataset(base.Dataset):

    def __init__(self,
                 train_paths,
                 dev_path,
                 test_path,
                 split_folder=None,
                 **kwargs):
        """Note this can be either ESS or Tayside"""
        assert isinstance(train_paths, list)
        super().__init__(train_paths,
                         dev_path,
                         test_path,
                         **kwargs)
        self.split_folder = base.Dataset.get_path(split_folder)

    def load_sentences(self, path, section):
        if section == 'train':
            if self.split_folder:
                dev_filename = os.path.join(self.split_folder, 'dev_ids.txt')
                dev_ids = EdIELoader.get_ids_from_file(dev_filename)
            else:
                dev_ids = []

            ess_folder = self.train_path[0]
            train = EdIELoader.load(ess_folder, ignore_ids=dev_ids)
            train_sentences = [t for t in train.sentences if len(t) > 2]

            for folder in self.train_path[1:]:
                train = EdIELoader.load(folder, ignore_ids=[])
                train_sentences.extend(train.sentences)
            return tuple(train_sentences)
        elif section == 'dev':
            if self.split_folder:
                train_filename = os.path.join(self.split_folder, 'train_ids.txt')
                train_ids = EdIELoader.get_ids_from_file(train_filename)
            else:
                train_ids = []

            dev = EdIELoader.load(self.dev_path, ignore_ids=train_ids)
            return tuple(dev.sentences)
        else:
            return super().load_sentences(path, section)


@dataclass(frozen=True)
class EdIEDoc(base.Document):
    """"""
    doc_id: str
    labels: tuple
    entities: tuple
    relations: tuple
    xml_parse: object

    SECTIONS = ('REPORT', 'CONCLUSION')
    NUM_REPLACE = re.compile(r'[0-9]+')
    # There is a weird deletion character in the data
    DROP_TOKENS = ['\x7f']

    XML_DOC = 'document'
    XML_TEXT = 'text'
    XML_STANDOFF = 'standoff'
    XML_DEFAULT_SOURCE = 'gold'
    XML_ST_ENTS = 'ents'
    XML_ST_RELS = 'relations'
    XML_ST_REL = 'relation'
    XML_ST_ARG = 'argument'
    XML_PARTS = 'parts'
    XML_PART = 'part'
    XML_PARAGRAPH = 'p'
    XML_PROCESS = 'proc'
    XML_ID = 'id'
    XML_LEMMA = 'l'
    XML_POS = 'p'
    XML_WORD = 'w'
    XML_SENT = 's'
    XML_ENTITY = 'ent'
    XML_ENTITY_TYPE = 'type'
    XML_ENTITY_REF = 'ref'
    XML_START_WORD = 'sw'
    XML_END_WORD = 'ew'
    XML_LABEL = 'label:'

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

    @property
    def label_factors_with_negation(self):
        """Get all unique modifiers and entities we have from gold dataset,
         signalling negation in front with neg_
        """

        def get_with_negation(iterable_name):
            res = []
            for s in self.sentences:
                iterable = getattr(s, iterable_name)
                for index, ent in enumerate(iterable):
                    ent = ent[2:]
                    if len(ent) == 0:
                        continue
                    if s.negation[index] == "neg":
                        ent = "neg_" + ent
                    res.append(ent)
            return res

        modifiers = set(get_with_negation("mod_tags"))
        entities = set(get_with_negation("ner_tags"))

        combine = tuple(sorted(modifiers.union(entities)))
        return combine

    @property
    def relation_factors(self):
        """Get all relations
        """
        heads = {}
        for relation in self.relations:
            head_id = relation['head']['start_word_id']
            if head_id not in heads:
                heads[head_id] = {'type': relation['head']['type'],
                                  'text': relation['head']['text'],
                                  'mods': []
                                  }
            heads[head_id]['mods'].append(relation['mod'])

        factors = []
        for head_id in heads:
            strings = []
            for mod in heads[head_id]['mods']:
                strings.append(mod['type'])
            strings = sorted(strings)
            strings.append(heads[head_id]['type'])
            factors.append('_'.join(strings))

        return factors

    @classmethod
    def get_document_labels(cl, parse_obj):
        labels = set()
        for ent_tag in parse_obj.iter(EdIEDoc.XML_ENTITY):
            ent_type = ent_tag.get(EdIEDoc.XML_ENTITY_TYPE)
            if ent_type.startswith(EdIEDoc.XML_LABEL):
                labels.add(ent_type[len(EdIEDoc.XML_LABEL):])
        return tuple(labels)

    @classmethod
    def build_doc_fields(cl, parse_obj, **kwargs):
        """Populate document level finding, modifier and relation list
        from standoff XML tag. Negation is included as part of finding/modifier

        returns: list, list, list
        """
        # NOTE: We assume all entity/negation/modifier annotations are made
        # in the section delimited by <standoff>
        standoff = tuple(parse_obj.iter(EdIEDoc.XML_STANDOFF))
        assert(len(standoff) == 1)
        standoff = standoff[0]
        # Store word id to tag lookup
        relations, findings, modifiers = [], dict(), dict()

        # Sometimes we want to load an ann.xml file that has no gold labels
        no_labels = kwargs.get('no_labels', None)

        if not no_labels:

            entity_source = kwargs.get('entity_source', EdIEDoc.XML_DEFAULT_SOURCE)
            ents = standoff.find("%s/[@source='%s']" % (EdIEDoc.XML_ST_ENTS, entity_source))
            if ents is None:
                raise ValueError("Couldn't find ents with source %s" % entity_source)
            for ent_tag in ents.iter(EdIEDoc.XML_ENTITY):
                ent_type = ent_tag.get(EdIEDoc.XML_ENTITY_TYPE)
                if not ent_type.startswith(EdIEDoc.XML_LABEL):
                    ent = EdIEEntity.from_xml(ent_tag)

                    if ent.is_modifier:
                        modifiers[ent.ent_id] = ent
                    else:
                        findings[ent.ent_id] = ent

            relation_source = kwargs.get('relation_source', EdIEDoc.XML_DEFAULT_SOURCE)
            rel_tags = standoff.find("%s/[@source='%s']" % (EdIEDoc.XML_ST_RELS, relation_source))
            if rel_tags is not None:
                for rel_tag in rel_tags:
                    rel_type = rel_tag.get('type')

                    head_arg, mod_arg, *rest = tuple(rel_tag.iter(EdIEDoc.XML_ST_ARG))
                    assert len(rest) == 0

                    head_id = head_arg.get('ref')
                    head = findings[head_id]
                    assert not head.is_modifier

                    mod_id = mod_arg.get('ref')
                    mod = modifiers[mod_id]
                    assert mod.is_modifier

                    rel = EdIERelation(rel_type, head=head, modifier=mod)
                    relations.append(rel)

        # We don't need the dict anymore
        modifiers = tuple(modifiers.values())
        findings = tuple(findings.values())

        return findings, modifiers, relations

    @classmethod
    def from_xml(cl, parse_obj, proc_all=False, **kwargs):
        """Parse a EdIE document from XML parser"""

        document, *rest = tuple(parse_obj.iter(EdIEDoc.XML_DOC))
        assert len(rest) == 0

        findings, modifiers, relations = EdIEDoc.build_doc_fields(document, **kwargs)

        # Create an interval tree to easily look up what tags exist
        # for a particular look up index
        find_itree = IntervalTree.from_tuples([(f.start_char_idx, f.end_char_idx, f)
                                               for f in findings])

        mod_itree = IntervalTree.from_tuples([(m.start_char_idx, m.end_char_idx, m)
                                              for m in modifiers])

        sentences = []
        sent_section = 'unknown'
        doc_id = document.get(EdIEDoc.XML_ID, None)
        # Iterate over all children tags that are sentences
        for sent_tag in document.iter(EdIEDoc.XML_SENT):
            sent_id = sent_tag.get(EdIEDoc.XML_ID, None)
            # Some sentences are not to be processed - this is flagged by proc
            process = sent_tag.get('proc', None)
            # Unless we set proc_all, only process proc='yes' sentences.
            if process == 'yes' or proc_all:
                sent = defaultdict(lambda: [])
                for word_tag in sent_tag.iter(EdIEDoc.XML_WORD):
                    # Ignore tokens that are errors / dataset irregularities
                    if word_tag.text in cl.DROP_TOKENS:
                        continue
                    word_id = word_tag.get(EdIEDoc.XML_ID, '')
                    char_idx = int(word_id[1:])
                    # word_string = EdIEDoc.NUM_REPLACE.sub('<d>', word_tag.text)
                    word_string = word_tag.text
                    lemma = word_tag.get(EdIEDoc.XML_LEMMA, '')
                    pos_tag = word_tag.get(EdIEDoc.XML_POS, '')

                    # Process findings
                    negation = 'pos'
                    ner_tag = 'O'
                    # Lookup word id in finding dictionary and default to O
                    intervals = find_itree[char_idx]
                    if intervals:
                        (*rest, ner), *more = intervals
                        assert len(more) == 0, 'Finding annotation overlap'
                        ner_tag = ner.get_bio(char_idx)
                        if ner.negated:
                            negation = 'neg'

                    # Process modifiers
                    mod_tag = 'O'
                    intervals = mod_itree[char_idx]
                    if intervals:
                        (*rest, mod), *more = intervals
                        assert len(more) == 0, 'Modifier annotation overlap'
                        mod_tag = mod.get_bio(char_idx)
                        if mod.negated:
                            negation = 'neg'

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

        labels = EdIEDoc.get_document_labels(document)
        entities = list(chain(findings, modifiers))

        return cl(doc_id=doc_id,
                  sentences=sentences,
                  labels=labels,
                  entities=entities,
                  relations=relations,
                  xml_parse=parse_obj)


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
                    # Below retains number of findings - otherwise we merge neighbouring mentions of same type
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
    def ner_tags_cui(self):
        return tuple(base.tag_to_cui(t, tag_to_cui)
                     for t in self.ner_tags)

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

    def get_entities(self, iob_field):
        entities, entity = [], dict()
        prev_tag = 'O'
        for i, ent in enumerate(iob_field):
            ent_tag = ent[2:] or 'O'
            # Found a start entity
            if ent.startswith('B-'):
                # close if half constructed
                if entity:
                    entity['end'] = i
                    entities.append(entity)
                    entity = dict()
                # start
                entity['start'] = i
                entity['type'] = ent_tag
                if self.negation[i] == 'neg':
                    entity['neg'] = True
            # Changed type
            elif ent_tag != prev_tag:
                # close if half constructed
                if entity:
                    entity['end'] = i
                    entities.append(entity)
                    entity = dict()
                if ent_tag != 'O':
                    # start
                    entity['start'] = i
                    entity['type'] = ent_tag
                    if self.negation[i] == 'neg':
                        entity['neg'] = True
            prev_tag = ent_tag
        # Close any left over entities
        if entity:
            entity['end'] = i + 1
            entities.append(entity)

        entity_objs = []
        word_ids = self.word_ids
        for ent in entities:
            label = ent['type']
            span = (word_ids[ent['start']], word_ids[max(0, ent['end'] - 1)])
            text = ' '.join(self.tokens[ent['start']: ent['end']])
            negated = ent.get('neg', False)

            entity_objs.append(EdIEEntity(label=label,
                                          span=span,
                                          negated=negated,
                                          text=text))

        return entity_objs

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


class EdIEEntity(object):
    """Encapsulate entity (finding/modifier) functionality"""

    NEG_PREFIX = 'neg_'
    MODIFIER_PREFIX_TIME = 'time_'
    MODIFIER_PREFIX_LOC = 'loc_'
    MODIFIER_PREFIXES = (MODIFIER_PREFIX_TIME, MODIFIER_PREFIX_LOC)

    def __init__(self, label, span, negated, text, ent_id=None):
        super().__init__()
        self.label = label
        assert len(span) == 2
        # The span is in terms of word ids
        self.span = span
        # the word ids are of the form w\d+ - where \d+ is the char offset
        self.start_char_idx = int(span[0][1:])
        self.end_char_idx = self.start_char_idx + len(text)
        self.negated = negated
        self.text = text
        self.ent_id = ent_id

    def __repr__(self):
        return '<EdIEEntity:%s> "%s"' % (self.full_type, self.text)

    @property
    def id(self):
        s = '%s%s%s' % (self.label, self.span, self.negated)
        return get_hash(s)

    def get_or_create_id(self):
        # If we assigned an id when we created this class, return that
        if self.ent_id is not None:
            return self.ent_id
        else:
            return self.id

    @property
    def sw(self):
        return self.span[0]

    @property
    def ew(self):
        return self.span[1]

    @property
    def full_type(self):
        if self.negated:
            return '%s%s' % (EdIEEntity.NEG_PREFIX, self.label)
        return self.label

    @property
    def is_modifier_time(self):
        return self.label.startswith(EdIEEntity.MODIFIER_PREFIX_TIME)

    @property
    def is_modifier_loc(self):
        return self.label.startswith(EdIEEntity.MODIFIER_PREFIX_LOC)

    @property
    def is_modifier(self):
        return any(self.label.startswith(p)
                   for p in EdIEEntity.MODIFIER_PREFIXES)

    def to_xml(self):
        ent = ET.Element(EdIEDoc.XML_ENTITY)
        ent.set(EdIEDoc.XML_ENTITY_TYPE, self.full_type)
        ent.set(EdIEDoc.XML_ID, self.get_or_create_id())

        parts = ET.SubElement(ent, EdIEDoc.XML_PARTS)
        part = ET.SubElement(parts, EdIEDoc.XML_PART)
        part.set(EdIEDoc.XML_START_WORD, self.sw)
        part.set(EdIEDoc.XML_END_WORD, self.ew)
        part.text = self.text

        return ent

    @classmethod
    def from_xml(cl, ent_xml):
        ent_type = ent_xml.get(EdIEDoc.XML_ENTITY_TYPE)
        ent_id = ent_xml.get(EdIEDoc.XML_ID)
        # Capture negation and remove neg prefix to create label
        negated = False
        label = ent_type
        if ent_type.startswith(EdIEEntity.NEG_PREFIX):
            negated = True
            label = ent_type[len(EdIEEntity.NEG_PREFIX):]
        # Parse EdIE-R deprecated output cases where neg was attr
        elif ent_xml.attrib.get(EdIEEntity.NEG_PREFIX[:-1], 'no') == 'yes':
            negated = True

        # Parse old EdIE-R format for modifiers
        if label == 'mod':
            stime = ent_xml.attrib.get('stime', None)
            sloc = ent_xml.attrib.get('sloc', None)
            if stime:
                label = '%s%s' % (EdIEEntity.MODIFIER_PREFIX_TIME, stime)
            elif sloc:
                label = '%s%s' % (EdIEEntity.MODIFIER_PREFIX_LOC, sloc)
            else:
                raise ValueError('Invalid old EdIE-R format')

        part, *rest = tuple(ent_xml.iter(EdIEDoc.XML_PART))
        assert(len(rest) == 0)

        start_word_id = part.get(EdIEDoc.XML_START_WORD)
        end_word_id = part.get(EdIEDoc.XML_END_WORD)
        span = (start_word_id, end_word_id)
        text = part.text
        return EdIEEntity(label=label,
                          span=span,
                          negated=negated,
                          text=text,
                          ent_id=ent_id)

    def get_bio(self, char_idx):
        if char_idx == self.start_char_idx:
            tag = 'B-%s' % self.label
        elif char_idx <= self.end_char_idx:
            tag = 'I-%s' % self.label
        else:
            tag = 'O'
        return tag


class EdIERelation(object):
    """A relation between modifiers and findings (head)"""
    def __init__(self, label, head, modifier):
        super().__init__()
        self.label = label
        self.modifier = modifier
        self.head = head

    def __repr__(self):
        fields = (self.label, self.modifier, self.head)
        return '<EdIERelation:%s> %r -> %r' % fields

    def __eq__(self, other):
        eq_labels = self.label == other.label
        eq_head = self.head.id == other.head.id
        eq_mod = self.modifier.id == other.modifier.id
        return all([eq_labels, eq_head, eq_mod])

    def is_valid(self):
        stroke_type = EdIELabel.S in self.head.label
        hem_type = EdIELabel.MH in self.head.label
        return stroke_type or hem_type

    def to_xml(self):

        rel_xml = ET.Element(EdIEDoc.XML_ST_REL)
        rel_xml.set(EdIEDoc.XML_ENTITY_TYPE, self.label)

        head = ET.SubElement(rel_xml, EdIEDoc.XML_ST_ARG)
        head.set(EdIEDoc.XML_TEXT, self.head.text)
        head.set(EdIEDoc.XML_ENTITY_REF, self.head.get_or_create_id())

        mod = ET.SubElement(rel_xml, EdIEDoc.XML_ST_ARG)
        mod.set(EdIEDoc.XML_TEXT, self.modifier.text)
        mod.set(EdIEDoc.XML_ENTITY_REF, self.modifier.get_or_create_id())
        return rel_xml


class EdIELabel(object):
    """A document label"""
    OLD = 'time_old'
    RECENT = 'time_recent'
    DEEP = 'loc_deep'
    CORTICAL = 'loc_cortical'
    S = 'stroke'
    IS = 'ischaemic_stroke'
    HS = 'haemorrhagic_stroke'
    T = 'tumour'
    MT = 'mening_tumour'
    MST = 'metast_tumour'
    GT = 'glioma_tumour'
    SH = 'subarachnoid_haemorrhage'
    SBH = 'subdural_haematoma'
    MH = 'microhaemorrhage'
    HT = 'haemorrhagic_transformation'
    A = 'atrophy'
    SVD = 'small_vessel_disease'

    IS_UNDER = 'Ischaemic stroke, underspecified'
    HS_UNDER = 'Haemorrhagic stroke, underspecified'
    MB_UNDER = 'Microbleed, underspecified'
    TU_OTHER = 'Tumour, other'

    def __init__(self, parts):
        super().__init__()
        # We don't care about order of parts
        self.parts = set(parts)

    def __eq__(self, other):
        return self.text == other.text

    def match(self, *candidate_tags):
        return all(getattr(EdIELabel, each) in self.parts
                   for each in candidate_tags)

    @property
    def text(self):
        txt = None
        if self.match('IS', 'DEEP'):
            if self.match('RECENT'):
                txt = 'Ischaemic stroke, deep, recent'
            elif self.match('OLD'):
                txt = 'Ischaemic stroke, deep, old'
            # when ischaemic stroke and deep but no time, then infer old
            else:
                txt = 'Ischaemic stroke, deep, old'
        elif self.match('IS', 'CORTICAL', 'RECENT'):
            txt = 'Ischaemic stroke, cortical, recent'
        elif self.match('IS', 'CORTICAL', 'OLD'):
            txt = 'Ischaemic stroke, cortical, old'
        elif self.match('IS'):
            txt = self.IS_UNDER
        elif self.match('HS', 'DEEP', 'RECENT'):
            txt = 'Haemorrhagic stroke, deep, recent'
        elif self.match('HS', 'DEEP', 'OLD'):
            txt = 'Haemorrhagic stroke, deep, old'
        elif self.match('HS', 'CORTICAL', 'RECENT'):
            txt = 'Haemorrhagic stroke, lobar, recent'
        elif self.match('HS', 'CORTICAL', 'OLD'):
            txt = 'Haemorrhagic stroke, lobar, old'
        elif self.match('HS'):
            txt = self.HS_UNDER
        elif self.match('S'):
            txt = 'Stroke, underspecified'
        elif self.match('MT'):
            txt = 'Tumour, meningioma'
        elif self.match('MST'):
            txt = 'Tumour, metastasis'
        elif self.match('GT'):
            txt = 'Tumour, glioma'
        elif self.match('T'):
            txt = self.TU_OTHER
        elif self.match('SVD'):
            txt = 'Small vessel disease'
        elif self.match('A'):
            txt = 'Atrophy'
        elif self.match('SBH'):
            txt = 'Subdural haematoma'
        # NOTE: we can't check for aneurysmal at this stage as EdIE-R does
        # we don't have the information here.
        # TODO: Subarachnoid haemorrhage, aneurysmal
        elif self.match('SH'):
            txt = 'Subarachnoid haemorrhage, other'
        elif self.match('MH', 'DEEP'):
            txt = 'Microbleed, deep'
        elif self.match('MH', 'CORTICAL'):
            txt = 'Microbleed, lobar'
        elif self.match('MH'):
            txt = self.MB_UNDER
        elif self.match('HT'):
            txt = 'Haemorrhagic transformation'
        return txt

    def to_xml(self):

        label_xml = ET.Element(EdIEDoc.XML_ENTITY)
        label_xml.set(EdIEDoc.XML_ENTITY_TYPE, 'label:%s' % self.text)
        label_xml.set(EdIEDoc.XML_ID, self.get_or_create_id())

        return label_xml
