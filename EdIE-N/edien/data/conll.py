import os
from dataclasses import dataclass
from edien.data import base


class CoNLLLoader(object):

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._sentences = []

    def __repr__(self):
        template = '<CoNLLLoader %s : %d sentences>'
        parts = (self.filename, len(self.sentences))
        return template % parts

    @property
    def sentences(self):
        return tuple(self._sentences)

    @classmethod
    def load(cl, filename):
        sentences = []
        num_sents = 0
        with open(filename, 'r') as f:
            parts = []
            for line in f.readlines():
                line = line.rstrip()
                if line:
                    parts.append(line.split('\t'))
                else:
                    sent = CoNLLSent(*zip(*parts), sent_id=str(num_sents))
                    if len(sent.tokens) > 1:
                        sentences.append(sent)
                    parts = []
                    num_sents += 1
        loader = cl(filename)
        loader._sentences = sentences
        return loader


class CoNLLDataset(base.Dataset):
    pass


@dataclass(frozen=True)
class CoNLLSent(base.Sentence):
    ner_tags: tuple
    sent_id: str
