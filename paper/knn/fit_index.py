import os
import numpy as np
import torch
import tqdm


from collections import namedtuple

from mlconf import YAMLLoaderAction, ArgumentParser
from edien import EdIENPath
from edien.components import BertSentenceEncoder
from edien.preprocess import PaddedVariabliser
from edien.vocab import BertCoder
from edien.train_utils import only_pseudorandomness_please
from utils import FaissIndex


if __name__ == "__main__":

    parser = ArgumentParser(description='Fit faiss index on training set')
    parser.add_argument('--save_index', required=True, type=str,
                        help='The name to give to the FAISS index folder')
    parser.add_argument('--load_blueprint', action=YAMLLoaderAction)
    conf = parser.parse_args()
    only_pseudorandomness_please(conf.seed)

    bp = conf.build()

    train = bp.data.train_vars(bp.paths)
    train_sents = bp.data.train_sents

    # Get BERT embedding dimension
    dim = bp.model.model.tasks['ner_tags'].in_dim

    if conf.device.startswith('cuda'):
        # Set default gpu
        bp.model.to(conf.device)

    # Create faiss index.
    index = FaissIndex(dim)

    Entry = namedtuple('Entry', ['tokens', 'word_mask', 'ner_tags'])
    Output = namedtuple('Output', ['ner_tags'])

    print('Fitting Faiss index..')
    for tokens, mask, ner_tags, s in tqdm.tqdm(zip(train.tokens,
                                                   train.word_mask,
                                                   train.ner_tags,
                                                   train_sents), 
                                               total=len(train.tokens)):

        entry = Entry([tokens], [mask], [ner_tags])

        data_encoder = PaddedVariabliser(device=conf.device)
        data = data_encoder.encode(entry)
        sent_lens = data_encoder.sequence_lengths['ner_tags']

        outputs = bp.model.model.encoder(data, sent_lens=sent_lens)
        outputs = outputs.detach()

        output = Output(outputs)
        result = data_encoder.decode(output).ner_tags[0]
        index.add(result, s)

    index.save(conf.save_index)
    prev = index.label_lookup
