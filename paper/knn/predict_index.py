import numpy as np
import torch
import tqdm

from collections import namedtuple

from mlconf import YAMLLoaderAction, ArgumentParser
from edien.components import BertSentenceEncoder
from edien.preprocess import PaddedVariabliser
from edien.train_utils import only_pseudorandomness_please
from edien.vocab import BertCoder
from utils import FaissIndex, k_nearest_interpolation, entropy, argmax


if __name__ == "__main__":

    parser = ArgumentParser(description='Use KNN with faiss index to obtain '
                            'interpolated probabilities for labels.')
    parser.add_argument('--K', default=10, type=str,
                        help='The number of nearest neighbours to use')
    parser.add_argument('--load_index', required=True, type=str,
                        help='The folder to load the FAISS index from')
    parser.add_argument('--load_blueprint', action=YAMLLoaderAction)

    conf = parser.parse_args()
    only_pseudorandomness_please(conf.seed)

    # Make sure we aren't trying to create vocab on test
    conf.data.vocab_encoder.load_if_exists = True

    print('Loading model specified in "%s"' % conf.load_blueprint)
    bp = conf.build()
    test = bp.data.test_vars(bp.paths)
    test_sents = bp.data.test_sents

    # Load faiss index.
    print('Loading faiss index from "%s"' % conf.load_index)
    index = FaissIndex.load(conf.load_index)
    print('Using %d nearest neighbours' % conf.K)

    hit, total = 0, 0

    Entry = namedtuple('Entry', ['tokens', 'word_mask', 'ner_tags'])
    Output = namedtuple('Output', ['ner_tags'])

    for tokens, mask, ner_tags, s in tqdm.tqdm(zip(test.tokens,
                                                   test.word_mask,
                                                   test.ner_tags,
                                                   test_sents),
                                               total=len(test_sents)):

        entry = Entry([tokens], [mask], [ner_tags])

        data_encoder = PaddedVariabliser()
        data = data_encoder.encode(entry)
        sent_lens = data_encoder.sequence_lengths['ner_tags']

        outputs = bp.model.model.encoder(data, sent_lens=sent_lens)
        outputs = outputs.detach()

        output = Output(outputs)
        # Obtain encoder outputs for each token
        result = data_encoder.decode(output).ner_tags[0]

        # get nearest neighbours
        dists, idxs = index.index.search(result, k=conf.K)
        for i, (dist, idx) in enumerate(zip(dists, idxs)):
            total += 1
            ner_labels = [index.sentence_lookup[match].ner_tags[index.label_lookup[match]] for match in idx]
            probs = k_nearest_interpolation(dist, ner_labels)
            # print(entropy(probs), probs)
            print(s.tokens[i], s.ner_tags[i], argmax(probs), probs)
            # print(s.tokens[i], s.tokens[i], s.ner_tags[i], argmax(probs))
            if s.ner_tags[i] == argmax(probs):
                hit += 1
        print()
    print('Accuracy: %.2f' % ((hit / max(1, total)) * 100))
