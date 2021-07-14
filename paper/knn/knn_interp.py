import numpy as np

from collections import defaultdict
from scipy.special import logsumexp
from utils import k_nearest_interpolation, entropy


if __name__ == "__main__":

    labels = ['a', 'b', 'c']
    dists = [1, 2, 3]
    probs = k_nearest_interpolation(dists, labels)
    print('Probabilities: %r' % probs)
    print('Entropy: %.2f' % entropy(probs))

    # Note that you could have many nearest neighbours that have the same label
    labels = ['a', 'a', 'b', 'c']
    dists = [1, 2, 2, 3]
    probs = k_nearest_interpolation(dists, labels)
    print('Probabilities: %r' % probs)
    print('Entropy: %.2f' % entropy(probs))
