import numpy as np
from collections import defaultdict, Counter
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description='Calculate and plot loose score')

    args = parser.parse_args()

    labels = ['O', 'B-PERSON', 'I-PERSON', 'B-LOCATION', 'I-LOCATION', 'O', 'O', 'O']
    gold = np.random.randint(0, 8, 100000)
    pred = np.random.randint(0, 8, 100000)

    lines = []
    for i, (g, p) in enumerate(zip(gold, pred)):
        lines.append('%d %d %s %s\n' % (i, i, labels[g], labels[p]))
        if np.random.randn() > .9:
            lines.append('\n')
    with open('conll_example', 'w') as f:
        f.writelines(lines)
