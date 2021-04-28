import json
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description='Plot error types')

    parser.add_argument('--results_file',
                        type=str,
                        required=True,
                        help='Path to results file')

    args = parser.parse_args()

    # Load CONLL file
    results = dict()
    with open(args.results_file, 'r') as f:
        for l in f:
            k, v = json.loads(l.rstrip())
            results[k] = v

        pass
    print(results)
    fig, ax = plt.subplots(figsize=(8, 6))

    keys = ['FP', 'FN', 'LABEL_ERROR', 'BOUNDARY_ERROR', 'LABEL+BOUNDARY_ERROR']
    view = ['FP', 'FN', 'Label', 'Boundary', 'Label & Boundary']
    labels = [r.replace('_', ' ') for r in results.keys()]
    idxs = dict(zip(results.keys(), [0, 1, 2.5, 3.5, 5, 6]))
    colours = dict(zip(view, [cm.tab20(x) for x in range(len(view))]))
    for k, counts in results.items():
        vals = np.array([counts.get(k, 0) for k in keys])
        relative_freqs = (vals / vals.sum()) * 100.
        prev_freq = None
        for met, val, relative_freq in zip(view, vals, relative_freqs):
            ax.barh(idxs[k], relative_freq, label=met, color=colours[met] ,left=prev_freq)
            if prev_freq is None:
                prev_freq = relative_freq
            else:
                prev_freq += relative_freq

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0)
    handles, lbls = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(),
              by_label.keys(),
              loc='upper center',
              bbox_to_anchor=(0.3, 1.15),
              ncol=5)
    plt.ylabel('Models')
    plt.xlabel('Distribution of error types (%)')
    plt.yticks(tuple(idxs.values()), labels)
    plt.xlim([0, 115])
    plt.show()
