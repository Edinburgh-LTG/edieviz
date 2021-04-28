import fileinput


if __name__ == "__main__":
    TP = 'True Positives'
    FN = 'False Negatives'
    FP = 'False Positives'
    TN = 'True Negatives'

    counts = dict()
    next_stat = None
    for line in fileinput.input():
        line = line.rstrip()
        if line.startswith('*'):
            assert next_stat is not None
            _, count = line.rsplit(maxsplit=1)
            counts[next_stat] = int(count)
        else:
            next_stat = line
    total = counts[TP] + counts[FN] + counts[FP] + counts[TN]
    # Specificity
    specificity = float(counts[TN]) / (counts[TN] + counts[FP]) * 100

    # Negative Predictive Value
    npv = float(counts[TN]) / (counts[TN] + counts[FN]) * 100

    # Precision
    precision = float(counts[TP]) / (counts[TP] + counts[FP]) * 100

    # Recall
    recall = float(counts[TP]) / (counts[TP] + counts[FN]) * 100

    # F1
    f1 = 2 * (precision * recall) / (precision + recall)

    # Accuracy
    acc = (float(counts[TP]) + counts[TN]) / total * 100

    print('S   = %.2f' % specificity)
    print('NPV = %.2f' % npv)
    print('P   = %.2f' % precision)
    print('R   = %.2f' % recall)
    print('F1  = %.2f' % f1)
    print('Acc = %.2f' % acc)
    print()

    print('%.2f & %.2f & %.2f & %.2f & %.2f & %.2f' % (specificity,
                                                       npv,
                                                       precision,
                                                       recall,
                                                       f1,
                                                       acc))
