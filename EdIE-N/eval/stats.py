def stats(f):
    TP, FP, FN = 0,0,0
    results = []
    for line in open(f).readlines():
        if line.startswith("ALL"):
            tp, fp, fn = line.split(";")[:3]
            TP += int(tp.split(" ")[-1])
            FP += int(fp.split(" ")[-1])
            FN += int(fn.split(" ")[-1])
        else:
            if "TP:" in line:
                results.append(line.strip())
    try:
        prec = TP/(TP+FP)
    except Exception:
        prec = 0.
    try:
        rec = TP/(TP+FN)
    except Exception:
        rec = 0.
    try:
        f1 = 2*prec*rec/(prec+rec)
    except Exception:
        f1 = 0.
    results.sort(key=lambda x: x.split(" "))
    for x in results:
        print(x)
    print(f"TOTAL TP: {TP}; FP: {FP}; FN: {FN}; precision {prec*100:.2f}; recall {rec*100:.2f}; FB1 {f1*100:.2f}")

if __name__ == "__main__":
    import sys
    stats(sys.argv[1])
