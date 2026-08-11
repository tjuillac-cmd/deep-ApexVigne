import glob, pickle
for fp in sorted(glob.glob("results/classique_*.pkl")):
    with open(fp, "rb") as f:
        d = pickle.load(f)
    print(fp, "oof_score_combined" in d)