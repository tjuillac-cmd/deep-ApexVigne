import pickle, glob
import numpy as np
from sklearn.isotonic import IsotonicRegression

def load_weekly_oof(model_tag, results_dir="results"):
    rows = []
    for fp in sorted(glob.glob(f"{results_dir}/{model_tag}_*.pkl")):
        with open(fp, "rb") as f:
            d = pickle.load(f)
        if "oof_score_combined" not in d:
            print(f"⚠️ {fp} sans données OOF (ancien format) — ignoré")
            continue
        rows.append({"date_debut": d["date_debut_str"],
                      "score": d["oof_score_combined"],
                      "ic_apex": d["oof_ic_apex_true"]})
    rows.sort(key=lambda r: r["date_debut"])
    return rows

def fit_weekly_calibration(weekly_data, min_n=150):
    calibrators = {}
    n_weeks = len(weekly_data)
    for i, wk in enumerate(weekly_data):
        lo, hi = i, i
        s, y = list(wk["score"]), list(wk["ic_apex"])
        while len(s) < min_n and (lo > 0 or hi < n_weeks - 1):
            if lo > 0:
                lo -= 1
                s += list(weekly_data[lo]["score"]); y += list(weekly_data[lo]["ic_apex"])
            if len(s) < min_n and hi < n_weeks - 1:
                hi += 1
                s += list(weekly_data[hi]["score"]); y += list(weekly_data[hi]["ic_apex"])
        calib = IsotonicRegression(out_of_bounds='clip').fit(s, y)
        calibrators[wk["date_debut"]] = {
            "model": calib, "n_points": len(s),
            "weeks_used": [weekly_data[j]["date_debut"] for j in range(lo, hi + 1)],
        }
    return calibrators

if __name__ == "__main__":
    for tag in ["classique", "degrade"]:
        weekly = load_weekly_oof(tag)
        if not weekly:
            print(f"⏩ Aucune donnée OOF pour {tag}"); continue
        calibrators = fit_weekly_calibration(weekly, min_n=150)
        for date, c in calibrators.items():
            print(f"[{tag}] {date}: n={c['n_points']:4d}  (semaines poolées: {c['weeks_used']})")
        with open(f"results/calibrators_{tag}.pkl", "wb") as f:
            pickle.dump(calibrators, f)
        print(f"✅ results/calibrators_{tag}.pkl")
