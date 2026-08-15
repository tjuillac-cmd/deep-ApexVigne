import pickle
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

bornes_par_tag = defaultdict(lambda: {
    "proba_min": [], "proba_max": [], "proba_q99": [],
    "std_min": [], "std_max": [], "std_q99": [],
})

for pkl_path in Path("results").glob("*.pkl"):
    tag = pkl_path.stem.split("_")[0]  # "classique_2025-06-16.pkl" -> "classique"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for k in bornes_par_tag[tag]:
        bornes_par_tag[tag][k].append(data[k])

resultat = {}
for tag, vals in bornes_par_tag.items():
    resultat[tag] = {
        "v_min_score": min(vals["proba_min"]),
        "v_max_score": max(vals["proba_q99"]),   # approximation du q99 global
        "v_max_score_strict": max(vals["proba_max"]),  # vrai max, si besoin
        "v_min_std": min(vals["std_min"]),
        "v_max_std": max(vals["std_q99"]),
        "v_max_std_strict": max(vals["std_max"]),
    }

with open("bornes_couleurs_reference.json", "w") as f:
    json.dump(resultat, f, indent=2)

print(json.dumps(resultat, indent=2))