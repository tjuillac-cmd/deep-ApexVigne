"""
Calcule des bornes de couleur (min/max/q01/q99) pour les cartes de
probabilité et d'incertitude, à partir de l'historique complet des .pkl
générés par le pipeline hebdomadaire, séparément pour chaque modèle
(classique/dégradé).

Les bornes basse (q01) et haute (q99) sont utilisées par défaut au lieu du
min/max strict, pour ne pas laisser quelques valeurs extrêmes écraser toute
l'échelle de couleur. Ces bornes servent à figer une échelle commune à
toutes les semaines, pour que les cartes restent comparables entre elles
dans le temps.

À lancer depuis un terminal, après avoir exécuté pipeline_hebdo.py :
    python3 calcul_bornes_couleurs.py

Produit :
- bornes_couleurs_reference.json, avec une entrée par tag de modèle
  ({"classique": {...}, "degrade": {...}})
"""

import pickle
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# jeu de listes par tag de modèle (classique/degrade), rempli au fil des .pkl
bornes_par_tag = defaultdict(lambda: {
    "proba_min": [], "proba_max": [], "proba_q01": [], "proba_q99": [],
    "std_min": [], "std_max": [], "std_q01": [], "std_q99": [],
})

# parcourt tous les résultats hebdomadaires déjà générés
for pkl_path in Path("results").glob("*.pkl"):
    tag = pkl_path.stem.split("_")[0]  # "classique_2025-06-16.pkl" -> "classique"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # Accumule les extrêmes de cette semaine dans les listes du tag correspondant
    for k in bornes_par_tag[tag]:
        bornes_par_tag[tag][k].append(data[k])

# Agrège les extrêmes hebdomadaires en bornes globales, par tag de modèle
resultat = {}
for tag, vals in bornes_par_tag.items():
    resultat[tag] = {
        "v_min_score": min(vals["proba_q01"]),           
        "v_max_score": max(vals["proba_q99"]),           
        "v_min_score_strict": min(vals["proba_min"]),
        "v_max_score_strict": max(vals["proba_max"]),
        "v_min_std": min(vals["std_q01"]),
        "v_max_std": max(vals["std_q99"]),
        "v_min_std_strict": min(vals["std_min"]),
        "v_max_std_strict": max(vals["std_max"]),
    }

# Sauvegarde pour réutilisation dans les cellules de plot des notebooks
with open("bornes_couleurs_reference.json", "w") as f:
    json.dump(resultat, f, indent=2)

print(json.dumps(resultat, indent=2))