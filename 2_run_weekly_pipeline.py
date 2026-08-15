"""
Exécute un unique notebook (paramétré par MODEL_TAG) pour chaque semaine
du 16 juin au 17 août 2026, pour les deux variantes de modèle (classique
et dégradé), via papermill.

À lancer depuis un terminal (dans l'environnement conda `apex`) :
    python 2_run_weekly_pipeline.py

Chaque exécution produit :
- une copie exécutée du notebook dans executed_notebooks/ (traçabilité)
- un fichier results/{tag}_{date}.pkl contenant df_summary + iC-apex
  (grâce à la cellule d'export ajoutée dans le notebook — voir
  1_modifications_notebooks.md)
"""

import papermill as pm
from datetime import datetime, timedelta
from pathlib import Path

# --- Adapte ce chemin à ton notebook réel ---
NOTEBOOK_PATH = "model_apex8.ipynb"

# --- Les deux variantes à exécuter, valeurs attendues par la cellule taggée "model_tag" ---
MODEL_TAGS = ["classique", "degrade"]

DATE_DEBUT = datetime(2025, 6, 16)
DATE_FIN_LIMITE = datetime(2025, 8, 17)  # borne exclusive du dernier date_fin

# --- Génère la liste des lundis (dates de début de semaine) ---
semaines = []
d = DATE_DEBUT
while d < DATE_FIN_LIMITE:
    semaines.append(d.strftime("%Y-%m-%d"))
    d += timedelta(days=7)

print(f"{len(semaines)} semaines à traiter : {semaines}\n")

Path("executed_notebooks").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

echecs = []

for tag in MODEL_TAGS:
    for date_debut_str in semaines:
        output_path = f"executed_notebooks/{tag}_{date_debut_str}.ipynb"
        print(f"▶ {tag} — semaine du {date_debut_str}")
        try:
            pm.execute_notebook(
                NOTEBOOK_PATH,
                output_path,
                parameters={
                    "MODEL_TAG": tag,
                    "date_debut_str": date_debut_str,
                },
            )
        except Exception as e:
            print(f"  ⚠️ Échec pour {tag} / {date_debut_str} : {e}")
            echecs.append((tag, date_debut_str, str(e)))

print("\n--- Terminé ---")
if echecs:
    print(f"{len(echecs)} échec(s) :")
    for tag, date_debut_str, err in echecs:
        print(f"  - {tag} / {date_debut_str} : {err}")
else:
    print("Toutes les semaines ont été traitées avec succès.")