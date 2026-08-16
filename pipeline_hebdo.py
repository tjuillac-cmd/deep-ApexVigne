"""
Exécute un unique notebook (paramétré par MODEL_TAG) pour chaque semaine
du 16 juin au 17 août 2026, pour les deux variantes de modèle (classique
et dégradé), via papermill.

À lancer depuis un terminal :
    python pipeline_hebdo.py

Chaque exécution produit :
- une copie exécutée du notebook dans executed_notebooks/ (traçabilité)
- un fichier results/{tag}_{date}.pkl contenant le résumé tabulé (avec AUC-ROC et Kendall tau)
ainsi qu'un dictionnaire descriptif des prédictions.
"""

import papermill as pm
from datetime import datetime, timedelta
from pathlib import Path

NOTEBOOK_PATH = "model_apex8.ipynb"

# deux variantes à exécuter, valeurs attendues par la cellule taggée "parameters"
MODEL_TAGS = ["classique", "degrade"]

DATE_DEBUT = datetime(2025, 6, 16)
DATE_FIN_LIMITE = datetime(2025, 8, 17)

# génère la liste des débuts de semaines
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
        print(f"- {tag} : semaine du {date_debut_str}")
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
            print(f"! Échec pour {tag} / {date_debut_str} : {e}")
            echecs.append((tag, date_debut_str, str(e)))

print("\nTerminé")
if echecs:
    print(f"{len(echecs)} échec(s) :")
    for tag, date_debut_str, err in echecs:
        print(f"  - {tag} / {date_debut_str} : {err}")
else:
    print("Toutes les semaines ont été traitées avec succès.")