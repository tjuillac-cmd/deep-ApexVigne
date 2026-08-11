"""
Agrège les résultats hebdomadaires (results/{tag}_{date}.pkl, produits par
2_run_weekly_pipeline.py) dans un fichier Excel avec deux feuilles
("modele" et "modele_degrade"), au même format que le fichier
kendall_tau.xlsx montré précédemment.

Les colonnes top1/top2/top3 correspondent au rang du fold selon le Kendall
tau (le meilleur d'abord), PAS au numéro d'origine du fold dans la cross-
validation — cohérent avec le fait que df_summary, côté notebook, ne garde
déjà que les 3 meilleurs folds triés par Kendall tau avant export.

À lancer depuis un terminal (dans l'environnement conda `apex`), après
2_run_weekly_pipeline.py :
    python 3_aggregate_to_excel.py
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --- Doit être le même dossier que PROJECT_DIR dans 2_run_weekly_pipeline.py ---
PROJECT_DIR = Path(r"C:\Users\juillact\Documents\deep-ApexVigne")

DATE_DEBUT = datetime(2025, 6, 16)
DATE_FIN_LIMITE = datetime(2025, 8, 17)

semaines = []
d = DATE_DEBUT
while d < DATE_FIN_LIMITE:
    semaines.append(d.strftime("%Y-%m-%d"))
    d += timedelta(days=7)

METRICS = [("AUC-ROC", "auc"), ("Kendall tau", "kt")]


def build_week_row(date_debut_str, export_data):
    df_summary = export_data["df_summary"]
    row = {
        "date_deb": pd.to_datetime(date_debut_str),
        "date_fin": pd.to_datetime(date_debut_str) + pd.Timedelta(days=6),
        "nb_obs": df_summary["Nb observations"].iloc[0],
        "nb_cell": df_summary["Nb cellules observées"].iloc[0],
    }
    for metric_col, prefix in METRICS:
        vals = df_summary[metric_col]  # déjà trié par Kendall tau décroissant en amont
        for rank in [1, 2, 3]:
            pos = rank - 1
            row[f"{prefix}_top{rank}"] = vals.iloc[pos] if pos < len(vals) else np.nan
        valid = vals.dropna()
        row[f"{prefix}_moy"] = valid.mean() if len(valid) else np.nan
        row[f"{prefix}_std"] = valid.std() if len(valid) else np.nan
    row["ic_apex_moy"] = export_data["ic_apex_moy"]
    row["std_ic_apex"] = export_data["std_ic_apex"]
    return row


def aggregate(tag, results_dir="results"):
    rows = []
    for date_debut_str in semaines:
        path = Path(results_dir) / f"{tag}_{date_debut_str}.pkl"
        if not path.exists():
            print(f"⚠️ Résultat manquant, semaine ignorée : {path}")
            continue
        with open(path, "rb") as f:
            export_data = pickle.load(f)
        rows.append(build_week_row(date_debut_str, export_data))
    return pd.DataFrame(rows)


def write_sheet(writer, df, sheet_name):
    """Écrit le DataFrame avec un double en-tête groupé, comme le fichier original."""
    df.to_excel(writer, sheet_name=sheet_name, startrow=2, header=False, index=False)
    ws = writer.sheets[sheet_name]

    groups = (
        ["Date", ""] + ["Effectifs", ""]
        + ["AUC-ROC"] + [""] * 4
        + ["Kendall tau"] + [""] * 4
        + ["iC-apex", ""]
    )
    subheaders = (
        ["date_deb", "date_fin", "nb_obs", "nb_cell"]
        + ["top1", "top2", "top3", "moy", "std"]
        + ["top1", "top2", "top3", "moy", "std"]
        + ["ic_apex_moy", "std_ic_apex"]
    )
    for col_idx, (g, s) in enumerate(zip(groups, subheaders), start=1):
        ws.cell(row=1, column=col_idx, value=g)
        ws.cell(row=2, column=col_idx, value=s)


df_classique = aggregate("classique")
df_degrade = aggregate("degrade")

output_path = "resultats_hebdo.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    write_sheet(writer, df_classique, "modele")
    write_sheet(writer, df_degrade, "modele_degrade")

print(f"\n✅ Export terminé : {output_path}")
print(f"  - {len(df_classique)}/{len(semaines)} semaines pour le modèle classique")
print(f"  - {len(df_degrade)}/{len(semaines)} semaines pour le modèle dégradé")