"""
round_csv.py
------------
Arrondit les colonnes numériques d'un CSV wide à N décimales.

Usage :
    python round_csv.py --input data\cov\cwb\cwb2025_cumsum.csv --decimals 3
"""

import argparse
import pandas as pd
import os

def round_csv(input_path: str, decimals: int) -> None:

    # Nom du fichier de sortie
    base, ext    = os.path.splitext(input_path)
    output_path  = f"{base}_rounded{ext}"

    print(f"📂 Chargement : {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Shape : {df.shape}")

    # Colonnes à arrondir (toutes sauf cell, latitude, longitude)
    cols_to_round = [c for c in df.columns if c not in ('cell', 'latitude', 'longitude')]

    df[cols_to_round] = df[cols_to_round].round(decimals)

    df.to_csv(output_path, index=False)

    size_in  = os.path.getsize(input_path)  / 1024**2
    size_out = os.path.getsize(output_path) / 1024**2

    print(f"\n✅ Fichier sauvegardé : {output_path}")
    print(f"   Décimales          : {decimals}")
    print(f"   Taille originale   : {size_in:.1f} Mo")
    print(f"   Taille arrondie    : {size_out:.1f} Mo")
    print(f"   Gain               : {size_in - size_out:.1f} Mo ({100*(1 - size_out/size_in):.0f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arrondit un CSV wide à N décimales')
    parser.add_argument('--input',    required=True, type=str, help='Chemin du CSV à arrondir')
    parser.add_argument('--decimals', default=3,     type=int, help='Nombre de décimales (défaut: 3)')
    args = parser.parse_args()

    round_csv(args.input, args.decimals)