import pandas as pd

# 1. Charger les deux fichiers CSV
df_a = pd.read_csv("data/cov/cwb_cumsum_wide.csv")
df_b = pd.read_csv("data/cov/cwb/correction_drome/drome2025_cumsum.csv")

# 2. Définir la colonne "cell" comme index pour l'alignement des lignes
df_a.set_index("cell", inplace=True)
df_b.set_index("cell", inplace=True)

# 3. Écraser les valeurs de A par celles de B là où l'index "cell" correspond
df_a.update(df_b)

# 4. Réinitialiser l'index pour retrouver "cell" comme une colonne normale
df_a.reset_index(inplace=True)

# 5. Sauvegarder le fichier A corrigé
df_a.to_csv("data/cov/cwb/cwb2025_cumsum_CORRECTED.csv", index=False)

print("Le fichier A a été mis à jour avec succès !")