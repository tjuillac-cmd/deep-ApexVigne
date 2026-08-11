import pandas as pd

df = pd.read_csv(r"cov\cwb\correction_drome\drome2025.csv", sep=";", decimal=",")

print(df.columns.tolist())
print(df.head())

# Filtrer uniquement du 1er janvier au 31 août
df = df[df["month"].isin([1, 2, 3, 4, 5, 6, 7, 8])]

# Créer la colonne somme journalière
df["cwb"] = df["pe_q"] - df["evap_q"]

# Agréger par cellule et par année
resultat = df.groupby(["cell", "year"]).agg(
    cwb_annuel=("cwb", "sum"),
    latitude=("latitude", "first"),
    longitude=("longitude", "first")
).reset_index()

resultat.to_csv(r"cov\cwb\correction_drome\resultat_annuel_cwb.csv", index=False)