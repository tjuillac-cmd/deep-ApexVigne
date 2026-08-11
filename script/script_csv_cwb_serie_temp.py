import pandas as pd

df = pd.read_csv("data\cov\cwb\cwb2025_CORRIGE_FINAL.csv", sep=";", decimal=",")  # adapte le séparateur si besoin

# Créer la colonne date
df["date"] = pd.to_datetime(df[["year", "month", "day_of_month"]].rename(columns={
    "year": "year", "month": "month", "day_of_month": "day"
})).dt.strftime("%Y-%m-%d")

# Pivot : une ligne par maille, une colonne par date
df_pivot = df.pivot_table(
    index=["cell", "latitude", "longitude"],
    columns="date",
    values="cwb_cumsum",  # ou "cwb" selon ce que tu veux
    aggfunc="first"
).reset_index()

# Aplatir les noms de colonnes
df_pivot.columns.name = None

df_pivot.to_csv("grille_pivot.csv", index=False)
print(f"✅ {len(df_pivot)} mailles, {len(df_pivot.columns)-3} dates")