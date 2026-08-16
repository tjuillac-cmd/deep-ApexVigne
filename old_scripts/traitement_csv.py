import pandas as pd

csv_name = "data/cov/siclima_extraction_3585_20260612.csv"
coords_name = "data/cov/coordonnees_mailles_safran.csv"

COVARIABLES = ["dli_q", "t_q", "co2", "drainc_q", "pe_q", "preliq_q", "wg_racine_q",
               "tsup_h_q", "tinf_h_q", "etp_q", "etppm", "evap_q", "hu_q", "runc_q", "swi_q", "q_q"]

# Ces colonnes seront remplacées par leur cumul
COVARIABLES_CUMUL = ["dli_q", "drainc_q", "pe_q", "preliq_q", "etp_q", "etppm", "evap_q", "runc_q"]

df = pd.read_csv(csv_name, sep=";")
coords = pd.read_csv(coords_name, sep=";")

df["date"] = pd.to_datetime(df[["year", "month", "day_of_month"]].rename(columns={"day_of_month": "day"}))
df["date"] = df["date"].dt.strftime("%Y-%m-%d")

df = df.sort_values(["cell", "date"]).reset_index(drop=True)

# Remplacement des colonnes brutes par leur cumul
for col in COVARIABLES_CUMUL:
    df[col] = df.groupby(["cell", "year"])[col].cumsum()

# Le vecteur utilise COVARIABLES inchangé, mais les colonnes cumulées ont remplacé les brutes
df["vecteur"] = df[COVARIABLES].values.tolist()

for year, df_year in df.groupby("year"):
    df_pivot = df_year.pivot(index="cell", columns="date", values="vecteur")
    df_pivot = df_pivot.reset_index()

    df_pivot = df_pivot.merge(coords, on="cell", how="left")
    date_cols = [c for c in df_pivot.columns if c not in ("cell", "latitude", "longitude")]
    df_pivot = df_pivot[["cell", "latitude", "longitude"] + date_cols]

    df_pivot.to_csv(f"data/cov/cov_siclima_{year}.csv", index=False)
    print(f"✓ cov_siclima_{year}.csv — {df_pivot.shape[0]} cellules, {df_pivot.shape[1]-3} dates")