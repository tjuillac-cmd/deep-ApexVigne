import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# 1. Chargement de ton fichier CSV
csv_name = "data/cov/cwb_cumsum_wide.csv"
df = pd.read_csv(csv_name)

print(f"📊 Fichier d'origine : {len(df)} mailles à 8 km.")

# Séparation des coordonnées et des dates
coord_cols = ["cell", "latitude", "longitude"]
date_cols = [col for col in df.columns if col not in coord_cols]

# 2. Calcul de la résolution d'1 km en degrés selon ta latitude moyenne
lat_moyenne = df["latitude"].mean()
res_lat_deg = 1 / 111.11
res_lon_deg = 1 / (111.11 * np.cos(np.radians(lat_moyenne)))

# 3. Création de la GRILLE FINE (1 km) sur l'emprise de tes données
lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
lat_min, lat_max = df["latitude"].min(), df["latitude"].max()

grid_lon = np.arange(lon_min - res_lon_deg, lon_max + res_lon_deg, res_lon_deg)
grid_lat = np.arange(lat_min - res_lat_deg, lat_max + res_lat_deg, res_lat_deg)

# Produit cartésien pour obtenir tous les points 1km possibles
lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)
df_grid_1km = pd.DataFrame({
    "longitude": lon_mesh.ravel(),
    "latitude": lat_mesh.ravel()
})

# 4. Approche NN stricte avec un KD-Tree
# On construit l'arbre avec les coordonnées 8 km d'origine
tree = cKDTree(df[["longitude", "latitude"]].values)

# Pour chaque point de 1 km, on cherche l'index du point 8 km le plus proche
# distance_upper_bound=0.08 s'assure qu'on ne va pas chercher une maille trop loin hors de ta zone
distances, indices = tree.query(
    df_grid_1km[["longitude", "latitude"]].values, 
    k=1, 
    distance_upper_bound=0.08 
)

# On ne garde que les points 1km qui sont bien à l'intérieur ou en bordure immédiate de tes mailles 8km
valid_mask = distances != np.inf
df_grid_1km = df_grid_1km[valid_mask].copy()
indices_valid = indices[valid_mask]

# 5. Assignation des 200+ colonnes de données d'un seul coup
print("⚡ Duplication des données temporelles sur la nouvelle grille...")
# On copie les données de la maille 8km la plus proche pour chaque ligne 1km
df_data_mapped = df[date_cols].iloc[indices_valid].reset_index(drop=True)

# Assemblage final
df_1km_final = pd.concat([df_grid_1km.reset_index(drop=True), df_data_mapped], axis=1)

# Recréation de la colonne 'cell' incrémentale
df_1km_final.insert(0, 'cell', range(1, len(df_1km_final) + 1))

# 6. Sauvegarde du résultat
output_name = "cwb_1km_nn.csv"
df_1km_final.to_csv(output_name, index=False)

print(f"🎉 Terminé ! Fichier '{output_name}' généré.")
print(f"📈 Nouveau nombre de mailles : {len(df_1km_final)} (Environ {len(df_1km_final)/len(df):.1f} fois plus de mailles).")