import os
import processing  # <--- SÉPARÉ DE QGIS.CORE
from qgis.core import QgsProject, QgsVectorLayer, QgsCoordinateReferenceSystem
# --- CONFIGURATION (À MODIFIER AVEC VOS CHEMINS) ---
CHEMIN_CSV = "C:/Users/juillact/Documents/deep-ApexVigne/data/cov/cwb_cumsum_wide.csv"
DOSSIER_SORTIE = "C:/Users/juillact/Documents/deep-ApexVigne/data/cov/cwb/rasters_errones/"
NOM_COLONNE_X = "longitude"
NOM_COLONNE_Y = "latitude"

# Liste des dates (Dimanches des semaines 23 à 35 - Année 2025)
DATES_CIBLES = [
    "2025-06-08", "2025-06-15", "2025-06-22", "2025-06-29",
    "2025-07-06", "2025-07-13", "2025-07-20", "2025-07-27",
    "2025-08-03", "2025-08-10", "2025-08-17", "2025-08-24", "2025-08-31"
]

os.makedirs(DOSSIER_SORTIE, exist_ok=True)

# 1. Chargement du CSV en tant que couche de points (EPSG:4326)
uri = f"file:///{CHEMIN_CSV}?delimiter=,&xField={NOM_COLONNE_X}&yField={NOM_COLONNE_Y}&crs=epsg:4326"
couche_points = QgsVectorLayer(uri, "Points_CSV_8km", "delimitedtext")

if not couche_points.isValid():
    raise Exception("Impossible de charger le fichier CSV. Vérifiez le chemin et le délimiteur.")

print("1. CSV à 8 km chargé (WGS84).")

# 2. Reprojection en Lambert-93 (mètres)
params_reproject = {
    'INPUT': couche_points,
    'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:2154'),
    'OUTPUT': 'TEMPORARY_OUTPUT'
}
couche_lambert = processing.run("native:reprojectlayer", params_reproject)['OUTPUT']
print("2. Reprojection en Lambert-93 terminée.")

# 3. Reconstruction des mailles d'origine de 8 km (Tampon carré de 4000m de rayon)
# CORRECTION : END_CAP_STYLE à 2 pour le style Carré, JOIN_STYLE à 2 pour Miter
params_buffer = {
    'INPUT': couche_lambert,
    'DISTANCE': 4000,    # 4 km de rayon = 8 km de côté
    'SEGMENTS': 5,
    'END_CAP_STYLE': 2,  # 2 = Style Carré (Square)
    'JOIN_STYLE': 2,     # 2 = Angles vifs (Miter)
    'MITER_LIMIT': 2,
    'DISSOLVE': False,
    'OUTPUT': 'TEMPORARY_OUTPUT'
}
couche_mailles_8km = processing.run("native:buffer", params_buffer)['OUTPUT']
print("3. Reconstruction des mailles parentes de 8 km terminée.")

# Récupération de l'étendue globale
etendue = couche_mailles_8km.extent()

# 4. Boucle de rastérisation à la résolution cible de 1 km
print("4. Début de la génération des rasters à la résolution 1 km...")
for date in DATES_CIBLES:
    chemin_raster = os.path.join(DOSSIER_SORTIE, f"CWB_1km_{date}.tif")
    
    params_rasterize = {
        'INPUT': couche_mailles_8km,
        'FIELD': date,
        'BURN': 0,
        'UNITS': 1,            # Unités géoréférencées (mètres)
        'WIDTH': 1000,         # RÉSOLUTION CIBLE : 1000m (1km)
        'HEIGHT': 1000,        # RÉSOLUTION CIBLE : 1000m (1km)
        'EXTENT': etendue,
        'NODATA': -9999,
        'OPTIONS': '',
        'DATA_TYPE': 5,        # Float32
        'OUTPUT': chemin_raster
    }
    
    processing.run("gdal:rasterize", params_rasterize)
    print(f" -> Raster 1 km généré : CWB_1km_{date}.tif")

print("\nOpération réussie ! Vos 13 rasters à 1 km de résolution sont prêts.")