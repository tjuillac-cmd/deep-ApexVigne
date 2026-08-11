import os
from qgis.core import QgsProject, QgsRasterLayer
from osgeo import gdal

# --- CONFIGURATION DES CHEMINS ---
DOSSIER_GLOBAUX = "C:/Users/juillact/Documents/deep-ApexVigne/data/cov/cwb/rasters_errones/"       
DOSSIER_DROME = "C:/Users/juillact/Documents/deep-ApexVigne/data/cov/cwb/correction_drome/"           
DOSSIER_CORRIGE = "C:/Users/juillact/Documents/deep-ApexVigne/data/rasters_corriges/"      

DATES_CIBLES = [
    "2025-06-08", "2025-06-15", "2025-06-22", "2025-06-29",
    "2025-07-06", "2025-07-13", "2025-07-20", "2025-07-27",
    "2025-08-03", "2025-08-10", "2025-08-17", "2025-08-24", "2025-08-31"
]

os.makedirs(DOSSIER_CORRIGE, exist_ok=True)

print("Début du remplacement matriciel direct (Méthode NumPy)...")

for date in DATES_CIBLES:
    fichier_global = os.path.join(DOSSIER_GLOBAUX, f"CWB_1km_{date}.tif").replace('\\', '/')
    fichier_drome = os.path.join(DOSSIER_DROME, f"CWB_1km_{date}.tif").replace('\\', '/')
    fichier_sortie = os.path.join(DOSSIER_CORRIGE, f"CWB_corrigé_1km_{date}.tif").replace('\\', '/')
    
    if not os.path.exists(fichier_global) or not os.path.exists(fichier_drome):
        print(f" -> [Ignoré] Fichiers manquants pour la date {date}")
        continue
        
    print(f" -> Correction matricielle pour la date {date}...")
    
    try:
        # 1. Ouvrir le raster global (Modèle d'emprise)
        ds_global = gdal.Open(fichier_global)
        band_global = ds_global.GetRasterBand(1)
        arr_global = band_global.ReadAsArray()
        nodata_global = band_global.GetNoDataValue()
        if nodata_global is None: nodata_global = -9999

        # 2. Ouvrir le raster de la Drôme
        ds_drome = gdal.Open(fichier_drome)
        band_drome = ds_drome.GetRasterBand(1)
        arr_drome = band_drome.ReadAsArray()
        nodata_drome = band_drome.GetNoDataValue()
        if nodata_drome is None: nodata_drome = -9999

        # 3. Récupérer les informations géographiques pour caler la Drôme sur le Global
        geo_transform_glob = ds_global.GetGeoTransform()
        geo_transform_drome = ds_drome.GetGeoTransform()
        
        # Calcul des décalages en pixels de la Drôme par rapport au Global
        # X_pixel = (X_coordonnée - X_origine) / Taille_Pixel
        offset_x = int(round((geo_transform_drome[0] - geo_transform_glob[0]) / geo_transform_glob[1]))
        offset_y = int(round((geo_transform_drome[3] - geo_transform_glob[3]) / geo_transform_glob[5]))
        
        # Dimensions de la matrice de la Drôme
        lignes_drome, cols_drome = arr_drome.shape
        
        # 4. Créer la copie de sortie basée sur la matrice globale
        arr_sortie = arr_global.copy()
        
        # 5. Remplacement strict en mémoire (uniquement là où la Drôme chevauche le global)
        # On définit la zone d'impact dans le repère du raster global
        g_start_y = max(0, offset_y)
        g_end_y = min(arr_global.shape[0], offset_y + lignes_drome)
        g_start_x = max(0, offset_x)
        g_end_x = min(arr_global.shape[1], offset_x + cols_drome)
        
        # On définit la zone correspondante dans le repère du raster Drôme
        d_start_y = max(0, -offset_y)
        d_end_y = d_start_y + (g_end_y - g_start_y)
        d_start_x = max(0, -offset_x)
        d_end_x = d_start_x + (g_end_x - g_start_x)
        
        # Extraction des sous-matrices pour travailler sur la zone de recouvrement
        sub_global = arr_sortie[g_start_y:g_end_y, g_start_x:g_end_x]
        sub_drome = arr_drome[d_start_y:d_end_y, d_start_x:d_end_x]
        
        # Masque booléen : où la Drôme a de la donnée (différent de son NoData)
        masque_correction = (sub_drome != nodata_drome)
        
        # Application de la correction sur la sous-partie
        sub_global[masque_correction] = sub_drome[masque_correction]
        
        # Redéploiement de la sous-partie modifiée dans la matrice finale
        arr_sortie[g_start_y:g_end_y, g_start_x:g_end_x] = sub_global

        # 6. Écriture du fichier GeoTIFF final sur le disque
        driver = gdal.GetDriverByName('GTiff')
        ds_sortie = driver.Create(fichier_sortie, ds_global.RasterXSize, ds_global.RasterYSize, 1, gdal.GDT_Float32, options=['COMPRESS=LZW'])
        ds_sortie.SetGeoTransform(geo_transform_glob)
        ds_sortie.SetProjection(ds_global.GetProjection())
        
        band_sortie = ds_sortie.GetRasterBand(1)
        band_sortie.WriteArray(arr_sortie)
        band_sortie.SetNoDataValue(nodata_global)
        
        # Fermeture des fichiers pour forcer l'écriture physique sur le disque
        ds_sortie = None
        ds_global = None
        ds_drome = None
        
        # 7. Chargement de la couche dans QGIS
        if os.path.exists(fichier_sortie):
            nom_couche = f"CWB_corrigé_1km_{date}"
            couche_raster = QgsRasterLayer(fichier_sortie, nom_couche)
            if couche_raster.isValid():
                QgsProject.instance().addMapLayer(couche_raster)
                print(f"    [OK] Fichier sauvegardé et chargé : {nom_couche}.tif")
                
    except Exception as e:
        print(f"    [Erreur] Échec du traitement pour la date {date} : {str(e)}")

print("\nTraitement terminé. Toutes les matrices ont été corrigées avec précision.")