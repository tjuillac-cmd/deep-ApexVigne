import rioxarray
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

GPKG_PATH = r"C:\Users\juillact\Documents\STAGE_APEX_VIGNE\stage_apex_vigne\study_zone.gpkg"
zone = gpd.read_file(GPKG_PATH).to_crs("EPSG:4326")
bbox = zone.total_bounds

url = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/prec/2020/CHELSA_prec_01_01_2020_V.2.1.tif"

da = rioxarray.open_rasterio(url, masked=True, chunks={"x": 512, "y": 512})
da = da.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326")
da = da.compute()

print(f"CRS raster : {da.rio.crs}")
print(f"CRS zone   : {zone.crs}")
print(f"Bbox zone  : {bbox}")
print(f"Bbox raster: {da.rio.bounds()}")
print(f"Min/Max    : {float(np.nanmin(da.values)):.3f} / {float(np.nanmax(da.values)):.3f}")
print(f"% NaN      : {float(np.isnan(da.values).mean())*100:.1f}%")

# Visualisation rapide
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
da.squeeze().plot(ax=ax1)
zone.plot(ax=ax1, facecolor="none", edgecolor="red", linewidth=2)
ax1.set_title("Raster + zone (avant clip)")
zone.plot(ax=ax2)
ax2.set_title("Zone seule")
plt.tight_layout()
plt.savefig("test_zone.png", dpi=100)
print("\n→ Image sauvegardée : test_zone.png")