import rioxarray
import geopandas as gpd
import pandas as pd
import xarray as xr
from pathlib import Path

# ── 1. Zone géographique ───────────────────────────────────────────────────
GPKG_PATH = r"C:\Users\juillact\Documents\STAGE_APEX_VIGNE\stage_apex_vigne\study_zone.gpkg"
zone = gpd.read_file(GPKG_PATH).to_crs("EPSG:4326")
bbox = zone.total_bounds

# ── 2. Paramètres ─────────────────────────────────────────────────────────
BASE_URL   = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily"
VARIABLES  = ["prec", "tas", "tasmax", "tasmin"]
START, END = "2019-01-01", "2025-08-01"
OUTPUT_DIR = Path(r"C:\Users\juillact\Documents\STAGE_APEX_VIGNE\stage_apex_vigne\chelsa_output")
OUTPUT_DIR.mkdir(exist_ok=True)

dates = pd.date_range(start=START, end=END, freq="D")

# ── 3. Fonction de lecture ─────────────────────────────────────────────────
def fetch_and_clip(variable, date, zone, bbox):
    dd, mm, yyyy = date.strftime("%d"), date.strftime("%m"), date.strftime("%Y")
    url = f"{BASE_URL}/{variable}/{yyyy}/CHELSA_{variable}_{dd}_{mm}_{yyyy}_V.2.1.tif"

    da = rioxarray.open_rasterio(url, masked=True, chunks={"x": 512, "y": 512})

    # Découpe bbox
    da = da.rio.clip_box(
        minx=bbox[0], miny=bbox[1],
        maxx=bbox[2], maxy=bbox[3],
        crs="EPSG:4326"
    )
    da = da.compute()

    # Découpe polygone — nodata=-9999 pour éviter les NaN parasites
    da = da.rio.clip(
        zone.geometry,
        zone.crs,
        drop=True,
        all_touched=True,    # inclut les pixels qui touchent le bord
        from_disk=False
    )

    da = da.squeeze("band", drop=True)
    da = da.expand_dims(time=[pd.Timestamp(date)])
    return da

# ── 4. Boucle par variable ─────────────────────────────────────────────────
for var in VARIABLES:
    print(f"\n{'='*50}\n  Variable : {var}\n{'='*50}")
    out_path = OUTPUT_DIR / f"chelsa_{var}_2019_2025.nc"

    if out_path.exists():
        print("  → Déjà traité, on passe.")
        continue

    results, errors = [], []

    for i, date in enumerate(dates):
        try:
            da = fetch_and_clip(var, date, zone, bbox)
            results.append(da)
            pct = (i + 1) / len(dates) * 100
            print(f"  ✓ {date.date()} ({pct:.1f}%)", end="\r")
        except Exception as e:
            errors.append((date.date(), str(e)))
            print(f"\n  ✗ {date.date()} — {e}")

        # Sauvegarde intermédiaire tous les 365 jours
        if len(results) > 0 and len(results) % 365 == 0:
            yr = date.year
            tmp = OUTPUT_DIR / f"chelsa_{var}_tmp_{yr}.nc"
            xr.concat(results, dim="time").to_netcdf(tmp)
            print(f"\n  → Intermédiaire : {tmp}")
            results = []

    # Fusion finale
    tmp_files = sorted(OUTPUT_DIR.glob(f"chelsa_{var}_tmp_*.nc"))
    all_parts = [xr.open_dataset(f) for f in tmp_files]
    if results:
        all_parts.append(xr.concat(results, dim="time"))

    if all_parts:
        ds = xr.concat(all_parts, dim="time") if len(all_parts) > 1 else all_parts[0]

        # Conversion K → °C
        if var in ["tas", "tasmax", "tasmin"]:
            ds = ds - 273.15
            units = "°C"
        else:
            units = "mm day-1"

        ds.attrs["units"]  = units
        ds.attrs["source"] = "CHELSA-daily V2.1"
        ds.to_netcdf(out_path)
        print(f"\n  ✅ Sauvegardé : {out_path}")

        for f in tmp_files:
            f.unlink()

    if errors:
        err_path = OUTPUT_DIR / f"errors_{var}.txt"
        with open(err_path, "w") as f:
            for d, msg in errors:
                f.write(f"{d}\t{msg}\n")
        print(f"  → {len(errors)} erreurs dans {err_path}")

print("\n✅ Terminé !")