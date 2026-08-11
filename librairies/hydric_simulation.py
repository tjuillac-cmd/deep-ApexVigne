"""
Utility functions for spatio-temporal hydric stress simulation (ApexVigne).
Adapted from D:/IA/simulation/librairies/ for a single continuous response variable.

Functions
---------
load_and_align_rasters      : Load alt/RU/CWB rasters, reproject all to alt grid
build_study_mask            : Boolean mask of valid (inside study zone) pixels
build_static_ground_truth   : Multivariate Gaussian mixture → static ic_apex map
build_temporal_ground_truth : AR(1) + CWB weekly modulation → weekly ic_apex maps
generate_virtual_parcelles  : Fixed observer positions within study zone
temporal_observation_weights: Temporal intensity curve (unimodal, bimodal, etc.)
sample_apex_counts          : (apex0, apex1, apex2) from true ic_apex via Dirichlet-Multinomial
sample_n_total              : Total apex count per observation (Negative Binomial)
get_value_at_coord          : Extract raster value at Lambert (x, y) coordinate
build_apex_dataframe        : Build DataFrame compatible with data_apex.csv schema
save_ground_truth_raster    : Export 2D array as GeoTIFF
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
import glob
import os
import uuid
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raster loading & alignment
# ─────────────────────────────────────────────────────────────────────────────

def load_and_align_rasters(alt_path, ru_path, cwb_dir, study_zone_gdf):
    """
    Load altitude, RU and all weekly CWB rasters.
    RU and CWB are reprojected/resampled to the altitude raster grid (reference).

    Returns
    -------
    alt_arr      : (H, W) float32 — altitude in meters, NaN outside study zone
    ru_arr       : (H, W) float32 — RU class codes (1,2,3,4,5,9), aligned to alt grid
    cwb_aligned  : dict[date_str → (H,W) float32] — weekly CWB values
    ref_transform: rasterio Affine transform of the reference (altitude) grid
    ref_crs      : CRS of the reference grid
    ref_shape    : (H, W) tuple
    """
    geoms = [study_zone_gdf.union_all()]

    # Reference grid: altitude raster
    with rasterio.open(alt_path) as src:
        ref_transform = src.transform
        ref_crs       = src.crs
        ref_shape     = (src.height, src.width)
        out, _        = rio_mask(src, geoms, crop=False, nodata=np.nan)
        alt_arr       = out[0].astype(np.float32)
        nodata_alt    = src.nodata
    if nodata_alt is not None:
        alt_arr[alt_arr == nodata_alt] = np.nan

    # RU raster: nearest-neighbor reproject to alt grid
    with rasterio.open(ru_path) as src:
        ru_reproj = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source        = rasterio.band(src, 1),
            destination   = ru_reproj,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = ref_transform,
            dst_crs       = ref_crs,
            resampling    = Resampling.nearest,
        )
    ru_arr = ru_reproj

    # Weekly CWB rasters: bilinear reproject to alt grid
    cwb_files   = sorted(glob.glob(os.path.join(cwb_dir, 'CWB_1km_*.tif')))
    cwb_aligned = {}
    for fpath in cwb_files:
        date_str = os.path.basename(fpath).replace('CWB_1km_', '').replace('.tif', '')
        with rasterio.open(fpath) as src:
            cwb_reproj = np.empty(ref_shape, dtype=np.float32)
            reproject(
                source        = rasterio.band(src, 1),
                destination   = cwb_reproj,
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = ref_transform,
                dst_crs       = ref_crs,
                resampling    = Resampling.bilinear,
            )
            nodata = src.nodata
        if nodata is not None:
            cwb_reproj[cwb_reproj == nodata] = np.nan
        cwb_aligned[date_str] = cwb_reproj

    return alt_arr, ru_arr, cwb_aligned, ref_transform, ref_crs, ref_shape


def build_study_mask(alt_arr):
    """Boolean mask: True where altitude is finite (i.e. inside study zone)."""
    return np.isfinite(alt_arr)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multivariate Gaussian Ground Truth  (adapted from virtual_species.py)
# ─────────────────────────────────────────────────────────────────────────────

def _mvgauss_vectorized(X, mu, cov_inv, cov_det, n_feat):
    """Vectorized multivariate Gaussian PDF on array X of shape (N, n_feat)."""
    diff     = X - mu
    exponent = -0.5 * np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
    coeff    = 1.0 / (np.sqrt((2 * np.pi) ** n_feat * cov_det) + 1e-300)
    return coeff * np.exp(exponent)


def build_static_ground_truth(alt_arr, ru_arr, study_mask,
                               ru_class_to_mm=None, smooth_sigma=2.5):
    """
    Build a static hydric-state map ∈ [0, 1] using a Gaussian mixture over
    (altitude_norm, RU_continuous_norm).

    Physical encoding
    -----------------
    - High altitude + high RU  →  high ic_apex  (no water stress)
    - Low altitude  + low RU   →  low  ic_apex  (water stress)

    Returns
    -------
    static_map : (H, W) float32, NaN outside study zone
    alt_norm   : (H, W) normalized altitude ∈ [0,1]
    ru_norm    : (H, W) normalized RU proxy ∈ [0,1]
    """
    if ru_class_to_mm is None:
        # Class midpoints in mm (reserves utiles)
        ru_class_to_mm = {1: 25, 2: 75, 3: 125, 4: 175, 5: 225, 9: 0}

    # RU classes → continuous mm proxy
    ru_cont = np.full_like(ru_arr, np.nan, dtype=np.float32)
    for cls, mm in ru_class_to_mm.items():
        ru_cont[ru_arr == cls] = float(mm)

    # Normalize to [0, 1] over study zone pixels
    def _norm01(arr, mask):
        v = arr[mask & np.isfinite(arr)]
        return (arr - v.min()) / (v.max() - v.min() + 1e-8)

    alt_norm = _norm01(alt_arr, study_mask)
    ru_norm  = _norm01(ru_cont,  study_mask)

    nrows, ncols = alt_arr.shape
    valid = study_mask & np.isfinite(alt_norm) & np.isfinite(ru_norm)
    X = np.column_stack([alt_norm[valid], ru_norm[valid]])  # (N_valid, 2)

    # Gaussian mixture:
    #   +1  → pushes ic_apex up   (no stress)
    #   -1  → pushes ic_apex down (stress)
    components = [
        # No-stress zone: high RU + moderate altitude
        dict(mu=[0.45, 0.85], cov=[[0.08, 0.02], [0.02, 0.06]], weight=1.0, sign=+1),
        # Stress zone: low RU + low altitude
        dict(mu=[0.20, 0.15], cov=[[0.05, 0.01], [0.01, 0.04]], weight=0.9, sign=-1),
        # No-stress zone: high altitude (cooling effect)
        dict(mu=[0.80, 0.50], cov=[[0.06, 0.00], [0.00, 0.10]], weight=0.7, sign=+1),
    ]

    lam = np.zeros(len(X))
    for comp in components:
        mu      = np.array(comp['mu'])
        cov     = np.array(comp['cov'])
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        density = _mvgauss_vectorized(X, mu, cov_inv, cov_det, 2)
        d_norm  = (density - density.min()) / (density.max() - density.min() + 1e-10)
        lam    += comp['weight'] * comp['sign'] * d_norm

    lam = (lam - lam.min()) / (lam.max() - lam.min() + 1e-8)

    # Place values back into 2D grid
    flat = np.full(nrows * ncols, np.nan, dtype=np.float32)
    flat[valid.ravel()] = lam.astype(np.float32)
    static_map = flat.reshape(nrows, ncols)

    # Spatially smooth (edge-aware: weights by finite-pixel density)
    if smooth_sigma > 0:
        finite   = np.isfinite(static_map)
        filled   = np.where(finite, static_map, 0.0)
        w_smooth = gaussian_filter(finite.astype(float), sigma=smooth_sigma)
        s_smooth = gaussian_filter(filled,                sigma=smooth_sigma)
        with np.errstate(invalid='ignore'):
            static_map = np.where(w_smooth > 1e-3, s_smooth / w_smooth, np.nan)
        # Re-normalize after smoothing
        v          = static_map[study_mask & np.isfinite(static_map)]
        static_map = (static_map - v.min()) / (v.max() - v.min() + 1e-8)
        static_map = np.clip(static_map, 0, 1).astype(np.float32)
        static_map[~study_mask] = np.nan

    return static_map, alt_norm.astype(np.float32), ru_norm.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Temporal ground truth with AR(1) noise + CWB modulation
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_ground_truth(static_map, cwb_aligned, study_mask,
                                  ar1_rho=0.8, cwb_weight=0.35,
                                  noise_sigma=0.04, spatial_smooth=1.5, seed=42):
    """
    Build weekly hydric-state maps by combining:
      - The static map (altitude + RU structure)
      - Weekly CWB raster (drives seasonal dynamics)
      - AR(1) spatially correlated noise (temporal autocorrelation)

    Model (per pixel x, per week t)
    --------------------------------
    η(x, t) = ρ · η(x, t-1)  +  √(1−ρ²) · σ · ε̃(x, t)
    ε̃(x, t) = spatially smoothed standard normal noise

    h(x, t) = clip((1−β)·static(x)  +  β·cwb_norm(x, t)  +  η(x, t),  0, 1)

    Parameters
    ----------
    ar1_rho      : temporal autocorrelation ρ ∈ [0, 1)
    cwb_weight   : β, weight of CWB (0 = fully static, 1 = fully CWB-driven)
    noise_sigma  : amplitude of the AR(1) noise process
    spatial_smooth: σ of Gaussian filter applied to noise (km, given 1km pixels)

    Returns
    -------
    temporal_maps : dict[date_str → (H, W) float32]  ic_apex ∈ [0,1], NaN outside
    sorted_dates  : list of date strings in chronological order
    """
    rng         = np.random.default_rng(seed)
    nrows, ncols = static_map.shape
    sorted_dates = sorted(cwb_aligned.keys())

    # Normalize CWB stack to [0, 1]:  high CWB (wet) = high ic_apex (no stress)
    cwb_stack = np.stack([cwb_aligned[d] for d in sorted_dates], axis=0)
    valid_cwb = cwb_stack[:, study_mask]
    cwb_p5    = float(np.nanpercentile(valid_cwb, 5))
    cwb_p95   = float(np.nanpercentile(valid_cwb, 95))
    cwb_norm  = np.clip((cwb_stack - cwb_p5) / (cwb_p95 - cwb_p5 + 1e-8), 0, 1)
    for t in range(len(sorted_dates)):
        cwb_norm[t][~study_mask] = np.nan

    # AR(1) noise: stationary variance = noise_sigma² / (1 - ρ²)
    eta       = np.zeros((nrows, ncols), dtype=np.float32)
    innov_std = np.sqrt(max(1.0 - ar1_rho**2, 0.0))
    temporal_maps = {}

    for t, date_str in enumerate(sorted_dates):
        eps_raw    = rng.standard_normal((nrows, ncols)).astype(np.float32)
        eps_smooth = gaussian_filter(eps_raw, sigma=spatial_smooth)
        eta        = ar1_rho * eta + innov_std * noise_sigma * eps_smooth

        h_t = (1.0 - cwb_weight) * static_map + cwb_weight * cwb_norm[t] + eta
        h_t = np.clip(h_t, 0.0, 1.0).astype(np.float32)
        h_t[~study_mask] = np.nan

        temporal_maps[date_str] = h_t

    return temporal_maps, sorted_dates


# ─────────────────────────────────────────────────────────────────────────────
# 4. Virtual parcelles (fixed spatial positions)
# ─────────────────────────────────────────────────────────────────────────────

def generate_virtual_parcelles(study_zone_gdf, n_parcelles=300, seed=42, buffer_m=-300):
    """
    Sample n_parcelles points uniformly at random inside the study zone.
    An inward buffer avoids boundary artefacts.

    Returns a GeoDataFrame in the study-zone CRS (EPSG:2154) with columns:
    id_parcelle, id_observateur, geometry.
    """
    rng  = np.random.default_rng(seed)
    zone = study_zone_gdf.union_all()
    inner = zone.buffer(buffer_m) if buffer_m != 0 else zone
    if inner.is_empty:
        inner = zone

    minx, miny, maxx, maxy = inner.bounds
    pts, parcel_ids, obs_ids = [], [], []

    while len(pts) < n_parcelles:
        xs = rng.uniform(minx, maxx, n_parcelles * 6)
        ys = rng.uniform(miny, maxy, n_parcelles * 6)
        for x, y in zip(xs, ys):
            if inner.contains(Point(x, y)):
                pts.append(Point(x, y))
                parcel_ids.append(str(uuid.uuid4())[:12])
                obs_ids.append(str(uuid.uuid4())[:12])
            if len(pts) >= n_parcelles:
                break

    return gpd.GeoDataFrame(
        {'id_parcelle': parcel_ids[:n_parcelles],
         'id_observateur': obs_ids[:n_parcelles]},
        geometry=pts[:n_parcelles],
        crs=study_zone_gdf.crs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Temporal observation intensity
# ─────────────────────────────────────────────────────────────────────────────

def temporal_observation_weights(n_weeks, pattern='unimodal', seed=42):
    """
    Return normalized observation-intensity weights over n_weeks (sum = 1).

    Patterns
    --------
    'unimodal'  : Gaussian peak at mid-season (peak around week 7/13 ≈ late July)
    'bimodal'   : Early-season burst + late-season burst
    'decreasing': Exponential decay (more observations early)
    'random'    : Stochastic weights
    'uniform'   : Flat (no temporal bias)
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 1, n_weeks)

    if pattern == 'unimodal':
        w = np.exp(-((t - 0.55) ** 2) / (2 * 0.18 ** 2))
    elif pattern == 'bimodal':
        w = (np.exp(-((t - 0.25) ** 2) / (2 * 0.08 ** 2)) +
             0.8 * np.exp(-((t - 0.75) ** 2) / (2 * 0.10 ** 2)))
    elif pattern == 'decreasing':
        w = np.exp(-3 * t) + 0.1
    elif pattern == 'random':
        w = rng.uniform(0.05, 1.0, n_weeks)
    else:  # uniform
        w = np.ones(n_weeks)

    w = np.maximum(w, 1e-6)
    return w / w.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Apex count generation from true ic_apex
# ─────────────────────────────────────────────────────────────────────────────

def sample_apex_counts(ic_apex_true, n_total, dirichlet_conc=6.0, rng=None):
    """
    Generate (apex0, apex1, apex2) from a true ic_apex value p.

    Model
    -----
    Expected probabilities:  μ = (p², 2p(1−p), (1−p)²)
    Observer noise:          probs ~ Dirichlet(conc · μ)
    Count draw:              (apex0, apex1, apex2) ~ Multinomial(n_total, probs)

    Unbiasedness check
    ------------------
    E[ic_apex_obs] = E[(apex0 + 0.5·apex1) / n_total]
                   = p² + p(1−p) = p  ✓

    Parameters
    ----------
    ic_apex_true    : true hydric state ∈ [0, 1]
    n_total         : total apex count assessed
    dirichlet_conc  : concentration parameter (higher = less observer noise)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    p  = float(np.clip(ic_apex_true, 0.01, 0.99))
    mu = np.array([p ** 2, 2 * p * (1 - p), (1 - p) ** 2], dtype=np.float64)
    mu = np.maximum(mu, 1e-6)
    mu /= mu.sum()
    probs  = rng.dirichlet(dirichlet_conc * mu)
    counts = rng.multinomial(int(n_total), probs)
    return int(counts[0]), int(counts[1]), int(counts[2])


def sample_n_total(mean_n=50, dispersion=0.4, rng=None):
    """
    Total apex count per observation drawn from a Negative Binomial
    to mimic real variability in assessments (some observers count more apices).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    var  = mean_n * (1.0 + dispersion)
    p_nb = mean_n / var
    r_nb = mean_n * p_nb / (1.0 - p_nb + 1e-8)
    n    = int(rng.negative_binomial(max(int(r_nb), 1),
                                      float(np.clip(p_nb, 0.001, 0.999))))
    return max(n, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Raster value extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_value_at_coord(x, y, arr, transform):
    """
    Extract scalar value from 2D array `arr` at Lambert-93 (x, y).
    Returns np.nan if out of bounds or NaN cell.
    """
    try:
        r, c = rowcol(transform, x, y)
        nrows, ncols = arr.shape
        r, c = int(r), int(c)
        if 0 <= r < nrows and 0 <= c < ncols:
            v = float(arr[r, c])
            return v if np.isfinite(v) else np.nan
        return np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 8. DataFrame builder (data_apex.csv schema)
# ─────────────────────────────────────────────────────────────────────────────

def build_apex_dataframe(observations):
    """
    Build a DataFrame in data_apex.csv schema from a list of observation dicts.

    Required keys per dict
    ----------------------
    date_session, longitude_wgs84, latitude_wgs84,
    apex0, apex1, apex2, id_parcelle, id_observateur

    Optional
    --------
    ic_apex_true  (ground-truth value, for validation only)
    """
    rows = []
    for obs in observations:
        n = obs['apex0'] + obs['apex1'] + obs['apex2']
        ic_obs = (obs['apex0'] + 0.5 * obs['apex1']) / max(n, 1)
        rows.append({
            'id_session'    : str(uuid.uuid4()),
            'date_creation' : str(obs['date_session']),
            'date_maj'      : str(obs['date_session']),
            'date_session'  : str(obs['date_session']),
            'apex0'         : obs['apex0'],
            'apex1'         : obs['apex1'],
            'apex2'         : obs['apex2'],
            'id_observateur': obs.get('id_observateur', str(uuid.uuid4())[:12]),
            'id_parcelle'   : obs['id_parcelle'],
            'id_stade'      : None,
            'commentaire'   : None,
            'device_hardware': 'simulated',
            'device_software': 'simulation_v1.0',
            'longitude'     : obs['longitude_wgs84'],
            'latitude'      : obs['latitude_wgs84'],
            'location'      : None,
            'en_parcelle'   : 1,
            'est_archivee'  : 0,
            'raison_archive': None,
            'app_version'   : 'sim_1.0',
            # Extra columns for validation (absent in real data)
            'ic_apex_true'  : obs.get('ic_apex_true', np.nan),
            'ic_apex_obs'   : round(ic_obs, 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Raster export
# ─────────────────────────────────────────────────────────────────────────────

def save_ground_truth_raster(arr, transform, crs, filepath):
    """Save a 2D float32 array as a single-band GeoTIFF (NaN = nodata)."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    nrows, ncols = arr.shape
    with rasterio.open(
        filepath, 'w',
        driver='GTiff', height=nrows, width=ncols,
        count=1, dtype='float32',
        crs=crs, transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)
