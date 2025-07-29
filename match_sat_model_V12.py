#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ce script est conçu pour l'analyse et la visualisation de données
météorologiques provenant de modèles (format RPN ou NetCDF) et de satellites
(format HDF5 CALIPSO). Il permet de :

1.  Charger les trajectoires de satellites (latitude, longitude, temps)
    ainsi que des variables spécifiques comme le contenu en eau de glace (IWC)
    et les hauteurs associées.
2.  Charger automatiquement les fichiers du modèle. Ceux-ci peuvent être :
    - des fichiers ``dp*`` contenant à la fois la variable principale et ``GZ``;
    - ou des paires de fichiers séparés où le nom de la variable apparaît dans le
      chemin (ex. ``..._TT_...`` et ``..._GZ_...``).
    Dans tous les cas, un NetCDF 4D est créé au besoin et réutilisé lors des
    exécutions suivantes.
3.  Filtrer les points de la trajectoire satellite qui se trouvent dans le
    domaine spatial du modèle.
4.  Générer plusieurs types de graphiques :
    * Une vue globale de la trajectoire satellite et des limites du modèle.
    * Un zoom sur la zone d'intersection (in-domain) avec les points appariés.
    * Des coupes verticales (profils) de la variable du modèle le long de la
        trajectoire satellite in-domain, avec l'altitude en kilomètres.
        Ces profils utiliseront, pour chaque granule, le fichier dp* dont t0
        est le plus proche du temps médian de survol in-domain.
"""
import sys, types

# cfgrib backend
_cfgrib = types.ModuleType("cfgrib")
_cfgrib.xarray_plugin = types.ModuleType("cfgrib.xarray_plugin")
sys.modules['cfgrib'] = _cfgrib
sys.modules['cfgrib.xarray_plugin'] = _cfgrib.xarray_plugin

# eccodes & gribapi
sys.modules['eccodes'] = types.ModuleType("eccodes")
sys.modules['gribapi'] = types.ModuleType("gribapi")
sys.modules['gribapi.bindings'] = types.ModuleType("gribapi.bindings")
sys.modules['gribapi.errors']   = types.ModuleType("gribapi.errors")
# ────────────────────────────────────────────────────────────
import os
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr
import h5py
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import pyproj
from typing import Tuple, List, Optional

# Workaround pour cartopy utilisant np.float (déprécié dans NumPy récent)
np.float = float

# --- Vérification de la disponibilité de rpnpy ---
try:
    import rpnpy.librmn.all as rmn
    from fstd2nc.mixins.dates import stamp2datetime
    HAS_RPN = True
    print("INFO: La bibliothèque 'rpnpy' est disponible pour la conversion RPN.")
except ImportError:
    print("ERREUR: 'rpnpy' ou 'fstd2nc' introuvable. Impossible de continuer.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONFIGURATION UTILISATEUR
# -----------------------------------------------------------------------------
SAT_DIR = os.path.join(os.environ['HOME'], 'Codes_travail', 'SAT_sim')
SAT_FILES = sorted(glob.glob(os.path.join(SAT_DIR, '*.h5')))
if not SAT_FILES:
    print(f"Erreur: Aucun .h5 dans {SAT_DIR}")
    sys.exit(1)

MODEL_DIR = os.path.join(os.environ['HOME'], 'Codes_travail', 'MOD_sim')

# Les fichiers du modèle seront détectés plus bas via _gather_model_file_pairs

VAR_MODEL_PRIMARY   = 'TT'
VAR_MODEL_GZ_NAME   = 'GZ'

SAT_FIELD_IWC         = '2C-ICE/Data Fields/IWC'
SAT_FIELD_HEIGHT      = '2C-ICE/Geolocation Fields/Height'
SAT_FIELD_IWC_FACTOR  = '2C-ICE/Swath Attributes/IWC.factor'
SAT_FIELD_IWC_MISSING = '2C-ICE/Swath Attributes/IWC.missing'

OUTPUT_IMAGE_BASE_NAME   = 'match_sat_model_results.png'
RES_FIG_DIR              = os.path.join(os.environ['HOME'], 'Codes_travail', 'RES_FIG')
MAIN_MAP_ENABLED         = True
ZOOM_ACT_MAIN_MAP        = True
VERTICAL_PROFILE_ENABLED = True
VERTICAL_PLOT_ALTITUDE_MAX_KM = 15
VERTICAL_PLOT_MODEL_NAME = 'GEM model'

mpl.rcParams['agg.path.chunksize'] = 10000

# -----------------------------------------------------------------------------
# FONCTIONS INTERNES
# -----------------------------------------------------------------------------
def _unwrap_structured_array(arr):
    return arr[arr.dtype.names[0]] if hasattr(arr.dtype, 'names') and arr.dtype.names else arr

def _gather_model_file_pairs(model_dir: str, main_var: str, gz_var: str) -> List[Tuple[str, str]]:
    """Détermine la liste des fichiers modèle à utiliser.

    Deux stratégies sont tentées :
    1.  Si des fichiers de type ``dp*`` (RPN) sont présents, on suppose que
        chaque fichier contient la variable principale **et** ``gz_var``.
    2.  Sinon, on cherche des paires de fichiers séparés contenant explicitement
        le nom des variables dans leur chemin (``*_<var>_*``).

    La fonction retourne une liste de tuples ``[(path_main, path_gz), ...]`` où
    ``path_main`` et ``path_gz`` peuvent être des fichiers RPN ou NetCDF déjà
    convertis.
    """

    # -- Cas 1 : fichiers dp* contenant toutes les variables
    base_rpn = sorted(f for f in glob.glob(os.path.join(model_dir, 'dp*')) if not f.endswith('.nc'))
    if base_rpn:
        return [(p, p) for p in base_rpn]

    # -- Cas 2 : variables séparées
    import re
    main_candidates = sorted(glob.glob(os.path.join(model_dir, f'*{main_var}*')))
    gz_candidates = sorted(glob.glob(os.path.join(model_dir, f'*{gz_var}*')))
    token_re = re.compile(r'(\d{8}\.\d{6}s)')
    pairs = []
    for m in main_candidates:
        token = token_re.search(os.path.basename(m))
        if not token:
            continue
        t = token.group(1)
        g_match = next((g for g in gz_candidates if t in g), None)
        if g_match:
            pairs.append((m, g_match))
    if pairs:
        return pairs

    raise FileNotFoundError(f"Aucun fichier modèle trouvé dans {model_dir}")

def _convert_rpn_to_netcdf_4d(rpn_path: str, varname: str) -> str:
    ncfile = rpn_path + f'_{varname}_4D.nc'
    if os.path.isfile(ncfile):
        return ncfile
    fid = rmn.fstopenall(rpn_path, rmn.FST_RO)
    keys = rmn.fstinl(fid, nomvar=varname)
    if not keys:
        rmn.fstcloseall(fid)
        raise RuntimeError(f"'{varname}' non trouvé dans {rpn_path}")
    records, times, levels = [], set(), set()
    for k in keys:
        rec = rmn.fstluk(k)
        if 'd' not in rec:
            continue
        t = stamp2datetime(rec['datev'])
        l = rec['ip1']
        times.add(t); levels.add(l)
        records.append((t, l, rec['d'], rec))
    sorted_times  = sorted(times)
    sorted_levels = sorted(levels)
    header0 = records[0][3]
    grid = rmn.readGrid(fid, header0)
    coords = rmn.gdll(grid)
    lat2d = coords['lat']
    lon2d = np.where(coords['lon']>180, coords['lon']-360, coords['lon'])
    ni, nj = header0['ni'], header0['nj']
    data4d = np.full((len(sorted_times), len(sorted_levels), nj, ni), np.nan, dtype=np.float32)
    for t, l, d, hdr in records:
        ti = sorted_times.index(t)
        li = sorted_levels.index(l)
        data4d[ti,li,:,:] = d
    rmn.fstcloseall(fid)
    ds = xr.Dataset(
        {varname:(('time','altitude_level','y','x'), data4d)},
        coords={
            'time': np.array(sorted_times, dtype='datetime64[ns]'),
            'altitude_level': sorted_levels,
            'lat': (('y','x'), lat2d),
            'lon': (('y','x'), lon2d),
        }
    )
    ds['altitude_level'].attrs.update(units='Pa', long_name='Pressure Level (Pa)')
    ds.to_netcdf(ncfile)
    return ncfile

def _load_model_variable_4d(path: str, varname: str) -> Tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """Ouvre une variable du modèle sous forme 4D (time, level, y, x).

    Le paramètre *path* peut être :
        - un fichier NetCDF déjà converti (xxx_4D.nc) contenant uniquement la variable,
        - un fichier RPN qui sera converti en NetCDF 4D si besoin,
        - la base "dp*" commune aux deux variables lorsque celles-ci sont dans le
          même fichier RPN.
    """

    if path.endswith('.nc'):
        ds = xr.open_dataset(path)
        return ds, ds['lat'].values, ds['lon'].values

    nc_existing = path + f'_{varname}_4D.nc'
    if os.path.isfile(nc_existing):
        ds = xr.open_dataset(nc_existing)
    else:
        ncfile = _convert_rpn_to_netcdf_4d(path, varname)
        ds = xr.open_dataset(ncfile)

    return ds, ds['lat'].values, ds['lon'].values

def _read_satellite_h5(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Charge les latitudes, longitudes et temps depuis un fichier satellite HDF5 (.h5).
    Cette fonction est adaptée pour les fichiers CALIPSO ou similaires,
    et tente également d'extraire l'IWC et les hauteurs.
    """
    print(f"  3.1) Lecture satellite HDF5 : {os.path.basename(path)}")
    iwc_data = None
    height_data = None

    with h5py.File(path, 'r') as h5f:
        def recurse(group, prefix=''):
            """
            Parcours récursif des groupes HDF5.
            - si c'est un Dataset, yield son chemin
            - si c'est un Group, récursivité
            - sinon (Datatype, Attribute, etc.) on ignore
            """
            for name in group.keys():
                obj = group[name]
                full_path = prefix + name
                if isinstance(obj, h5py.Dataset):
                    yield full_path
                elif isinstance(obj, h5py.Group):
                    yield from recurse(obj, full_path + '/')
                # tous les autres (Datatype, etc.) sont ignorés

        all_ds_paths = list(recurse(h5f))
        geo_paths = [d for d in all_ds_paths if 'geolocation' in d.lower()]

        lat_name = next((d for d in geo_paths if 'latitude' in d.lower()), None)
        lon_name = next((d for d in geo_paths if 'longitude' in d.lower()), None)
        profile_time_name = next((d for d in geo_paths if 'profile_time' in d.lower()), None)
        tai_start_name    = next((d for d in geo_paths if 'tai_start' in d.lower()), None)

        if not lat_name or not lon_name or not profile_time_name or not tai_start_name:
            raise ValueError(
                f"Variables de géolocalisation ou de temps non trouvées dans {os.path.basename(path)}.\n"
                f"Noms recherchés: lat='{lat_name}', lon='{lon_name}', "
                f"profile_time='{profile_time_name}', tai_start='{tai_start_name}'"
            )

        # Lecture des arrays bruts
        raw_lat = h5f[lat_name][...]
        raw_lon = h5f[lon_name][...]
        profile_time_raw = h5f[profile_time_name][...]
        tai_start_raw    = h5f[tai_start_name][...]

        # Déballage
        lat = _unwrap_structured_array(raw_lat).flatten()
        lon = _unwrap_structured_array(raw_lon).flatten()
        lon = ((lon + 180) % 360) - 180  # normalisation en [-180,180]

        unwrapped_tai_start = _unwrap_structured_array(tai_start_raw).flatten()[0]
        base_epoch = datetime(1993, 1, 1)
        granule_start = np.datetime64(base_epoch + timedelta(seconds=int(unwrapped_tai_start)), 'ns')

        unwrapped_profile_time = _unwrap_structured_array(profile_time_raw).flatten().astype(int)
        time_arr = granule_start + np.timedelta64(1, 's') * unwrapped_profile_time

        # Extraction IWC si présent
        if SAT_FIELD_IWC in h5f:
            raw_iwc = h5f[SAT_FIELD_IWC][...]
            iwc = _unwrap_structured_array(raw_iwc)
            # facteur
            fac = 1.0
            if SAT_FIELD_IWC_FACTOR in h5f.attrs:
                fac = h5f.attrs[SAT_FIELD_IWC_FACTOR]
                if isinstance(fac, np.ndarray): fac = fac.flatten()[0]
            if fac != 0:
                iwc = iwc / fac
            iwc_missing = None
            if SAT_FIELD_IWC_MISSING in h5f.attrs:
                iwc_missing = h5f.attrs[SAT_FIELD_IWC_MISSING]
                if isinstance(iwc_missing, np.ndarray): iwc_missing = iwc_missing.flatten()[0]
            if iwc_missing is not None:
                iwc = np.ma.masked_where(iwc == iwc_missing, iwc)
            iwc_data = iwc
            print(f"    IWC satellite extrait. Forme: {iwc.shape}")

        # Extraction hauteur si présent
        if SAT_FIELD_HEIGHT in h5f:
            raw_h = h5f[SAT_FIELD_HEIGHT][...]
            h = _unwrap_structured_array(raw_h).astype(np.float32)
            height_data = h
            print(f"    Hauteur satellite extraite. Forme: {h.shape}")

        return lat, lon, time_arr, iwc_data, height_data


def _filter_in_domain(lat_sat, lon_sat, time_sat, lat2d, lon2d, iwc=None, height=None):
    print(f"  4.1) Appariement de {len(lat_sat)} points...")
    pts_mod = np.vstack((lat2d.ravel(), lon2d.ravel())).T
    tree = cKDTree(pts_mod)
    pts_sat = np.vstack((lat_sat, lon_sat)).T
    d, idx = tree.query(pts_sat)
    dlat = abs(lat2d[1,0]-lat2d[0,0]) if lat2d.shape[0]>1 else 0.1
    dlon = abs(lon2d[0,1]-lon2d[0,0]) if lon2d.shape[1]>1 else 0.1
    thr = np.hypot(dlat/2, dlon/2)
    mask = d <= thr
    print(f"  4.1) {mask.sum()} points in-domain")
    yi, xi = np.unravel_index(idx[mask], lat2d.shape)
    iwc_in = iwc[mask] if (iwc is not None and len(iwc)==len(mask)) else None
    hgt_in = height[mask] if (height is not None and len(height)==len(mask)) else None
    return lat_sat[mask], lon_sat[mask], time_sat[mask], (yi,xi), mask, iwc_in, hgt_in

def _plot_main_map(
    traj_lat_segments, traj_lon_segments,
    in_domain_coords, matched_indices,
    grid_lat2d, grid_lon2d,
    output_basename
):
    """
    Trace vue globale & zoom polaire.
    """
    from matplotlib.patches import Polygon as MplPolygon

    # 1) Concaténation des trajectoires et in-domain
    all_lats = np.concatenate(traj_lat_segments)
    all_lons = np.concatenate(traj_lon_segments)
    in_lats  = np.concatenate([coords[0] for coords in in_domain_coords])
    in_lons  = np.concatenate([coords[1] for coords in in_domain_coords])

    # 2) Étendue complète du modèle
    lonmin, lonmax = grid_lon2d.min(), grid_lon2d.max()
    latmin, latmax = grid_lat2d.min(), grid_lat2d.max()

    # 3) Détection du zoom
    zoom_active = globals().get('ZOOM_ACT_MAIN_MAP', False)
    ncols = 2 if zoom_active else 1
    fig = plt.figure(figsize=(14, 6) if zoom_active else (7, 6))
    gs  = gridspec.GridSpec(1, ncols, wspace=0.3)

    # ----- Panneau 1 : Vue globale polaire -----
    ax0 = fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo())
    ax0.set_extent([lonmin, lonmax, latmin, latmax], ccrs.PlateCarree())
    ax0.coastlines('50m')
    ax0.add_feature(cfeature.LAND)
    ax0.gridlines(color='gray', linestyle=':')

    # a) Affichage de la grille modèle
    ax0.scatter(
        grid_lon2d.ravel(), grid_lat2d.ravel(),
        s=5, color='lightgray', alpha=0.7,
        transform=ccrs.PlateCarree(),
        label='Grille modèle'
    )

    # b) Trajectoires satellites
    palette = cm.get_cmap('tab10', len(traj_lat_segments))
    for i, (lats, lons) in enumerate(zip(traj_lat_segments, traj_lon_segments)):
        ax0.plot(
            lons, lats,
            linestyle='--', color=palette(i),
            transform=ccrs.Geodetic(),
            label=f'Granule {i+1}'
        )

    # c) Segments in-domain
    for i, (lats_i, lons_i, *_ ) in enumerate(in_domain_coords):
        ax0.plot(
            lons_i, lats_i,
            linestyle='-', color=palette(i),
            transform=ccrs.Geodetic(),
            label=f'In-domain {i+1}'
        )

    # d) Cadre vert sur le global pour le zoom
    if zoom_active:
        in_lats0, in_lons0 = in_domain_coords[0]
        yi0, xi0           = matched_indices[0]
        center = len(in_lats0) // 2
        cy, cx = yi0[center], xi0[center]

        half = 50
        y0, y1 = max(0, cy-half), min(grid_lat2d.shape[0], cy+half)
        x0, x1 = max(0, cx-half), min(grid_lon2d.shape[1], cx+half)

        zoom_lats = grid_lat2d[y0:y1, x0:x1]
        zoom_lons = grid_lon2d[y0:y1, x0:x1]

        bbox = [
            (zoom_lons.min(), zoom_lats.min()),
            (zoom_lons.min(), zoom_lats.max()),
            (zoom_lons.max(), zoom_lats.max()),
            (zoom_lons.max(), zoom_lats.min()),
        ]
        rect0 = MplPolygon(
            bbox, closed=True,
            edgecolor='green', facecolor='none', linewidth=2,
            transform=ccrs.PlateCarree(),
            label='Zone zoom (Granule 1)'
        )
        ax0.add_patch(rect0)

    ax0.set_title('Vue globale polaire')
    ax0.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    # ----- Panneau 2 : Zoom in-domain -----
    if zoom_active:
        ax1 = fig.add_subplot(gs[1], projection=ccrs.NorthPolarStereo())
        ax1.set_extent(
            [bbox[0][0], bbox[2][0], bbox[0][1], bbox[2][1]],
            ccrs.PlateCarree()
        )
        ax1.coastlines('50m')
        ax1.add_feature(cfeature.LAND)
        ax1.gridlines(color='gray', linestyle=':')

        # 1) Grille modèle masquée hors zoom
        ax1.scatter(
            grid_lon2d.ravel(), grid_lat2d.ravel(),
            s=5, color='lightgray', alpha=0.7,
            transform=ccrs.PlateCarree()
        )

        # 2) Trajectoire in-domain Granule 1
        ax1.plot(
            in_lons0, in_lats0,
            linestyle='-', color=palette(0),
            transform=ccrs.Geodetic()
        )

        # 3) Points appariés
        flat0 = yi0 * grid_lon2d.shape[1] + xi0
        ax1.scatter(
            grid_lon2d.ravel()[flat0],
            grid_lat2d.ravel()[flat0],
            s=30, marker='*', color='red',
            transform=ccrs.PlateCarree()
        )

        # 4) Nouveau polygone pour le panneau zoom
        rect1 = MplPolygon(
            bbox, closed=True,
            edgecolor='green', facecolor='none', linewidth=2,
            transform=ccrs.PlateCarree(),
            label='Zone zoom (Granule 1)'
        )
        ax1.add_patch(rect1)

        ax1.set_title('Zoom in-domain (Granule 1)')

    plt.suptitle('Colocalisation des trajectoires satellites avec le domaine du modèle')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outp = os.path.join(RES_FIG_DIR, output_basename.replace('.png', '_polar_zoom.png'))
    plt.savefig(outp, dpi=300)
    plt.close(fig)
    print(f"    → {os.path.basename(outp)}")


def _plot_model_vertical_curtain_dp(
    model_ds: xr.Dataset,
    lat_sat: np.ndarray,
    lon_sat: np.ndarray,
    matched_pts: Tuple[np.ndarray, np.ndarray],
    granule_idx: int,
    main_var: str,
    altitude_var: str,
    model_display_name: str,
    altitude_max_km: float,
    output_base_name: str
):
    """
    Tracé rapide (imshow) d'un rideau vertical du modèle pour main_var (IWCR ou TT).
    - model_ds : xarray.Dataset (un seul pas de temps) issu de .sel(time=..., method='nearest').
    - matched_pts = (yi, xi) sont les indices in-domain.
    """
    yi, xi = matched_pts

    # 1) Extraire la coupe levels×points (lazy → .values déclenche l'I/O minimal)
    da = model_ds[main_var].isel(y=yi, x=xi)  # <--- Passage par model_ds, pas sel_ds
    data = da.values.astype(float)
    data[data == 0.0] = np.nan

    # 2) Altitude par niveau (constante en y,x) → on prend y=0,x=0
    alt = model_ds[altitude_var].isel(y=0, x=0).values / 100.0  # dam → km

    # 3) Trier par altitude croissante
    order = np.argsort(alt)
    alt_sorted = alt[order]
    data_sorted = data[order, :]

    # 4) Choix du Norm et du colormap
    from matplotlib.colors import Normalize, LogNorm
    vals = data_sorted[np.isfinite(data_sorted)]

    if vals.size and np.nanmin(vals) > 0:
        # Cas positif (IWCR…)
        p5, p95 = np.nanpercentile(vals, [5, 95])
        vmin = max(p5, 1e-6)
        vmax = max(p95, vmin * 10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = 'jet'
    else:
        # Cas signé (TT…)
        vmin = np.nanmin(vals) if vals.size else None
        vmax = np.nanmax(vals) if vals.size else None
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = 'coolwarm'

    # 5) Tracé avec imshow (rapide)
    nlev, npts = data_sorted.shape
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        data_sorted.T,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        norm=norm,
        extent=[0, npts, alt_sorted[0], alt_sorted[-1]]
    )
    cb = fig.colorbar(im, ax=ax, extend='both')
    cb.set_label(f"{main_var} (unités)")

    # 6) Mise en forme des axes
    ax.set_ylim(0, altitude_max_km)
    ax.set_ylabel('Altitude (km)')

    nt = min(6, npts)
    ticks = np.linspace(0, npts - 1, nt, dtype=int)
    labels = [
        f"Lon: {lon_sat[i]:.2f}°\nLat: {lat_sat[i]:.2f}°"
        for i in ticks
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Points le long de la trajectoire in-domain')

    t0 = np.datetime_as_string(model_ds['time'].values[0], unit='s')
    ax.set_title(
        f"{model_display_name} – Profil vertical de {main_var} t0={t0} UTC "
        f"(Granule {granule_idx})"
    )

    plt.tight_layout()
    out_name = output_base_name.replace('.png', f'_modc_{granule_idx}.png')
    full_path = os.path.join(RES_FIG_DIR, out_name)
    fig.savefig(full_path, dpi=300)
    plt.close(fig)
    print(f"  → Sauvegardé : {os.path.basename(full_path)}")



def _plot_satellite_curtain(
    lat_sat_in, lon_sat_in, time_sat_in, iwc_sat_in, height_sat_in,
    granule_idx, output_base_name
):
    """
    Trace le rideau vertical de l'IWC satellite le long de la trajectoire in-domain.
    """
    print(f"  → Tracé rideau satellite (granule {granule_idx})...")
    n_pts, n_levels = iwc_sat_in.shape
    data = iwc_sat_in.T
    alt = height_sat_in[0,:] / 1000.0
    order = np.argsort(alt)
    alt = alt[order]; data = data[order,:]
    x = np.arange(n_pts)
    x_edges = np.concatenate(([x[0]-0.5], x+0.5))
    dz = np.diff(alt)
    z_edges = np.empty(len(alt)+1)
    z_edges[1:-1] = alt[:-1] + dz/2
    z_edges[0]      = alt[0] - dz[0]/2
    z_edges[-1]     = alt[-1] + dz[-1]/2
    positive = data[np.isfinite(data)&(data>0)]
    norm = mpl.colors.LogNorm(vmin=max(np.nanpercentile(positive,5),1e-6),
                              vmax=max(np.nanpercentile(positive,95),1e-6)) if positive.size else None
    fig, ax = plt.subplots(figsize=(10,4))
    pcm = ax.pcolormesh(x_edges, z_edges, data, shading='flat', cmap='viridis', norm=norm)
    cb = fig.colorbar(pcm, ax=ax, extend='both') if norm else fig.colorbar(pcm, ax=ax)
    cb.set_label('IWC satellite (g·m⁻³)')
    ax.set_ylim(0, VERTICAL_PLOT_ALTITUDE_MAX_KM)
    ax.set_ylabel('Altitude (km)')
    nt = min(6, n_pts)
    ticks = np.linspace(0, n_pts-1, nt, dtype=int)
    labels = [f"Lon: {lon_sat_in[i]:.2f}°\nLat: {lat_sat_in[i]:.2f}°" for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha='right')
    time_ref = np.datetime_as_string(time_sat_in[ticks[0]], unit='s')
    ax.set_title(f"CALIPSO – Rideau vertical d'IWC à {time_ref} UTC (Granule {granule_idx})")
    plt.tight_layout()
    outp = output_base_name.replace('.png', f'_satc_{granule_idx}.png')
    full_out = os.path.join(RES_FIG_DIR, outp)
    fig.savefig(full_out, dpi=300); plt.close(fig)
    print(f"  → Sauvegardé : {os.path.basename(full_out)}")

# -----------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    """
    Fonction principale réécrite pour :
    - Charger manuellement chaque pas de temps RPN→NetCDF via _load_model_variable_4d()
    - Sélectionner le Dataset dont t0 est le plus proche du temps médian in-domain
    - Contourner totalement open_mfdataset / Dask / fstd2nc.mixins.extern
    """
    start_time_script = time.time()
    print("==== DÉMARRAGE DU SCRIPT ====")

    # --- Étape 0 : nettoyage du répertoire de sortie ---
    if os.path.isdir(RES_FIG_DIR):
        for f in glob.glob(os.path.join(RES_FIG_DIR, '*.png')):
            os.remove(f)
    else:
        os.makedirs(RES_FIG_DIR)
    print(f"Étape 0 : vidé -> {RES_FIG_DIR}")

    # --- Étape 1 : chargement manuel de chaque pas de temps du modèle ---
    print("\nÉtape 1: Chargement manuel des pas de temps RPN → NetCDF 4D…")
    model_ds_list = []
    model_times   = []
    model_lat2d   = None
    model_lon2d   = None

    pairs = _gather_model_file_pairs(MODEL_DIR, VAR_MODEL_PRIMARY, VAR_MODEL_GZ_NAME)

    for main_path, gz_path in pairs:
        print(f"  → Fichiers : {os.path.basename(main_path)} / {os.path.basename(gz_path)}")
        ds_main, lat2d, lon2d = _load_model_variable_4d(main_path, VAR_MODEL_PRIMARY)
        ds_gz, _, _ = _load_model_variable_4d(gz_path, VAR_MODEL_GZ_NAME)

        ds_gz = ds_gz.assign_coords(lat=ds_main['lat'], lon=ds_main['lon'])
        ds = xr.merge([ds_main, ds_gz])

        t0 = ds['time'].values[0]
        model_ds_list.append(ds)
        model_times.append(t0)

        if model_lat2d is None:
            model_lat2d = lat2d
            model_lon2d = lon2d

    model_times = np.array(model_times, dtype='datetime64[ns]')
    print(f"  → {len(model_ds_list)} pas de temps chargés, t0 = {model_times}")

    # --- Préparation des listes pour la carte principale ---
    all_sat_lats = []
    all_sat_lons = []
    all_in_dom   = []
    all_mat_pts  = []

    # --- Étape 2 : traitement des granules satellite ---
    print("\nÉtape 2: Traitement des fichiers satellite…")
    for idx, sat_fp in enumerate(SAT_FILES, start=1):
        print(f"\n--- Granule {idx}/{len(SAT_FILES)} : {os.path.basename(sat_fp)} ---")
        try:
            lat_s, lon_s, time_s, iwc_s, hgt_s = _read_satellite_h5(sat_fp)
            lat_in, lon_in, time_in, pts, mask, iwc_in, hgt_in = _filter_in_domain(
                lat_s, lon_s, time_s,
                model_lat2d, model_lon2d,
                iwc=iwc_s, height=hgt_s
            )

            all_sat_lats.append(lat_s)
            all_sat_lons.append(lon_s)

            if lat_in.size == 0:
                print(f"  Aucun point in-domain pour le granule {idx}")
                continue

            all_in_dom.append((lat_in, lon_in))
            all_mat_pts.append(pts)

            # 2.a) temps médian in-domain
            sorted_t = np.sort(time_in)
            median_t = sorted_t[len(sorted_t)//2]

            # 2.b) choisir le pas de temps du modèle le plus proche
            idx_model = int(np.argmin(np.abs(model_times - median_t)))
            sel_ds    = model_ds_list[idx_model]
            sel_t0    = model_times[idx_model]

            print(
                f"  Temps médian sat : {np.datetime_as_string(median_t,'s')}, "
                f"→ modèle t0 sélectionné : {np.datetime_as_string(sel_t0,'s')}"
            )

            # 2.c) tracer les coupes verticales
            if VERTICAL_PROFILE_ENABLED:
                _plot_model_vertical_curtain_dp(
                    sel_ds,
                    lat_in, lon_in,
                    pts,
                    granule_idx=idx,
                    main_var=VAR_MODEL_PRIMARY,
                    altitude_var=VAR_MODEL_GZ_NAME,
                    model_display_name=VERTICAL_PLOT_MODEL_NAME,
                    altitude_max_km=VERTICAL_PLOT_ALTITUDE_MAX_KM,
                    output_base_name=OUTPUT_IMAGE_BASE_NAME
                )
                _plot_satellite_curtain(
                    lat_in, lon_in, time_in,
                    iwc_in, hgt_in,
                    idx, OUTPUT_IMAGE_BASE_NAME
                )

        except Exception as e:
            print(f"AVERTISSEMENT: impossible de traiter '{os.path.basename(sat_fp)}' : {e}")

    # --- Étape 3 : génération de la carte principale ---
    print("\nÉtape 3: Carte principale (globale & zoom)…")
    if MAIN_MAP_ENABLED and all_in_dom:
        _plot_main_map(
            all_sat_lats, all_sat_lons,
            all_in_dom, all_mat_pts,
            model_lat2d, model_lon2d,
            OUTPUT_IMAGE_BASE_NAME
        )

    print(f"\n✅ Script terminé en {time.time() - start_time_script:.1f} s")


if __name__ == '__main__':
    main()