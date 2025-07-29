#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ce script est conçu pour l'analyse et la visualisation de données


météorologiques provenant de modèles (format RPN ou NetCDF) et de satellites
(format HDF5 CALIPSO). Il permet de :

1.  Charger les trajectoires de satellites (latitude, longitude, temps)
    ainsi que des variables spécifiques comme le contenu en eau de glace (IWC)
    et les hauteurs associées.
2.  Charger ou convertir les données du modèle (une variable principale comme
    IWCR et la hauteur géopotentielle GZ) en un format NetCDF 4D (temps, niveau, y, x).
3.  Filtrer les points de la trajectoire satellite qui se trouvent dans le
    domaine spatial du modèle.
4.  Générer plusieurs types de graphiques :
    * Une vue globale de la trajectoire satellite et des limites du modèle.
    * Un zoom sur la zone d'intersection (in-domain) avec les points appariés.
    * Des cartes polaires cumulées (optionnel).
    * Des coupes verticales (profils) de la variable du modèle le long de la
        trajectoire satellite in-domain, avec l'altitude en kilomètres (optionnel).
        Ces profils incluront également les données satellite correspondantes pour comparaison.

Niveau de programmeur cible : Intermédiaire.
Des commentaires détaillés sont fournis pour faciliter la compréhension et la modification.

Prérequis :
Assurez-vous que les bibliothèques Python suivantes sont installées :
-   numpy
-   xarray
-   h5py
-   scipy (pour cKDTree et interpolation)
-   matplotlib
-   cartopy
-   pandas
-   rpnpy (pour la conversion des fichiers RPN)
-   fstd2nc (pour la conversion des dates RPN)

Pour les environnements de calcul (HPC), vous pourriez avoir besoin de charger des modules
avant d'exécuter ce script, par exemple :
$ module load python3/miniconda3 python3/python-rpn python3/outils-divers
Puis activez votre environnement Conda :
$ conda activate base_plus
"""

import os
import sys
import glob
import time
import numpy as np
import xarray as xr
import h5py
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d # Pour l'interpolation verticale des données satellite
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import defaultdict
from datetime import datetime, timedelta 
import pandas as pd 
import pyproj
import pdb

# --- Correction pour la compatibilité Python < 3.9 pour les annotations de type (à retirer au besoin) ---
from typing import Tuple, List, Optional

# Workaround pour cartopy utilisant np.float (déprécié dans les versions récentes de NumPy)
np.float = float

# --- Vérification de la disponibilité de rpnpy ---
try:
    import rpnpy.librmn.all as rmn
    from fstd2nc.mixins.dates import stamp2datetime
    HAS_RPN = True
    print("INFO: La bibliothèque 'rpnpy' est disponible pour la conversion RPN.")
except ImportError:
    HAS_RPN = False
    print("AVERTISSEMENT: La bibliothèque 'rpnpy' ou 'fstd2nc' est introuvable.")
    print("La conversion des fichiers RPN ne sera pas possible. Assurez-vous d'avoir chargé les modules nécessaires.")

# -----------------------------------------------------------------------------
# CONFIGURATION UTILISATEUR
# Modifiez les chemins et les options ci-dessous selon vos besoins.
# -----------------------------------------------------------------------------

# --- Chemins des fichiers d'entrée ---
SAT_DIR = os.path.join(os.environ['HOME'], 'Codes_travail', 'SAT_sim')
SAT_FILES = sorted(glob.glob(os.path.join(SAT_DIR, '*.h5')))

if not SAT_FILES:
    print(f"Erreur: Aucun fichier .h5 trouvé dans le répertoire satellite configuré : {SAT_DIR}")
    sys.exit(1)

# configuration pour un fichier modèle RPN unique contenant les deux variables
# Décommentez et définissez ce chemin si vous avez un fichier RPN unique
# SINGLE_MODEL_RPN_FILE = os.path.join(os.environ['HOME'], 'Codes_travail', 'MOD_sim', 'dm2007010100_20070101.001000s')
SINGLE_MODEL_RPN_FILE = None # Laissez à None si vous utilisez toujours deux fichiers séparés

MODEL_RPN_MAIN_VAR = os.path.join(os.environ['HOME'], 'Codes_travail', 'MOD_sim',
                                  'Arctic_gem48_no_spn_3km_D720_origine_final_IWCR_20070101.020000s')
MODEL_RPN_GZ_VAR = os.path.join(os.environ['HOME'], 'Codes_travail', 'MOD_sim',
                                'Arctic_gem48_no_spn_3km_D720_origine_final_GZ_20070101.020000s')

VAR_MODEL_PRIMARY = 'IWCR'
VAR_MODEL_GZ_NAME = 'GZ'

SAT_FIELD_IWC = '2C-ICE/Data Fields/IWC'
SAT_FIELD_HEIGHT = '2C-ICE/Geolocation Fields/Height'
SAT_FIELD_IWC_FACTOR = '2C-ICE/Swath Attributes/IWC.factor'
SAT_FIELD_IWC_MISSING = '2C-ICE/Swath Attributes/IWC.missing'

OUTPUT_IMAGE_BASE_NAME = 'match_sat_model_results.png'
RES_FIG_DIR = '/home/chevalier/Codes_travail/RES_FIG'

MAIN_MAP_ENABLED = True
ZOOM_ACT_MAIN_MAP = True
VERTICAL_PROFILE_ENABLED = True

VERTICAL_PLOT_ALTITUDE_MAX_KM = 15
VERTICAL_PLOT_MODEL_NAME = 'GEM model' 
TIME_MATCH_THRESHOLD_HOURS = 1

mpl.rcParams['agg.path.chunksize'] = 10000

# -----------------------------------------------------------------------------
# FONCTIONS INTERNES
# -----------------------------------------------------------------------------

def _unwrap_structured_array(arr):
    """
    Fonction utilitaire pour "déballer" les données si elles sont dans un tableau structuré (par ex. pour CALIPSO).
    """
    return arr[arr.dtype.names[0]] if hasattr(arr.dtype, 'names') and arr.dtype.names else arr


def _convert_rpn_to_netcdf_4d(rpn_path: str, varname_to_extract: str) -> str:
    """
    Convertit un fichier RPN pour une variable donnée en un fichier NetCDF 4D
    (dimensions: temps, niveau_altitude, y, x).
    Si le fichier NetCDF converti existe déjà, il est réutilisé.

    Args:
        rpn_path (str): Chemin complet vers le fichier RPN source.
        varname_to_extract (str): Nom de la variable RPN à extraire (ex: 'IWCR', 'GZ').

    Returns:
        str: Chemin complet vers le fichier NetCDF 4D converti.

    Raises:
        RuntimeError: Si la variable n'est pas trouvée ou si aucune donnée n'est extraite.
        ImportError: Si 'rpnpy' n'est pas disponible.
    """
    if not HAS_RPN:
        raise ImportError("La conversion RPN est requise mais 'rpnpy' n'est pas disponible.")

    ncfile = rpn_path + f'_{varname_to_extract}_4D.nc'

    if os.path.isfile(ncfile):
        print(f"  1.1) NetCDF 4D existant détecté pour '{varname_to_extract}' : {os.path.basename(ncfile)}")
        return ncfile
        
    print(f"  1.1) Conversion RPN → NetCDF 4D pour '{varname_to_extract}' : {os.path.basename(rpn_path)}")
    
    fid = None
    try:
        fid = rmn.fstopenall(rpn_path, rmn.FST_RO)
        
        keys = rmn.fstinl(fid, nomvar=varname_to_extract)
        if not keys:
            raise RuntimeError(f"Variable '{varname_to_extract}' non trouvée dans {os.path.basename(rpn_path)}")
        
        records_for_var = []
        unique_times = set()
        unique_levels = set()

        for k in keys:
            rec = rmn.fstluk(k)
            if 'd' not in rec:
                print(f"    Avertissement: L'enregistrement pour '{varname_to_extract}' à {rec.get('datev', 'temps inconnu')} ne contient pas de données ('d'). Ignoré.")
                continue

            current_time = stamp2datetime(rec['datev'])
            current_level = rec['ip1']
            
            unique_times.add(current_time)
            unique_levels.add(current_level)
            
            records_for_var.append({
                'time': current_time,
                'level': current_level,
                'data': rec['d'],
                'header': rec
            })

        if not records_for_var:
            raise RuntimeError(f"Aucune donnée valide pour la variable '{varname_to_extract}' trouvée dans {os.path.basename(rpn_path)}")

        sorted_times = sorted(list(unique_times))
        sorted_levels = sorted(list(unique_levels))
        
        rec0 = records_for_var[0]['header']
        grid = rmn.readGrid(fid, rec0)
        coords = rmn.gdll(grid)
        lat2d = coords['lat']
        lon2d = np.where(coords['lon'] > 180, coords['lon'] - 360, coords['lon'])
        
        ni, nj = rec0['ni'], rec0['nj']
        
        var_4d_data = np.full((len(sorted_times), len(sorted_levels), nj, ni), np.nan, dtype=np.float32)
        
        for record in records_for_var:
            t_idx = sorted_times.index(record['time'])
            l_idx = sorted_levels.index(record['level'])
            var_4d_data[t_idx, l_idx, :, :] = record['data']
        
        rmn.fstcloseall(fid)
        fid = None

        ds = xr.Dataset(
            {varname_to_extract: (('time', 'altitude_level', 'y', 'x'), var_4d_data)},
            coords={
                'time': np.array(sorted_times, dtype='datetime64[ns]'),
                'altitude_level': sorted_levels,
                'lat': (('y', 'x'), lat2d),
                'lon': (('y', 'x'), lon2d),
            }
        )
        ds['altitude_level'].attrs['units'] = 'Pa'
        ds['altitude_level'].attrs['long_name'] = 'Pressure Level (Pa)'

        ds.to_netcdf(ncfile)
        print(f"  1.1) Fichier NetCDF 4D sauvegardé : {os.path.basename(ncfile)}")
        return ncfile

    except Exception as e:
        print(f"Erreur lors de la conversion du fichier RPN '{os.path.basename(rpn_path)}' : {e}")
        if fid:
            rmn.fstcloseall(fid)
        raise

def _load_model_variable_4d(rpn_path: str, varname: str) -> Tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """
    Charge les données 4D (temps, niveau, y, x) d'une variable du modèle.
    Tente de charger un fichier NetCDF converti existant, sinon convertit le RPN.
    """
    if not os.path.exists(rpn_path) and not os.path.exists(rpn_path + f'_{varname}_4D.nc'):
        print(f"Erreur: Fichier modèle '{os.path.basename(rpn_path)}' ou son équivalent NetCDF 4D introuvable.")
        sys.exit(1)
        
    ncfile = _convert_rpn_to_netcdf_4d(rpn_path, varname)
    
    ds = xr.open_dataset(ncfile)
    
    lat2d = ds['lat'].values
    lon2d = ds['lon'].values
    
    print(f"  2.1) Données modèle chargées pour '{varname}', forme : {ds[varname].shape}")
    
    return ds, lat2d, lon2d

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
        def find_datasets_recursively(group, prefix=''):
            for k, obj in group.items():
                full_path = prefix + k
                if isinstance(obj, h5py.Dataset):
                    yield full_path
                elif isinstance(obj, h5py.Group):
                    yield from find_datasets_recursively(obj, full_path + '/')
        
        all_datasets_paths = list(find_datasets_recursively(h5f))
        
        geo_paths = [d for d in all_datasets_paths if 'geolocation' in d.lower()]
        
        lat_name = next((d for d in geo_paths if 'latitude' in d.lower() or 'lat' in d.lower()), None)
        lon_name = next((d for d in geo_paths if 'longitude' in d.lower() or 'lon' in d.lower()), None)
        profile_time_name = next((d for d in geo_paths if 'profile_time' in d.lower()), None)
        tai_start_name = next((d for d in geo_paths if 'tai_start' in d.lower()), None)
            
        if not lat_name or not lon_name or not profile_time_name or not tai_start_name:
            raise ValueError(f"Variables de géolocalisation ou de temps non trouvées dans {os.path.basename(path)}.\n"
                             f"Noms recherchés: lat='{lat_name}', lon='{lon_name}', profile_time='{profile_time_name}', tai_start='{tai_start_name}'")

        raw_lat = h5f[lat_name][...]
        raw_lon = h5f[lon_name][...]
        
        profile_time_raw = h5f[profile_time_name][...]
        tai_start_raw = h5f[tai_start_name][...]

        lat = _unwrap_structured_array(raw_lat).flatten()
        lon = _unwrap_structured_array(raw_lon).flatten()
        
        lon = ((lon + 180) % 360) - 180
        
        unwrapped_tai_start = _unwrap_structured_array(tai_start_raw).flatten()[0]
        
        base_epoch_datetime = datetime(1993, 1, 1, 0, 0, 0)
        
        granule_start_time_td = timedelta(seconds=int(unwrapped_tai_start))
        granule_start_time = np.datetime64(base_epoch_datetime + granule_start_time_td, 'ns')

        unwrapped_profile_time = _unwrap_structured_array(profile_time_raw).flatten().astype(int)

        time_arr = granule_start_time + np.timedelta64(1, 's') * unwrapped_profile_time
        
        try:
            if SAT_FIELD_IWC in h5f:
                raw_iwc = h5f[SAT_FIELD_IWC][...]
                iwc_data = _unwrap_structured_array(raw_iwc)
                
                iwc_factor = 1.0
                if SAT_FIELD_IWC_FACTOR in h5f.attrs: 
                    iwc_factor = h5f.attrs[SAT_FIELD_IWC_FACTOR]
                    if isinstance(iwc_factor, np.ndarray) and iwc_factor.size > 0:
                        iwc_factor = iwc_factor.flatten()[0]
                elif SAT_FIELD_IWC_FACTOR in h5f: 
                     iwc_factor_dset = h5f[SAT_FIELD_IWC_FACTOR]
                     iwc_factor = _unwrap_structured_array(iwc_factor_dset)[0]
                else:
                    print(f"    Avertissement: Facteur pour IWC ('{SAT_FIELD_IWC_FACTOR}') non trouvé. Utilisation de 1.0.")

                if iwc_factor != 0:
                    iwc_data = iwc_data / iwc_factor
                else:
                    print("    Avertissement: Facteur IWC est zéro, division par zéro évitée.")
                    iwc_data = np.full_like(iwc_data, np.nan)

                iwc_missing = None
                if SAT_FIELD_IWC_MISSING in h5f.attrs:
                    iwc_missing = h5f.attrs[SAT_FIELD_IWC_MISSING]
                    if isinstance(iwc_missing, np.ndarray) and iwc_missing.size > 0:
                        iwc_missing = iwc_missing.flatten()[0]
                elif SAT_FIELD_IWC_MISSING in h5f: 
                    iwc_missing_dset = h5f[SAT_FIELD_IWC_MISSING]
                    iwc_missing = _unwrap_structured_array(iwc_missing_dset)[0]
                
                if iwc_missing is not None:
                    iwc_data = np.ma.masked_where(iwc_data == iwc_missing, iwc_data)
                
                print(f"    IWC satellite extrait. Forme: {iwc_data.shape}")
            else:
                print(f"    Avertissement: Champ IWC ('{SAT_FIELD_IWC}') non trouvé dans le fichier satellite.")

            if SAT_FIELD_HEIGHT in h5f:
                raw_height = h5f[SAT_FIELD_HEIGHT][...]
                height_data = _unwrap_structured_array(raw_height).astype(np.float32)
                print(f"    Hauteur satellite extraite. Forme: {height_data.shape}")
            else:
                print(f"    Avertissement: Champ Hauteur ('{SAT_FIELD_HEIGHT}') non trouvé dans le fichier satellite.")

        except Exception as e:
            print(f"    AVERTISSEMENT: Erreur lors de l'extraction des données IWC/Hauteur du satellite: {e}")
            iwc_data = None
            height_data = None

        return lat, lon, time_arr, iwc_data, height_data

def _filter_in_domain(
    lat_sat: np.ndarray, lon_sat: np.ndarray, time_sat: np.ndarray,
    model_lat2d: np.ndarray, model_lon2d: np.ndarray,
    iwc_sat: Optional[np.ndarray] = None, height_sat: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filtre les points de la trajectoire satellite qui tombent dans le domaine spatial du modèle.
    Utilise un KD-Tree pour trouver les points de grille modèle les plus proches.
    Retourne également les données IWC et hauteur filtrées si elles sont fournies.
    """
    print(f"  4.1) Appariement de {len(lat_sat)} points satellite avec la grille du modèle...")
    
    model_points_flat = np.vstack((model_lat2d.ravel(), model_lon2d.ravel())).T
    tree = cKDTree(model_points_flat)

    satellite_points_flat = np.vstack((lat_sat, lon_sat)).T
    
    dists, indices = tree.query(satellite_points_flat)
    
    dlat = abs(model_lat2d[1, 0] - model_lat2d[0, 0]) if model_lat2d.shape[0] > 1 else 0.1
    dlon = abs(model_lon2d[0, 1] - model_lon2d[0, 0]) if model_lon2d.shape[1] > 1 else 0.1
    threshold = np.hypot(dlat / 2, dlon / 2)
    
    mask_in_domain = dists <= threshold
    
    print(f"  4.1) {mask_in_domain.sum()} points satellite trouvés dans le domaine (seuil={threshold:.3f}°).")
    
    iwc_in_domain = None
    if iwc_sat is not None and iwc_sat.shape[0] == len(mask_in_domain):
        iwc_in_domain = iwc_sat[mask_in_domain]
    elif iwc_sat is not None:
        print(f"    AVERTISSEMENT: Dimensions IWC satellite ({iwc_sat.shape[0]}) ne correspondent pas au masque ({len(mask_in_domain)}). IWC satellite ignoré pour le filtrage.")

    height_in_domain = None
    if height_sat is not None and height_sat.shape[0] == len(mask_in_domain):
        height_in_domain = height_sat[mask_in_domain]
    elif height_sat is not None:
        print(f"    AVERTISSEMENT: Dimensions hauteur satellite ({height_sat.shape[0]}) ne correspondent pas au masque ({len(mask_in_domain)}). Hauteur satellite ignorée pour le filtrage.")

    flat_idx = indices[mask_in_domain]
    yi, xi = np.unravel_index(flat_idx, model_lat2d.shape)

    return (
        lat_sat[mask_in_domain],
        lon_sat[mask_in_domain],
        time_sat[mask_in_domain],
        (yi, xi),           
        mask_in_domain,
        iwc_in_domain,
        height_in_domain
        )

def _plot_main_map(
    traj_lat_segments, traj_lon_segments,
    in_domain_coords, matched_indices,
    grid_lat2d, grid_lon2d,
    output_basename
) -> None:
    """
    Trace la vue globale (projection polaire) du domaine modèle
    et, si ZOOM_ACT_MAIN_MAP == True, ajoute un cadre de zoom + un 
    zoom détaillé en second panneau (aussi en projection polaire),
    en affichant tous les points de la grille modèle et le rectangle vert.
    """
    from matplotlib.patches import Polygon as MplPolygon

    # 1) Concaténation des trajectoires
    all_traj_lats = np.concatenate(traj_lat_segments)
    all_traj_lons = np.concatenate(traj_lon_segments)
    in_lats = np.concatenate([coords[0] for coords in in_domain_coords])
    in_lons = np.concatenate([coords[1] for coords in in_domain_coords])

    # 2) Étendue de la grille complète
    full_lon_min, full_lon_max = grid_lon2d.min(), grid_lon2d.max()
    full_lat_min, full_lat_max = grid_lat2d.min(), grid_lat2d.max()

    # 3) Détection du zoom
    zoom_active = globals().get('ZOOM_ACT_MAIN_MAP', False)
    ncols = 2 if zoom_active else 1
    fig = plt.figure(figsize=(14, 6) if zoom_active else (7, 6))
    gs = gridspec.GridSpec(1, ncols, wspace=0.3)

    # ----- Panneau 1 : Vue globale polaire -----
    ax0 = fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo())
    ax0.set_extent(
        [full_lon_min, full_lon_max, full_lat_min, full_lat_max],
        ccrs.PlateCarree()
    )
    ax0.coastlines('50m')
    ax0.add_feature(cfeature.LAND)
    ax0.gridlines(color='gray', linestyle=':')

    # a) toute la grille modèle
    ax0.scatter(grid_lon2d.ravel(), grid_lat2d.ravel(),s=5, color='lightgray', alpha=0.7,transform=ccrs.PlateCarree(),label='Grille modèle')

    # b) trajectoires
    palette = cm.get_cmap('tab10', len(traj_lat_segments))
     # tout le segment des trajectoires
    for i, (lats, lons) in enumerate(zip(traj_lat_segments, traj_lon_segments)):
        ax0.plot(lons, lats,linewidth=1.5,linestyle='dashed',color=palette(i),transform=ccrs.Geodetic(),label=f'Granule {i+1}')
     # segment in-domain des trajectoires seulement 
    for i, (in_lats, in_lons, *_ ) in enumerate(in_domain_coords):
        ax0.plot(in_lons, in_lats,linestyle='-',color=palette(i),transform=ccrs.Geodetic(),label=f'In-domain {i+1}')

    # c) cadre vert si zoom (sur la première granule only)
    if zoom_active:
        # on ne prend que la 1ʳᵉ granule pour le zoom
        in_lats0, in_lons0, *_, = in_domain_coords[0]
        yi0, xi0 = matched_indices[0]

        center0 = len(in_lats0) // 2
        cy, cx = yi0[center0], xi0[center0]

        half = 50
        y0, y1 = max(0, cy - half), min(grid_lat2d.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(grid_lon2d.shape[1], cx + half)

        zoom_lats = grid_lat2d[y0:y1, x0:x1]
        zoom_lons = grid_lon2d[y0:y1, x0:x1]
        lon_min, lon_max = zoom_lons.min(), zoom_lons.max()
        lat_min, lat_max = zoom_lats.min(), zoom_lats.max()

        bbox = [
            (lon_min, lat_min),
            (lon_min, lat_max),
            (lon_max, lat_max),
            (lon_max, lat_min),
        ]
        rect0 = MplPolygon(
            bbox, closed=True,
            edgecolor='green', facecolor='none', linewidth=2,
            transform=ccrs.PlateCarree(),
            label='Zone zoom (Granule 1)'
        )
        ax0.add_patch(rect0)
        
    ax0.set_title('Vue globale polaire')
    ax0.legend(loc='lower center',bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    # ----- Panneau 2 : Zoom in-domain (polaire) -----
    if zoom_active:
        ax1 = fig.add_subplot(gs[1], projection=ccrs.NorthPolarStereo())
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
        ax1.coastlines('50m')
        ax1.add_feature(cfeature.LAND)
        ax1.gridlines(color='gray', linestyle=':')

        # 1) tous les points de la grille (seront masqués hors étendue)
        ax1.scatter(
            grid_lon2d.ravel(), grid_lat2d.ravel(),
            s=5, color='lightgray', alpha=0.7,
            transform=ccrs.PlateCarree(),
            label='Grille modèle'
        )

        # 2) trajectoire in-domain de la 1ʳᵉ granule
        ax1.plot(in_lons0, in_lats0,linestyle='-',color=palette(0), transform=ccrs.Geodetic(),label='In-domain Granule 1'
        )

        # 3) points appariés (étoiles rouges) de la 1ʳᵉ granule
        flat0 = yi0 * grid_lon2d.shape[1] + xi0
        m_lons0 = grid_lon2d.ravel()[flat0]
        m_lats0 = grid_lat2d.ravel()[flat0]
        ax1.scatter(
            m_lons0, m_lats0,
            s=30, marker='*', color='red',
            transform=ccrs.PlateCarree(),
            label='Points appariés Granule 1'
        )

        # 4) même rectangle vert dans le zoom
        rect1 = MplPolygon(
            bbox, closed=True,
            edgecolor='green', facecolor='none', linewidth=2,
            transform=ccrs.PlateCarree(),
            label='Zone zoom (Granule 1)'
        )
        ax1.add_patch(rect1)

        ax1.set_title('Zoom in-domain (Granule 1)')
        ax1.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
            fontsize='small'
        )
    
    # ----- Finalisation & sauvegarde -----
    plt.suptitle('Colocalisation des trajectoires satelittes avec le domaine du modèle')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(
        RES_FIG_DIR,
        output_basename.replace('.png', '_polar_zoom.png')
    )
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"    → {os.path.basename(out_path)}")


# -----------------------------------------------------------------------------
# Fonction de tracé d'une coupe verticale du modèle le long d'une trajectoire
# -----------------------------------------------------------------------------
def _plot_model_vertical_curtain(
    model_ds,                # xarray.Dataset contenant les variables du modèle
    lat_sat,                 # 1-D array des latitudes in-domain
    lon_sat,                 # 1-D array des longitudes in-domain
    time_sat,                # 1-D array des datetime64 des instants in-domain
    matched_model_points,    # tuple (yi, xi) des indices y et x du modèle appariés
    granule_idx: int,        # indice du granule pour le titre / nom de fichier
    var_name: str,           # ex. 'IWCR'
    altitude_var_name: str,  # ex. 'GZ'
    model_display_name: str, # ex. 'GEM model'
    altitude_max_km: float,  # ex. 8.0
    output_base_name: str,   # ex. 'match_sat_model_results.png'
    time_match_threshold_hours: float = 6.0
):

    print(f"  → Tracé rideau modèle (granule {granule_idx})...")

    # 1) Dépack et dimensions
    yi, xi = matched_model_points
    npts = len(yi)
    nlev = model_ds[var_name].shape[1]
    times = model_ds['time'].values

    # 2) Extraire IWC pour chaque point et niveau
    data = np.full((nlev, npts), np.nan, dtype=float)
    for i_pt, (y, x) in enumerate(zip(yi, xi)):
        dt = np.abs((times - time_sat[i_pt]) / np.timedelta64(1, 'h'))
        idx = int(np.argmin(dt)) # idx est l’indice temporel du pas de temps du modèle qui « correspond le mieux » au temps de mesure satellite pour le point courant.
        if dt[idx] <= time_match_threshold_hours:
            data[:, i_pt] = model_ds[var_name].isel(time=idx, y=y, x=x).values

    # 3) Remplacer les zéros par NaN afin de ne pas les tracer
    data[data == 0.0] = np.nan 
    # data est un tableau de forme (nlev, npts) qui contient, pour chaque niveau de pression (nlev) et pour chaque point de 
    # la trajectoire in-domain (npts), la valeur de la variable d’intérêt (par ex. IWC) prélevée dans le modèle.
    
    # 4) Choisir un profil altitude (en km) sur un point valide
    valid_any = ~np.all(np.isnan(data), axis=0)
    if not valid_any.any():
        print("    Aucun profil vertical significatif.")
        return
    ref = np.where(valid_any)[0][0]
    y0, x0 = yi[ref], xi[ref]
    dt0 = np.abs((times - time_sat[ref]) / np.timedelta64(1, 'h'))
    tidx0 = int(np.argmin(dt0))
    alt = model_ds[altitude_var_name].isel(time=tidx0, y=y0, x=x0).values / 100.0 # Les unités de la variable data GZ extraite du fichier RPN est en dam (décamètre), et alors on divise par 100 pour la conversion en km

    # 5) Trier par altitude croissante
    order = np.argsort(alt)
    alt = alt[order]
    data = data[order, :]

    # 6) Bords de cellules pour pcolormesh
    x = np.arange(npts)
    x_edges = np.concatenate(([x[0] - 0.5], x + 0.5))
    dz = np.diff(alt)
    z_edges = np.empty(len(alt) + 1)
    z_edges[1:-1] = alt[:-1] + dz / 2
    z_edges[0] = alt[0] - dz[0] / 2
    z_edges[-1] = alt[-1] + dz[-1] / 2

    # 7) Colormap autoscale sur valeurs non-NaN
    vals = data[np.isfinite(data)]
    if vals.size:
        vmin, vmax = np.nanpercentile(vals, [5, 95])
    else:
        vmin, vmax = 1e-12, 1.0

    # 8) Tracé
    fig, ax = plt.subplots(figsize=(10, 4))
    norm = mpl.colors.LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
    pcm = ax.pcolormesh(
        x_edges, z_edges, data,
        shading='flat', cmap='jet', norm=norm
    )
    cb = fig.colorbar(pcm, ax=ax, extend='both')
    cb.set_label(f"{var_name} (g·m⁻³)")

    # 9) Axes
    ax.set_ylim(0, altitude_max_km)
    ax.set_ylabel('Altitude (km)')
    nt = min(6, npts)
    ticks = np.linspace(0, npts - 1, nt, dtype=int)
    labels = [f"Lon: {lon_sat[i]:.2f}°\nLat: {lat_sat[i]:.2f}°" for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Points de la grille modèle à proximité de la granule satelitte')
    ax.set_title(
        f"{model_display_name}: Profil vertical de {var_name} à "
        f"{np.datetime_as_string(time_sat[ref], unit='s')} UTC "
        f"(Granule {granule_idx})"
    )

    plt.tight_layout()
    out = output_base_name.replace('.png', f'_modc_{granule_idx}.png')
    full_path_out = os.path.join(RES_FIG_DIR, out)
    fig.savefig(full_path_out, dpi=300)
    plt.close(fig)
    print(f"  → Sauvegardé : {os.path.basename(out)}")

def _plot_satellite_curtain(
    lat_sat_in: np.ndarray,
    lon_sat_in: np.ndarray,
    time_sat_in: np.ndarray,
    iwc_sat_in: np.ndarray,
    height_sat_in: np.ndarray,
    granule_idx: int,
    output_base_name: str
) -> None:
    """
    Trace le rideau vertical de l'IWC satellite le long de sa trajectoire in-domain.
    Comparable à _plot_vertical_curtain pour le modèle, mais à partir des données CALIPSO.
    """
    print(f"  → Tracé rideau satellite (granule {granule_idx})...")

    # Nombre de points et de couches verticales
    n_pts, n_levels = iwc_sat_in.shape

    # 1) Construire la matrice (levels × points)
    #    iwc_sat_in : shape (points, levels) → data : (levels, points)
    data = iwc_sat_in.T

    # 2) Altitudes (en km) par niveau
    #    height_sat_in : shape (points, levels) — on suppose que chaque profil a les mêmes hauteurs
    alt = height_sat_in[0, :] / 1_000.0  # conversion m → km

    # 3) Trier par altitude croissante
    order = np.argsort(alt)
    alt = alt[order]
    data = data[order, :]

    # 4) Bords de cellules pour pcolormesh
    # X : points indexés de 0 à n_pts-1
    x = np.arange(n_pts)
    x_edges = np.concatenate(([x[0] - 0.5], x + 0.5))

    # Z : niveaux d'altitude
    dz = np.diff(alt)
    z_edges = np.empty(len(alt) + 1)
    z_edges[1:-1] = alt[:-1] + dz / 2
    z_edges[0]      = alt[0] - dz[0] / 2
    z_edges[-1]     = alt[-1] + dz[-1] / 2

    # 5) Colormap sur IWC (log scale), garantir vmin>0 et vmax>vmin
    # on ne retient que les valeurs strictement > 0 pour la stats
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size:
        p5, p95 = np.nanpercentile(positive, [5, 95])
        vmin = max(p5, 1e-6)
        vmax = max(p95, vmin * 10)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        # cas où tout est nul ou invalide : on passe en échelle linéaire
        norm = None

    # 6) Tracé
    fig, ax = plt.subplots(figsize=(10, 4))
    pcm = ax.pcolormesh(
        x_edges, z_edges, data,
        shading='flat', cmap='viridis',
        norm=norm
    )
    if norm:
        cb = fig.colorbar(pcm, ax=ax, extend='both')
        cb.set_label('IWC satellite (g·m⁻³)')
    else:
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label('IWC satellite (g·m⁻³) – échelle linéaire')

    # 7) Axes
    # ax.set_ylim(0, np.max(alt) * 1.05)
    ax.set_ylim(0, VERTICAL_PLOT_ALTITUDE_MAX_KM)
    ax.set_ylabel('Altitude (km)')

    # X-ticks avec lat/lon
    nt = min(6, n_pts)
    ticks = np.linspace(0, n_pts - 1, nt, dtype=int)
    labels = [
        f"Lon: {lon_sat_in[i]:.2f}°\nLat: {lat_sat_in[i]:.2f}°"
        for i in ticks
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel("Point le long de la trajectoire satellite")

    # Titre
    time_ref = np.datetime_as_string(time_sat_in[ticks[0]], unit='s')
    ax.set_title(
        f"CALIPSO – Rideau vertical d'IWC à {time_ref} UTC\n"
        f"(Granule {granule_idx})"
    )

    plt.tight_layout()
    out_name = output_base_name.replace('.png', f'_satc_{granule_idx}.png')
    out_path = os.path.join(RES_FIG_DIR, out_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"  → Sauvegardé : {os.path.basename(out_name)}")

# -----------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    """
    Fonction principale du script. Orchestre le chargement des données,
    le filtrage, et la génération des différents tracés.
    """
    start_time_script = time.time()
    print("==== DÉMARRAGE DU SCRIPT ====")

    # Étape 0: nettoyage du dossier de sortie
    if os.path.exists(RES_FIG_DIR):
        for f in glob.glob(os.path.join(RES_FIG_DIR, '*.png')):
            os.remove(f)
    else:
        os.makedirs(RES_FIG_DIR)
    print(f"Étape 0 : répertoire de sortie vidé -> {RES_FIG_DIR}")

    if SINGLE_MODEL_RPN_FILE and os.path.exists(SINGLE_MODEL_RPN_FILE):
        print("\nÉtape 1: Chargement des variables du modèle à partir d'un fichier RPN unique...")
        ds_main_var, model_lat2d_grid, model_lon2d_grid = _load_model_variable_4d(
            SINGLE_MODEL_RPN_FILE, VAR_MODEL_PRIMARY
        )
        ds_gz_var, _, _ = _load_model_variable_4d(
            SINGLE_MODEL_RPN_FILE, VAR_MODEL_GZ_NAME
        )
    else:
        print("\nÉtape 1: Chargement de la variable principale du modèle à partir de son fichier RPN...")
        ds_main_var, model_lat2d_grid, model_lon2d_grid = _load_model_variable_4d(
            MODEL_RPN_MAIN_VAR, VAR_MODEL_PRIMARY
        )
        print("\nÉtape 2: Chargement de la variable de hauteur géopotentielle (GZ) à partir de son fichier RPN...")
        ds_gz_var, _, _ = _load_model_variable_4d(
            MODEL_RPN_GZ_VAR, VAR_MODEL_GZ_NAME
        )

    print("\nÉtape 3: Fusion des Datasets du modèle...")
    try:
        common_lat = ds_main_var['lat']
        common_lon = ds_main_var['lon']
        ds_gz_var = ds_gz_var.assign_coords(lat=common_lat, lon=common_lon)
        model_ds_merged = xr.merge([ds_main_var, ds_gz_var])
        print("  3.0) Datasets du modèle (variable principale et GZ) fusionnés avec succès.")
    except Exception as e:
        print(f"Erreur lors de la fusion des Datasets du modèle. Erreur: {e}")
        sys.exit(1)

    all_satellite_lats_full = []
    all_satellite_lons_full = []
    all_in_domain_data_list = []
    all_matched_model_points_list = []
    all_in_domain_masks = []

    print("\nÉtape 4: Traitement des fichiers satellite, appariement et génération des profils verticaux pour le modèle (si activé)...")
    for idx, sat_filepath in enumerate(SAT_FILES, 1):
        print(f"\n--- Traitement du Granule {idx}/{len(SAT_FILES)} : {os.path.basename(sat_filepath)} ---")
        try:
            lat_satellite_full, lon_satellite_full, time_satellite_full, iwc_satellite_full, height_satellite_full = _read_satellite_h5(sat_filepath)

            lat_in_domain, lon_in_domain, time_in_domain, matched_model_points, mask_in_domain, iwc_in_domain, height_in_domain = \
                _filter_in_domain(
                    lat_satellite_full, lon_satellite_full, time_satellite_full,
                    model_lat2d_grid, model_lon2d_grid,
                    iwc_sat=iwc_satellite_full, height_sat=height_satellite_full
                )

            all_satellite_lats_full.append(lat_satellite_full)
            all_satellite_lons_full.append(lon_satellite_full)
            all_in_domain_masks.append(mask_in_domain)

            if lat_in_domain.size > 0:
                all_in_domain_data_list.append((lat_in_domain, lon_in_domain, time_in_domain, iwc_in_domain, height_in_domain))
                all_matched_model_points_list.append(matched_model_points)

                if VERTICAL_PROFILE_ENABLED and time_in_domain.size > 0:
                    _plot_model_vertical_curtain(
                        model_ds_merged,
                        lat_in_domain,
                        lon_in_domain,
                        time_in_domain,
                        matched_model_points,
                        granule_idx=idx,
                        var_name=VAR_MODEL_PRIMARY,
                        altitude_var_name=VAR_MODEL_GZ_NAME,
                        model_display_name=VERTICAL_PLOT_MODEL_NAME,
                        altitude_max_km=VERTICAL_PLOT_ALTITUDE_MAX_KM,
                        output_base_name=OUTPUT_IMAGE_BASE_NAME,
                        time_match_threshold_hours=TIME_MATCH_THRESHOLD_HOURS
                    )
                    _plot_satellite_curtain(
                        lat_in_domain,
                        lon_in_domain,
                        time_in_domain,
                        iwc_in_domain,
                        height_in_domain,
                        idx,
                        OUTPUT_IMAGE_BASE_NAME
                    )
                else:
                    print(f"    Avertissement: Pas de temps de survol valide pour le granule {idx}, la coupe verticale est ignorée.")
            else:
                print(f"  4.0) Aucun point in-domain trouvé pour le granule {idx}. Les tracés spécifiques à ce granule sont ignorés.")

        except (ValueError, RuntimeError, KeyError, ImportError) as e:
            print(f"AVERTISSEMENT: Impossible de traiter le fichier satellite '{os.path.basename(sat_filepath)}'. Erreur: {e}")
            continue

    print("\nÉtape 5: Génération de la carte principale (globale et zoom)...")
    if all_in_domain_data_list and MAIN_MAP_ENABLED:
        _plot_main_map(
            all_satellite_lats_full,
            all_satellite_lons_full,
            all_in_domain_data_list,
            all_matched_model_points_list,
            model_lat2d_grid,
            model_lon2d_grid,
            OUTPUT_IMAGE_BASE_NAME
        )
    elif MAIN_MAP_ENABLED is False: 
        print ("L'option de génération de la figure principale a été désactivée")
    else:
        print("\n\nAucun point de satellite trouvé dans le domaine du modèle sur l'ensemble des fichiers.")
        print("Les graphiques principaux et polaires ne seront pas générés.")

    script_duration = time.time() - start_time_script
    print(f"\n✅ Script terminé en {script_duration:.1f} secondes.")

if __name__ == '__main__':
    main()