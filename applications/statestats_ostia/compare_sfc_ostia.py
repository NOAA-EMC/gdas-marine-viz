#!/usr/bin/env python3
"""
compare_sfc_ostia.py
Validate background SST (foundation temperature) and sea-ice concentration
from ops and parallel experiments against OSTIA L4 analyses.

All fields are interpolated to a common 0.5° regular grid.
Background daily means are computed from all available cycles before
comparison against the 12 Z OSTIA analysis.

Outputs
-------
1.  Per-basin time series of bias and RMSE  (sst_ts.png, ice_ts.png)
2.  Spatial maps of time-mean bias and RMSE (sst_maps_ops.png,
    sst_maps_par.png, ice_maps_ops.png, ice_maps_par.png)

Usage
-----
    python3 compare_sfc_ostia.py [--ops-dir ops] [--exps-dir exp1 exp2 ...] \
        [--ostia-dir ostia_raw] [--mask-file RECCAP2_region_masks_all.nc] \
        [--output-dir plots_sfc]
"""

import argparse
import glob
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================================== #
#  Target grid  (0.5°)
# =========================================================================== #
TGT_LAT = np.arange(-89.75, 90.0, 0.5)   # 360 points
TGT_LON = np.arange(0.25, 360.0, 0.5)    # 720 points

# Grid cell areas (km²) on the 0.5° target grid
_R_EARTH_KM = 6371.0
_DLAT = np.radians(0.5)
_DLON = np.radians(0.5)
_LAT_EDGES = np.radians(np.arange(-90.0, 90.5, 0.5))  # 361 edges
CELL_AREA = (_R_EARTH_KM ** 2 * _DLON *
             (np.sin(_LAT_EDGES[1:]) - np.sin(_LAT_EDGES[:-1])))[:, None] \
            * np.ones((1, len(TGT_LON)))   # (360, 720) km²

# Ice-extent threshold
ICE_EXTENT_THRESH = 0.15


# =========================================================================== #
#  Helpers – discover background files
# =========================================================================== #
_OPS_PAT = "gdas.t{HH}z.sfcf006.nc"
_PAR_PAT = "gdas.t{HH}z.sfc.f006.nc"


def discover_sfc_files(base_dir):
    """Return {YYYYMMDD: [file_paths]} for available sfcf006 files.

    Tries both ops naming (sfcf006.nc) and parallel naming (sfc.f006.nc).
    """
    days = defaultdict(list)
    for pat in (
        "gdas.*/*/model/atmos/history/gdas.*.sfcf006.nc",
        "gdas.*/*/model/atmos/history/gdas.*.sfc.f006.nc",
        "gdas.*/*/model/atmos/gdas.*.sfcf006.nc",
        "gdas.*/*/model/atmos/gdas.*.sfc.f006.nc",
        "gdas.*/*/atmos/gdas.*.sfcf006.nc",
        "gdas.*/*/atmos/gdas.*.sfc.f006.nc",
    ):
        for f in sorted(glob.glob(os.path.join(base_dir, pat))):
            m = re.search(
                r"gdas\.(\d{8})/(\d{2})/(?:model/atmos(?:/history)?|atmos)/",
                f,
            )
            if m:
                days[m.group(1)].append(f)
    # deduplicate in case both patterns match the same file
    for d in days:
        days[d] = sorted(set(days[d]))
    return dict(days)


def ostia_path(ostia_dir, date_str):
    """Return the OSTIA file path for a given YYYYMMDD date string."""
    return os.path.join(
        ostia_dir,
        date_str[:4], date_str[4:6],
        f"{date_str}120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc",
    )


# =========================================================================== #
#  Interpolation to target grid
# =========================================================================== #
def _interp_gaussian_to_target(lat2d, lon2d, field2d):
    """Interpolate a 2-D field on the Gaussian grid to TGT_LAT × TGT_LON.

    Uses bin-averaging for speed and robustness with the irregular Gaussian
    grid.  Returns an (nlat, nlon) array on the target grid with NaN where
    no data fell.
    """
    nlat = len(TGT_LAT)
    nlon = len(TGT_LON)
    lat_edges = np.linspace(-90, 90, nlat + 1)
    lon_edges = np.linspace(0, 360, nlon + 1)

    flat_lat = lat2d.ravel()
    flat_lon = lon2d.ravel()
    flat_val = field2d.ravel()

    # remove NaN / masked
    good = np.isfinite(flat_val) & np.isfinite(flat_lat) & np.isfinite(flat_lon)
    flat_lat = flat_lat[good]
    flat_lon = flat_lon[good]
    flat_val = flat_val[good]

    if len(flat_val) == 0:
        return np.full((nlat, nlon), np.nan)

    # ensure lon in [0, 360)
    flat_lon = flat_lon % 360.0

    ilat = np.clip(np.digitize(flat_lat, lat_edges) - 1, 0, nlat - 1)
    ilon = np.clip(np.digitize(flat_lon, lon_edges) - 1, 0, nlon - 1)

    sumv = np.zeros((nlat, nlon))
    cnt  = np.zeros((nlat, nlon))
    np.add.at(sumv, (ilat, ilon), flat_val)
    np.add.at(cnt,  (ilat, ilon), 1.0)

    result = np.where(cnt > 0, sumv / cnt, np.nan)
    return result


def _interp_ostia_to_target(ostia_lat, ostia_lon, field2d):
    """Interpolate OSTIA (regular 0.05° grid, lon in [-180,180]) to target.

    Uses scipy RegularGridInterpolator (bilinear).
    """
    # OSTIA lat is ascending, lon is ascending  (-179.975 .. 179.975)
    # Shift lon to [0, 360) for the target grid
    # We rearrange the OSTIA data so lon goes 0..360
    idx_shift = np.searchsorted(ostia_lon, 0.0)
    lon_shifted = np.concatenate([ostia_lon[idx_shift:] + 0,
                                  ostia_lon[:idx_shift] + 360])
    field_shifted = np.concatenate([field2d[:, idx_shift:],
                                    field2d[:, :idx_shift]], axis=1)

    # Build interpolator
    interp = RegularGridInterpolator(
        (ostia_lat, lon_shifted), field_shifted,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    tgt_lon_grid, tgt_lat_grid = np.meshgrid(TGT_LON, TGT_LAT)
    pts = np.stack([tgt_lat_grid.ravel(), tgt_lon_grid.ravel()], axis=-1)
    return interp(pts).reshape(len(TGT_LAT), len(TGT_LON))


# =========================================================================== #
#  Read a single background file → tref & icec on target grid
# =========================================================================== #
def read_bkg(filepath):
    """Return (tref_tgt, icec_tgt) on the 0.5° target grid.

    - SST (tref): valid over ocean (land==0).
    - Ice (icec): valid over ocean and ice (land==0 or land==2);
      land-only points (land==1) are masked.
    """
    ds = nc.Dataset(filepath)
    lat2d = ds.variables["lat"][:].squeeze()
    lon2d = ds.variables["lon"][:].squeeze()
    tref  = ds.variables["tref"][0, :, :].astype(np.float64)
    icec  = ds.variables["icec"][0, :, :].astype(np.float64)
    land  = ds.variables["land"][0, :, :].astype(np.float64)
    ds.close()

    # SST: ocean only (land==0)
    tref = np.where(land == 0, tref, np.nan)
    # Ice concentration: ocean + ice (land != 1)
    icec = np.where(land != 1, icec, np.nan)

    tref_tgt = _interp_gaussian_to_target(lat2d, lon2d, tref)
    icec_tgt = _interp_gaussian_to_target(lat2d, lon2d, icec)

    return tref_tgt, icec_tgt


def daily_mean_bkg(file_list):
    """Compute the daily mean of tref and icec from multiple cycles."""
    trefs, icecs = [], []
    for f in file_list:
        try:
            t, i = read_bkg(f)
            trefs.append(t)
            icecs.append(i)
        except Exception as e:
            print(f"    WARNING: skipping {f}: {e}")
    if not trefs:
        return None, None
    tref_mean = np.nanmean(np.array(trefs), axis=0)
    icec_mean = np.nanmean(np.array(icecs), axis=0)
    return tref_mean, icec_mean


# =========================================================================== #
#  Read OSTIA → sst & ice on target grid
# =========================================================================== #
def read_ostia(filepath):
    """Return (sst_tgt, ice_tgt) on the 0.5° target grid."""
    ds = nc.Dataset(filepath)
    olat = ds.variables["lat"][:].astype(np.float64)
    olon = ds.variables["lon"][:].astype(np.float64)
    sst  = ds.variables["analysed_sst"][0, :, :].astype(np.float64)
    ice  = ds.variables["sea_ice_fraction"][0, :, :].astype(np.float64)
    ds.close()

    # Fill masked values with NaN
    if hasattr(sst, "mask"):
        sst = np.where(sst.mask, np.nan, sst.data)
    if hasattr(ice, "mask"):
        ice = np.where(ice.mask, np.nan, ice.data)

    sst_tgt = _interp_ostia_to_target(olat, olon, sst)
    ice_tgt = _interp_ostia_to_target(olat, olon, ice)
    return sst_tgt, ice_tgt


# =========================================================================== #
#  RECCAP2 basin masks on target grid
# =========================================================================== #
def load_basin_masks(mask_file):
    """Return {basin_name: 2-D bool mask on TGT grid}.

    The RECCAP2 mask is on a 1° grid; we nearest-neighbour it to 0.5°.
    """
    ds = nc.Dataset(mask_file)
    mlat = ds.variables["lat"][:].astype(np.float64)
    mlon = ds.variables["lon"][:].astype(np.float64)

    basins = {}
    for bname in ("open_ocean", "atlantic", "pacific",
                   "indian", "southern", "arctic"):
        raw = ds.variables[bname][:].astype(np.float64)
        # Nearest-neighbour to 0.5° target grid
        ilat = np.clip(np.searchsorted(mlat, TGT_LAT) , 0, len(mlat) - 1)
        ilon = np.clip(np.searchsorted(mlon, TGT_LON) , 0, len(mlon) - 1)
        ilon_g, ilat_g = np.meshgrid(ilon, ilat)
        basins[bname] = raw[ilat_g, ilon_g] > 0
    ds.close()
    return basins


# =========================================================================== #
#  Accumulate daily statistics
# =========================================================================== #
def compute_daily_stats(bkg_files_by_day, ostia_dir, basins):
    """
    For each day that has both background and OSTIA data, compute
    per-basin bias/RMSE for SST and hemisphere bias/RMSE for ice.

    Returns
    -------
    sst_basin_stats : dict
        { basin: { "dates": [], "sst_bias": [], "sst_rmse": [] } }
    ice_hemi_stats : dict
        { "north"|"south": { "dates": [], "ice_bias": [], "ice_rmse": [] } }
    spatial : dict
        { "sst_diff_sum":  2-D,  "sst_diff_sq_sum": 2-D,
          "ice_diff_sum":  2-D,  "ice_diff_sq_sum": 2-D,
          "sst_count": 2-D,  "ice_count": 2-D }
    """
    nlat, nlon = len(TGT_LAT), len(TGT_LON)

    sst_basin_stats = {b: {"dates": [], "sst_bias": [], "sst_rmse": []}
                       for b in basins}

    ice_hemi_stats = {h: {"dates": [], "ice_bias": [], "ice_rmse": []}
                      for h in ("north", "south")}

    # Precompute hemisphere masks on target grid
    LAT_2D = TGT_LAT[:, None] * np.ones((1, nlon))
    hemi_masks = {"north": LAT_2D >= 45, "south": LAT_2D < -45}

    spatial = {k: np.zeros((nlat, nlon)) for k in
               ("sst_diff_sum", "sst_diff_sq_sum",
                "ice_diff_sum", "ice_diff_sq_sum",
                "sst_count", "ice_count")}

    sorted_days = sorted(bkg_files_by_day.keys())
    n_missing_ostia = 0
    for day_str in sorted_days:
        opath = ostia_path(ostia_dir, day_str)
        if not os.path.isfile(opath):
            print(f"    {day_str} ... skip (missing OSTIA: {opath})")
            n_missing_ostia += 1
            continue

        print(f"    {day_str} ...", end="", flush=True)

        # Daily mean background
        tref_bkg, icec_bkg = daily_mean_bkg(bkg_files_by_day[day_str])
        if tref_bkg is None:
            print(" skip (no bkg)")
            continue

        # OSTIA
        sst_ostia, ice_ostia = read_ostia(opath)

        # Mask OSTIA SST where OSTIA ice > 0
        sst_ostia = np.where((ice_ostia > 0) | np.isnan(ice_ostia), np.nan, sst_ostia)

        # Also mask background SST where OSTIA ice > 0
        tref_bkg_masked = np.where((ice_ostia > 0) | np.isnan(ice_ostia), np.nan, tref_bkg)

        # SST difference (bkg − OSTIA)
        sst_diff = tref_bkg_masked - sst_ostia
        # Ice difference
        ice_diff = icec_bkg - ice_ostia

        # Spatial accumulation (only where both are valid)
        sst_good = np.isfinite(sst_diff)
        ice_good = np.isfinite(ice_diff)

        # SST spatial
        spatial["sst_diff_sum"][sst_good]    += sst_diff[sst_good]
        spatial["sst_diff_sq_sum"][sst_good] += sst_diff[sst_good] ** 2
        spatial["sst_count"][sst_good]       += 1.0

        # Ice spatial
        spatial["ice_diff_sum"][ice_good]    += ice_diff[ice_good]
        spatial["ice_diff_sq_sum"][ice_good] += ice_diff[ice_good] ** 2
        spatial["ice_count"][ice_good]       += 1.0

        dt = datetime.strptime(day_str, "%Y%m%d")

        # Per-basin SST stats
        for bname, bmask in basins.items():
            m = bmask & sst_good
            n = np.sum(m)
            if n > 0:
                sst_b = float(np.mean(sst_diff[m]))
                sst_r = float(np.sqrt(np.mean(sst_diff[m] ** 2)))
            else:
                sst_b, sst_r = np.nan, np.nan

            sst_basin_stats[bname]["dates"].append(dt)
            sst_basin_stats[bname]["sst_bias"].append(sst_b)
            sst_basin_stats[bname]["sst_rmse"].append(sst_r)

        # Per-hemisphere ice stats
        for hname, hmask in hemi_masks.items():
            m_ice = hmask & ice_good
            n_ice = np.sum(m_ice)
            if n_ice > 0:
                ice_b = float(np.mean(ice_diff[m_ice]))
                ice_r = float(np.sqrt(np.mean(ice_diff[m_ice] ** 2)))
            else:
                ice_b, ice_r = np.nan, np.nan

            ice_hemi_stats[hname]["dates"].append(dt)
            ice_hemi_stats[hname]["ice_bias"].append(ice_b)
            ice_hemi_stats[hname]["ice_rmse"].append(ice_r)

        print(f" OK ({np.sum(sst_good)} sst pts, {np.sum(ice_good)} ice pts)")

    if n_missing_ostia > 0:
        print(f"  NOTE: skipped {n_missing_ostia} day(s) with missing OSTIA files")

    # Convert lists to arrays
    for bname in sst_basin_stats:
        for k in sst_basin_stats[bname]:
            sst_basin_stats[bname][k] = np.array(sst_basin_stats[bname][k])
    for hname in ice_hemi_stats:
        for k in ice_hemi_stats[hname]:
            ice_hemi_stats[hname][k] = np.array(ice_hemi_stats[hname][k])

    return sst_basin_stats, ice_hemi_stats, spatial


# =========================================================================== #
#  Sea-ice extent on a common mask (intersection of ops, parallel & OSTIA)
# =========================================================================== #
def compute_ice_extent(days_by_run, ostia_dir):
    """
    Compute daily sea-ice extent for OSTIA and all runs on a common mask.
    For each day the valid-ice mask is the intersection of all available
    sources so that extent differences reflect only ice concentration
    differences, not mask differences.

    Returns
    -------
    ice_extent : dict
        { "north"|"south": { "dates": [], "ostia_km2": [],
                             "runs": {run_label: []} } }
    """
    nlat, nlon = len(TGT_LAT), len(TGT_LON)
    LAT_2D = TGT_LAT[:, None] * np.ones((1, nlon))
    hemi_masks = {"north": LAT_2D >= 45, "south": LAT_2D < -45}

    ice_extent = {h: {"dates": [], "ostia_km2": [],
                      "runs": {label: [] for label in days_by_run}}
                  for h in ("north", "south")}

    # Collect all days present in any run
    all_days = sorted(set().union(*(days.keys() for days in days_by_run.values())))

    for day_str in all_days:
        opath = ostia_path(ostia_dir, day_str)
        if not os.path.isfile(opath):
            continue

        # Read OSTIA ice
        _, ice_ostia = read_ostia(opath)

        ice_by_run = {}
        for label, days in days_by_run.items():
            if day_str in days:
                _, ice_by_run[label] = daily_mean_bkg(days[day_str])

        # Common valid-ice mask: intersection of all available sources
        common_valid = np.isfinite(ice_ostia)
        for icec in ice_by_run.values():
            if icec is not None:
                common_valid &= np.isfinite(icec)

        dt = datetime.strptime(day_str, "%Y%m%d")

        for hname, hmask in hemi_masks.items():
            mask = hmask & common_valid

            ost_ext = float(np.sum(
                CELL_AREA[mask & (ice_ostia > ICE_EXTENT_THRESH)]))

            ice_extent[hname]["dates"].append(dt)
            ice_extent[hname]["ostia_km2"].append(ost_ext)
            for label in days_by_run:
                icec = ice_by_run.get(label)
                ext = float(np.sum(
                    CELL_AREA[mask & (icec > ICE_EXTENT_THRESH)])) \
                    if icec is not None else np.nan
                ice_extent[hname]["runs"][label].append(ext)

        print(f"    {day_str}  ice extent OK "
              f"(common pts: {int(np.sum(common_valid))})")

    # Convert to arrays
    for hname in ice_extent:
        for k in ice_extent[hname]:
            if k == "runs":
                for label in ice_extent[hname][k]:
                    ice_extent[hname][k][label] = np.array(ice_extent[hname][k][label])
            else:
                ice_extent[hname][k] = np.array(ice_extent[hname][k])

    return ice_extent


# =========================================================================== #
#  Integrated Ice Edge Error (IIEE) on a common mask
# =========================================================================== #
def compute_ice_edge_error(days_by_run, ostia_dir):
    """
    Compute daily Integrated Ice Edge Error (IIEE) for all runs against
    OSTIA, per hemisphere.

    IIEE is the area of the symmetric difference between the ice-covered
    regions of background and OSTIA (using the 15 % threshold).  It is
    decomposed into:
      - overestimation  (bkg says ice, OSTIA does not)
      - underestimation (OSTIA says ice, bkg does not)

    Land and lakes are masked out by requiring finite ice values in both
    OSTIA and the background (intersection mask).

    Returns
    -------
    iiee : dict
        { "north"|"south":
            { "dates": [],
                            "runs": {run_label: {"iiee": [], "over": [], "under": []}} } }
    """
    nlat, nlon = len(TGT_LAT), len(TGT_LON)
    LAT_2D = TGT_LAT[:, None] * np.ones((1, nlon))
    hemi_masks = {"north": LAT_2D >= 45, "south": LAT_2D < -45}

    iiee = {h: {"dates": [],
                "runs": {label: {"iiee": [], "over": [], "under": []}
                         for label in days_by_run}}
            for h in ("north", "south")}

    all_days = sorted(set().union(*(days.keys() for days in days_by_run.values())))

    for day_str in all_days:
        opath = ostia_path(ostia_dir, day_str)
        if not os.path.isfile(opath):
            continue

        _, ice_ostia = read_ostia(opath)
        ostia_valid = np.isfinite(ice_ostia)
        ostia_ice = ostia_valid & (ice_ostia > ICE_EXTENT_THRESH)

        ice_by_run = {}
        for label, days in days_by_run.items():
            if day_str in days:
                _, ice_by_run[label] = daily_mean_bkg(days[day_str])

        dt = datetime.strptime(day_str, "%Y%m%d")

        for hname, hmask in hemi_masks.items():
            iiee[hname]["dates"].append(dt)
            for label in days_by_run:
                icec = ice_by_run.get(label)
                if icec is not None:
                    # Common ocean mask: both OSTIA and bkg valid (land/lakes out)
                    valid = hmask & ostia_valid & np.isfinite(icec)
                    bkg_ice = valid & (icec > ICE_EXTENT_THRESH)
                    ost_ice_common = valid & ostia_ice
                    over = bkg_ice & ~ost_ice_common   # bkg ice, OSTIA open water
                    under = ost_ice_common & ~bkg_ice   # OSTIA ice, bkg open water
                    iiee_val = float(np.sum(CELL_AREA[over | under]))
                    over_val = float(np.sum(CELL_AREA[over]))
                    under_val = float(np.sum(CELL_AREA[under]))
                else:
                    iiee_val = over_val = under_val = np.nan

                iiee[hname]["runs"][label]["iiee"].append(iiee_val)
                iiee[hname]["runs"][label]["over"].append(over_val)
                iiee[hname]["runs"][label]["under"].append(under_val)

        print(f"    {day_str}  IIEE OK")

    # Convert to arrays
    for hname in iiee:
        for k in iiee[hname]:
            if k == "runs":
                for label in iiee[hname][k]:
                    for metric in iiee[hname][k][label]:
                        iiee[hname][k][label][metric] = np.array(iiee[hname][k][label][metric])
            else:
                iiee[hname][k] = np.array(iiee[hname][k])

    return iiee


# =========================================================================== #
#  Plotting – time series
# =========================================================================== #
def _format_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


def _safe_tag(label):
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return tag.strip("._-") or "exp"


def plot_timeseries(stats_by_run, var, out_dir):
    """
    Plot per-region time series of bias and RMSE.
    var = "sst" (regions = ocean basins) or "ice" (regions = north / south).
    """
    regions = sorted({region
                      for stats in stats_by_run.values()
                      for region in stats.keys()})
    # Put "global" (was "open_ocean") first for SST time series
    if var == "sst" and "open_ocean" in regions:
        regions = ["open_ocean"] + [r for r in regions if r != "open_ocean"]
    nregions = len(regions)
    if nregions == 0:
        return

    fig, axes = plt.subplots(nregions, 2, figsize=(16, 3.5 * nregions),
                             squeeze=False, constrained_layout=True)

    units = "K" if var == "sst" else "fraction"
    title_var = "Foundation SST" if var == "sst" else "Ice Concentration"
    region_label = "basin" if var == "sst" else "hemisphere"
    fig.suptitle(f"{title_var} vs OSTIA — per-{region_label} bias & RMSE ({units})",
                 fontsize=15, fontweight="bold")

    # Debug dump: print the exact values used to build the time-series curves.
    print(f"\nTime-series values for {var.upper()} ({units}):")
    for label, stats in stats_by_run.items():
        print(f"  Run: {label}")
        if not stats:
            print("    No statistics available")
            continue
        for rname in regions:
            d = stats.get(rname)
            if d is None or len(d["dates"]) == 0:
                continue
            display_name = "global" if rname == "open_ocean" else rname
            print(f"    Region: {display_name}")
            for dt, b, r in zip(d["dates"], d[f"{var}_bias"], d[f"{var}_rmse"]):
                print(f"      {dt:%Y-%m-%d}  bias={b:+.6f}  rmse={r:.6f}")

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

    for row, rname in enumerate(regions):
        ax_bias, ax_rmse = axes[row]
        plotted_bias = False
        plotted_rmse = False

        for idx, (label, stats) in enumerate(stats_by_run.items()):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            d = stats.get(rname)
            if d is None or len(d["dates"]) == 0:
                continue
            bias = d[f"{var}_bias"]
            rmse = d[f"{var}_rmse"]
            avg_b = np.nanmean(bias)
            avg_r = np.nanmean(rmse)
            ax_bias.plot(d["dates"], bias, marker + "-", color=color,
                         label=f"{label} ({avg_b:+.4f} {units})", markersize=3)
            ax_rmse.plot(d["dates"], rmse, marker + "-", color=color,
                         label=f"{label} ({avg_r:.4f} {units})", markersize=3)
            plotted_bias = True
            plotted_rmse = True

        ax_bias.axhline(0, color="black", lw=2, ls="-", zorder=1)
        display_name = "global" if rname == "open_ocean" else rname
        ax_bias.set_ylabel(f"Bias ({units})")
        ax_bias.set_title(f"{display_name} — Bias (bkg − OSTIA)")
        if plotted_bias:
            ax_bias.legend(fontsize=7)
        else:
            ax_bias.text(0.5, 0.5, "No valid data", transform=ax_bias.transAxes,
                         ha="center", va="center", fontsize=9)
        ax_bias.grid(True, alpha=0.3)
        _format_xaxis(ax_bias)

        ax_rmse.set_ylabel(f"RMSE ({units})")
        ax_rmse.set_title(f"{display_name} — RMSE")
        if plotted_rmse:
            ax_rmse.legend(fontsize=7)
        else:
            ax_rmse.text(0.5, 0.5, "No valid data", transform=ax_rmse.transAxes,
                         ha="center", va="center", fontsize=9)
        ax_rmse.grid(True, alpha=0.3)
        _format_xaxis(ax_rmse)

    outfile = os.path.join(out_dir, f"{var}_ts.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  Saved {outfile}")


# =========================================================================== #
#  Plotting – spatial maps
# =========================================================================== #
def plot_spatial_maps(spatial, exp_label, var, out_dir,
                      vmax_bias=None, vmax_rmse=None):
    """
    2-row figure: top = time-mean bias, bottom = RMSE.
    var = "sst" or "ice".
    Uses PlateCarree projection with land / coastlines.
    If vmax_bias / vmax_rmse are given, use them for shared colour scales.
    """
    cnt = spatial[f"{var}_count"]
    cnt_safe = np.where(cnt > 0, cnt, np.nan)

    mean_diff = spatial[f"{var}_diff_sum"] / cnt_safe
    rmse      = np.sqrt(spatial[f"{var}_diff_sq_sum"] / cnt_safe)

    units = "K" if var == "sst" else "fraction"
    title_var = "Foundation SST" if var == "sst" else "Ice Concentration"

    # Add cyclic point to avoid seam at 0/360°
    mean_diff_c, lon_c = add_cyclic_point(mean_diff, coord=TGT_LON)
    rmse_c, _          = add_cyclic_point(rmse,      coord=TGT_LON)
    LON_c, LAT_c = np.meshgrid(lon_c, TGT_LAT)

    proj = ccrs.PlateCarree(central_longitude=0)

    fig, (ax_bias, ax_rmse) = plt.subplots(
        2, 1, figsize=(14, 10), constrained_layout=True,
        subplot_kw={"projection": proj})
    fig.suptitle(f"{exp_label} — {title_var} vs OSTIA ({units})",
                 fontsize=15, fontweight="bold")

    # Bias
    if vmax_bias is None:
        finite = mean_diff[np.isfinite(mean_diff)]
        if len(finite) > 0:
            vmax_bias = float(np.percentile(np.abs(finite), 95))
            vmax_bias = max(vmax_bias, 0.01)
        else:
            vmax_bias = 1.0
    im = ax_bias.pcolormesh(LON_c, LAT_c, mean_diff_c,
                            transform=ccrs.PlateCarree(),
                            cmap="RdBu_r",
                            vmin=-vmax_bias, vmax=vmax_bias, shading="auto")
    ax_bias.contour(LON_c, LAT_c, mean_diff_c, levels=[0], colors="black",
                    linewidths=0.8, transform=ccrs.PlateCarree())
    ax_bias.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", lw=0.4, zorder=2)
    ax_bias.coastlines(lw=0.5, zorder=3)
    ax_bias.set_global()
    fig.colorbar(im, ax=ax_bias, shrink=0.7, label=f"Bias ({units})")
    ax_bias.set_title(f"Time-mean bias  (bkg − OSTIA)")

    # RMSE
    if vmax_rmse is None:
        finite_r = rmse[np.isfinite(rmse)]
        if len(finite_r) > 0:
            vmax_rmse = float(np.percentile(finite_r, 95))
            vmax_rmse = max(vmax_rmse, 0.01)
        else:
            vmax_rmse = 1.0
    im2 = ax_rmse.pcolormesh(LON_c, LAT_c, rmse_c,
                             transform=ccrs.PlateCarree(),
                             cmap="YlOrRd",
                             vmin=0, vmax=vmax_rmse, shading="auto")
    ax_rmse.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", lw=0.4, zorder=2)
    ax_rmse.coastlines(lw=0.5, zorder=3)
    ax_rmse.set_global()
    fig.colorbar(im2, ax=ax_rmse, shrink=0.7, label=f"RMSE ({units})")
    ax_rmse.set_title(f"Time-mean RMSE")

    tag = _safe_tag(exp_label)
    outfile = os.path.join(out_dir, f"{var}_maps_{tag}.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  Saved {outfile}")


# =========================================================================== #
#  Plotting – sea-ice extent time series
# =========================================================================== #
def plot_ice_extent(ice_extent, out_dir):
    """
    Plot sea-ice extent (million km²) time series for Arctic and Antarctic
    on the same axes.  Shows OSTIA and all available experiment backgrounds.
    """
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.set_title(f"Sea-Ice Extent "
                 f"(>{ICE_EXTENT_THRESH*100:.0f}% conc., common mask)",
                 fontsize=14, fontweight="bold")

    for hname in ("north", "south"):
        pole_label = "Arctic" if hname == "north" else "Antarctic"
        d = ice_extent.get(hname)
        if d is None or len(d["dates"]) == 0:
            continue

        dates = d["dates"]

        vals = d["ostia_km2"] / 1e6   # → million km²
        if not np.all(np.isnan(vals)):
            avg = np.nanmean(vals)
            ax.plot(dates, vals, "^-", color="tab:green", markersize=3,
                    label=f"{pole_label} OSTIA ({avg:.3f} M km²)")

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
        for idx, (label, vals) in enumerate(sorted(d["runs"].items())):
            vals = vals / 1e6   # → million km²
            if np.all(np.isnan(vals)):
                continue
            avg = np.nanmean(vals)
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax.plot(dates, vals, marker + "-", color=color, markersize=3,
                    label=f"{pole_label} {label} ({avg:.3f} M km²)")

    ax.set_ylabel("Sea-Ice Extent (million km²)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    _format_xaxis(ax)

    outfile = os.path.join(out_dir, "ice_extent.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  Saved {outfile}")


# =========================================================================== #
#  Plotting – Integrated Ice Edge Error (IIEE) time series
# =========================================================================== #
def plot_ice_edge_error(iiee, out_dir):
    """
    Plot IIEE time series — one figure per hemisphere (Arctic / Antarctic).

    Each figure has three rows:
            1. Total IIEE  (all available experiments)
      2. Overestimation area  (bkg ice where OSTIA has open water)
      3. Underestimation area (OSTIA ice where bkg has open water)
    """
    row_info = [
        ("iiee",  "Total IIEE"),
        ("over",  "Overestimation (bkg ice, OSTIA open)"),
        ("under", "Underestimation (OSTIA ice, bkg open)"),
    ]

    for hname in ("north", "south"):
        pole_label = "Arctic" if hname == "north" else "Antarctic"
        d = iiee.get(hname)
        if d is None or len(d["dates"]) == 0:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 constrained_layout=True, squeeze=False)
        fig.suptitle(f"{pole_label} — Integrated Ice Edge Error vs OSTIA "
                     f"(>{ICE_EXTENT_THRESH*100:.0f}% threshold)",
                     fontsize=14, fontweight="bold")

        dates = d["dates"]

        for row, (metric, ylabel_extra) in enumerate(row_info):
            ax = axes[row, 0]

            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
            markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
            for idx, (exp, series) in enumerate(sorted(d["runs"].items())):
                vals = series[metric] / 1e6   # → million km²
                if np.all(np.isnan(vals)):
                    continue
                avg = np.nanmean(vals)
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                ax.plot(dates, vals, marker + "-", color=color, markersize=3,
                        label=f"{exp} ({avg:.3f} M km²)")

            ax.set_ylabel(f"{ylabel_extra}\n(million km²)")
            ax.set_title(ylabel_extra)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            _format_xaxis(ax)

        outfile = os.path.join(out_dir, f"ice_edge_error_{hname}.png")
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print(f"  Saved {outfile}")


# =========================================================================== #
#  Plotting – polar stereographic ice spatial maps (per hemisphere)
# =========================================================================== #
def plot_ice_spatial_polar(spatial, exp_label, out_dir,
                           vmax_bias=None, vmax_rmse=None):
    """
    Plot ice bias & RMSE on polar stereographic projections,
    one figure per hemisphere (north / south).
    vmax_bias / vmax_rmse are dicts {"north": val, "south": val} for shared
    colour scales; if None, auto-compute from data.
    """
    cnt = spatial["ice_count"]
    cnt_safe = np.where(cnt > 0, cnt, np.nan)

    mean_diff = spatial["ice_diff_sum"] / cnt_safe
    rmse      = np.sqrt(spatial["ice_diff_sq_sum"] / cnt_safe)

    tag = _safe_tag(exp_label)

    # Add cyclic point to avoid seam at 0/360°
    mean_diff_c, lon_c = add_cyclic_point(mean_diff, coord=TGT_LON)
    rmse_c, _          = add_cyclic_point(rmse,      coord=TGT_LON)

    for hname in ("north", "south"):
        pole_label = "Arctic" if hname == "north" else "Antarctic"

        if hname == "north":
            proj = ccrs.NorthPolarStereo()
            lat_min, lat_max = 45, 90
            extent = [-180, 180, 45, 90]
        else:
            proj = ccrs.SouthPolarStereo()
            lat_min, lat_max = -90, -45
            extent = [-180, 180, -90, -45]

        # Mask to hemisphere
        lat_mask = (TGT_LAT >= lat_min) & (TGT_LAT <= lat_max)

        bias_sub = mean_diff_c[lat_mask, :]
        rmse_sub = rmse_c[lat_mask, :]
        LON_sub, LAT_sub = np.meshgrid(lon_c, TGT_LAT[lat_mask])

        fig, (ax_bias, ax_rmse) = plt.subplots(
            1, 2, figsize=(16, 7),
            subplot_kw={"projection": proj},
            constrained_layout=True)
        fig.suptitle(f"{exp_label} — {pole_label} Ice Concentration vs OSTIA",
                     fontsize=14, fontweight="bold")

        # --- Bias ---
        if vmax_bias is not None and hname in vmax_bias:
            vmax = vmax_bias[hname]
        else:
            finite = bias_sub[np.isfinite(bias_sub)]
            if len(finite) > 0:
                vmax = float(np.percentile(np.abs(finite), 95))
                vmax = max(vmax, 0.01)
            else:
                vmax = 0.1
        im = ax_bias.pcolormesh(LON_sub, LAT_sub, bias_sub,
                                transform=ccrs.PlateCarree(),
                                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                shading="auto")
        ax_bias.contour(LON_sub, LAT_sub, bias_sub, levels=[0],
                        colors="black", linewidths=0.8,
                        transform=ccrs.PlateCarree())
        ax_bias.set_extent(extent, crs=ccrs.PlateCarree())
        ax_bias.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", lw=0.4, zorder=2)
        ax_bias.coastlines(lw=0.5, zorder=3)
        gl = ax_bias.gridlines(draw_labels=False, lw=0.3, color="gray", alpha=0.5)
        ax_bias.set_title("Time-mean Bias (bkg − OSTIA)")
        fig.colorbar(im, ax=ax_bias, shrink=0.7, label="Bias (fraction)")

        # --- RMSE ---
        if vmax_rmse is not None and hname in vmax_rmse:
            vmax_r = vmax_rmse[hname]
        else:
            finite_r = rmse_sub[np.isfinite(rmse_sub)]
            if len(finite_r) > 0:
                vmax_r = float(np.percentile(finite_r, 95))
                vmax_r = max(vmax_r, 0.01)
            else:
                vmax_r = 0.1
        im2 = ax_rmse.pcolormesh(LON_sub, LAT_sub, rmse_sub,
                                 transform=ccrs.PlateCarree(),
                                 cmap="YlOrRd", vmin=0, vmax=vmax_r,
                                 shading="auto")
        ax_rmse.set_extent(extent, crs=ccrs.PlateCarree())
        ax_rmse.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5", lw=0.4, zorder=2)
        ax_rmse.coastlines(lw=0.5, zorder=3)
        gl2 = ax_rmse.gridlines(draw_labels=False, lw=0.3, color="gray", alpha=0.5)
        ax_rmse.set_title("Time-mean RMSE")
        fig.colorbar(im2, ax=ax_rmse, shrink=0.7, label="RMSE (fraction)")

        outfile = os.path.join(out_dir, f"ice_maps_{tag}_{hname}.png")
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print(f"  Saved {outfile}")


# =========================================================================== #
#  Helpers – compute shared colour-scale limits across experiments
# =========================================================================== #
def _shared_spatial_limits(spatial_list, var):
    """Return (vmax_bias, vmax_rmse) from multiple spatial dicts for *var*."""
    all_bias, all_rmse = [], []
    for sp in spatial_list:
        if sp is None:
            continue
        cnt = sp[f"{var}_count"]
        cnt_safe = np.where(cnt > 0, cnt, np.nan)
        md = sp[f"{var}_diff_sum"] / cnt_safe
        rm = np.sqrt(sp[f"{var}_diff_sq_sum"] / cnt_safe)
        fb = md[np.isfinite(md)]
        fr = rm[np.isfinite(rm)]
        if len(fb) > 0:
            all_bias.append(fb)
        if len(fr) > 0:
            all_rmse.append(fr)
    if all_bias:
        vmax_bias = float(np.percentile(np.abs(np.concatenate(all_bias)), 95))
        vmax_bias = max(vmax_bias, 0.01)
    else:
        vmax_bias = 1.0
    if all_rmse:
        vmax_rmse = float(np.percentile(np.concatenate(all_rmse), 95))
        vmax_rmse = max(vmax_rmse, 0.01)
    else:
        vmax_rmse = 1.0
    return vmax_bias, vmax_rmse


def _shared_polar_limits(spatial_list):
    """Return (vmax_bias_dict, vmax_rmse_dict) per hemisphere for ice."""
    vmax_bias = {}
    vmax_rmse = {}
    for hname in ("north", "south"):
        if hname == "north":
            lat_min, lat_max = 45, 90
        else:
            lat_min, lat_max = -90, -45
        lat_mask = (TGT_LAT >= lat_min) & (TGT_LAT <= lat_max)

        all_b, all_r = [], []
        for sp in spatial_list:
            if sp is None:
                continue
            cnt = sp["ice_count"]
            cnt_safe = np.where(cnt > 0, cnt, np.nan)
            md = (sp["ice_diff_sum"] / cnt_safe)[lat_mask, :]
            rm = np.sqrt((sp["ice_diff_sq_sum"] / cnt_safe))[lat_mask, :]
            fb = md[np.isfinite(md)]
            fr = rm[np.isfinite(rm)]
            if len(fb) > 0:
                all_b.append(fb)
            if len(fr) > 0:
                all_r.append(fr)
        if all_b:
            vmax_bias[hname] = max(
                float(np.percentile(np.abs(np.concatenate(all_b)), 95)), 0.01)
        else:
            vmax_bias[hname] = 0.1
        if all_r:
            vmax_rmse[hname] = max(
                float(np.percentile(np.concatenate(all_r), 95)), 0.01)
        else:
            vmax_rmse[hname] = 0.1
    return vmax_bias, vmax_rmse


# =========================================================================== #
#  Daily sea-ice edge maps (0.15 contour for OSTIA and all runs)
# =========================================================================== #
def plot_daily_ice_edges(days_by_run, ostia_dir, out_dir):
    """
    For each day, plot the 15 % sea-ice edge for OSTIA and all available
    runs on the same polar-stereographic map.  Produces two PNGs per day
    (arctic + antarctic) in <out_dir>/ice_edge/.
    """
    edge_dir = os.path.join(out_dir, "ice_edge")
    os.makedirs(edge_dir, exist_ok=True)

    # Add cyclic column so contour wraps around 0/360°
    lon_c = np.append(TGT_LON, TGT_LON[0] + 360.0)   # 721 points

    all_days = sorted(set().union(*(days.keys() for days in days_by_run.values())))

    for day_str in all_days:
        opath = ostia_path(ostia_dir, day_str)
        if not os.path.isfile(opath):
            continue

        # --- read ice fields --------------------------------------------------
        _, ice_ostia = read_ostia(opath)

        ice_by_run = {}
        for label, days in days_by_run.items():
            if day_str in days:
                _, ice_by_run[label] = daily_mean_bkg(days[day_str])

        # Add cyclic column to each field
        def _cyclic(field):
            return np.concatenate([field, field[:, :1]], axis=1)

        ice_ostia_c = _cyclic(np.where(np.isfinite(ice_ostia), ice_ostia, 0.0))
        ice_runs_c = {
            label: _cyclic(np.where(np.isfinite(icec), icec, 0.0))
            for label, icec in ice_by_run.items() if icec is not None
        }

        LON_c, LAT_c = np.meshgrid(lon_c, TGT_LAT)

        # --- one figure per hemisphere ----------------------------------------
        for hname in ("north", "south"):
            pole_label = "Arctic" if hname == "north" else "Antarctic"

            if hname == "north":
                proj = ccrs.NorthPolarStereo()
                extent = [-180, 180, 50, 90]
                lat_mask = TGT_LAT >= 50
            else:
                proj = ccrs.SouthPolarStereo()
                extent = [-180, 180, -90, -50]
                lat_mask = TGT_LAT <= -50

            LON_sub = LON_c[lat_mask, :]
            LAT_sub = LAT_c[lat_mask, :]

            fig, ax = plt.subplots(figsize=(8, 8),
                                   subplot_kw={"projection": proj},
                                   constrained_layout=True)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.gridlines(draw_labels=False, lw=0.3, color="gray", alpha=0.5)

            # OSTIA edge
            ax.contour(LON_sub, LAT_sub,
                       ice_ostia_c[lat_mask, :],
                       levels=[ICE_EXTENT_THRESH],
                       colors=["tab:green"], linewidths=2.0,
                       transform=ccrs.PlateCarree(), zorder=1)

            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
            for idx, (label, ice_c) in enumerate(sorted(ice_runs_c.items())):
                ax.contour(LON_sub, LAT_sub,
                           ice_c[lat_mask, :],
                           levels=[ICE_EXTENT_THRESH],
                           colors=[colors[idx % len(colors)]], linewidths=1.5,
                           transform=ccrs.PlateCarree(), zorder=1)

            # Land & coastlines on top to hide contours over land
            ax.add_feature(cfeature.LAND, facecolor="0.85",
                           edgecolor="0.5", lw=0.4, zorder=2)
            ax.coastlines(lw=0.5, zorder=3)

            # Legend via proxy artists
            handles = [Line2D([], [], color="tab:green", lw=2, label="OSTIA")]
            for idx, label in enumerate(sorted(ice_runs_c.keys())):
                handles.append(Line2D([], [], color=colors[idx % len(colors)], lw=1.5,
                                      label=f"{label} bkg"))
            ax.legend(handles=handles, loc="lower left", fontsize=9)

            ax.set_title(f"{pole_label} Sea-Ice Edge "
                         f"({ICE_EXTENT_THRESH*100:.0f}% conc.) — {day_str}",
                         fontsize=13, fontweight="bold")

            outfile = os.path.join(edge_dir,
                                   f"ice_edge_{hname}_{day_str}.png")
            fig.savefig(outfile, dpi=150)
            plt.close(fig)

        print(f"    {day_str}  ice edge maps saved")

    print(f"  Ice-edge maps → {os.path.realpath(edge_dir)}/")


# =========================================================================== #
#  Plotting – basin map
# =========================================================================== #
def plot_basin_map(basins, out_dir):
    """
    Plot a map showing the RECCAP2 ocean basins in different colours.
    Saves basin_map.png.
    """
    import matplotlib.colors as mcolors

    nlat, nlon = len(TGT_LAT), len(TGT_LON)
    basin_map = np.full((nlat, nlon), np.nan)

    # Assign each basin an integer label (order matters for legend)
    basin_names = ["open_ocean", "atlantic", "pacific",
                   "indian", "southern", "arctic"]
    display_names = ["global", "atlantic", "pacific",
                     "indian", "southern", "arctic"]
    colors = ["#a6cee3", "#1f78b4", "#b2df8a",
              "#33a02c", "#fb9a99", "#e31a1c"]

    for idx, bname in enumerate(basin_names):
        if bname in basins:
            basin_map[basins[bname]] = idx

    # Build a discrete colourmap
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, len(basin_names), 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Add cyclic point
    basin_map_c, lon_c = add_cyclic_point(basin_map, coord=TGT_LON)
    LON_c, LAT_c = np.meshgrid(lon_c, TGT_LAT)

    proj = ccrs.PlateCarree(central_longitude=0)
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True,
                           subplot_kw={"projection": proj})
    ax.set_title("RECCAP2 Ocean Basins", fontsize=15, fontweight="bold")

    im = ax.pcolormesh(LON_c, LAT_c, basin_map_c,
                       transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm, shading="auto")
    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5",
                   lw=0.4, zorder=2)
    ax.coastlines(lw=0.5, zorder=3)
    ax.set_global()

    # Legend via proxy patches
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    ax.legend(handles, display_names, loc="lower left", fontsize=9,
              framealpha=0.9, ncol=2)

    outfile = os.path.join(out_dir, "basin_map.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  Saved {outfile}")


# =========================================================================== #
#  Plotting – ops vs parallel difference maps (green = parallel better)
# =========================================================================== #
def plot_diff_ops_vs_par(ops_spatial, par_spatial, var, out_dir):
    """
    Plot the difference between ops and parallel |bias| and RMSE.

    For each metric the difference is  ops_value − parallel_value,
    so *positive* means parallel is better (green) and *negative* means
    ops is better (red).   Uses RdYlGn colourmap.
    """
    cnt_ops = ops_spatial[f"{var}_count"]
    cnt_par = par_spatial[f"{var}_count"]
    cnt_ops_safe = np.where(cnt_ops > 0, cnt_ops, np.nan)
    cnt_par_safe = np.where(cnt_par > 0, cnt_par, np.nan)

    bias_ops = ops_spatial[f"{var}_diff_sum"] / cnt_ops_safe
    bias_par = par_spatial[f"{var}_diff_sum"] / cnt_par_safe
    rmse_ops = np.sqrt(ops_spatial[f"{var}_diff_sq_sum"] / cnt_ops_safe)
    rmse_par = np.sqrt(par_spatial[f"{var}_diff_sq_sum"] / cnt_par_safe)

    # |bias| difference and RMSE difference: positive ⇒ ops worse
    abias_diff = np.abs(bias_ops) - np.abs(bias_par)
    rmse_diff  = rmse_ops - rmse_par

    units = "K" if var == "sst" else "fraction"
    title_var = "Foundation SST" if var == "sst" else "Ice Concentration"

    # Add cyclic point
    abias_diff_c, lon_c = add_cyclic_point(abias_diff, coord=TGT_LON)
    rmse_diff_c, _      = add_cyclic_point(rmse_diff,  coord=TGT_LON)
    LON_c, LAT_c = np.meshgrid(lon_c, TGT_LAT)

    # Symmetric colour limits
    finite_b = abias_diff[np.isfinite(abias_diff)]
    finite_r = rmse_diff[np.isfinite(rmse_diff)]
    vmax_b = max(float(np.percentile(np.abs(finite_b), 95)), 0.001) \
             if len(finite_b) > 0 else 0.1
    vmax_r = max(float(np.percentile(np.abs(finite_r), 95)), 0.001) \
             if len(finite_r) > 0 else 0.1

    proj = ccrs.PlateCarree(central_longitude=0)
    fig, (ax_bias, ax_rmse) = plt.subplots(
        2, 1, figsize=(14, 10), constrained_layout=True,
        subplot_kw={"projection": proj})
    fig.suptitle(f"{title_var} vs OSTIA — ops minus parallel ({units})\n"
                 f"green = parallel better  ·  red = ops better",
                 fontsize=14, fontweight="bold")

    # |Bias| difference
    im = ax_bias.pcolormesh(LON_c, LAT_c, abias_diff_c,
                            transform=ccrs.PlateCarree(),
                            cmap="RdYlGn",
                            vmin=-vmax_b, vmax=vmax_b, shading="auto")
    ax_bias.contour(LON_c, LAT_c, abias_diff_c, levels=[0], colors="black",
                    linewidths=0.8, transform=ccrs.PlateCarree())
    ax_bias.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5",
                        lw=0.4, zorder=2)
    ax_bias.coastlines(lw=0.5, zorder=3)
    ax_bias.set_global()
    fig.colorbar(im, ax=ax_bias, shrink=0.7, label=f"|Bias| diff ({units})")
    ax_bias.set_title("|Bias| ops − |Bias| parallel")

    # RMSE difference
    im2 = ax_rmse.pcolormesh(LON_c, LAT_c, rmse_diff_c,
                             transform=ccrs.PlateCarree(),
                             cmap="RdYlGn",
                             vmin=-vmax_r, vmax=vmax_r, shading="auto")
    ax_rmse.contour(LON_c, LAT_c, rmse_diff_c, levels=[0], colors="black",
                    linewidths=0.8, transform=ccrs.PlateCarree())
    ax_rmse.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.5",
                        lw=0.4, zorder=2)
    ax_rmse.coastlines(lw=0.5, zorder=3)
    ax_rmse.set_global()
    fig.colorbar(im2, ax=ax_rmse, shrink=0.7, label=f"RMSE diff ({units})")
    ax_rmse.set_title("RMSE ops − RMSE parallel")

    outfile = os.path.join(out_dir, f"{var}_diff_ops_vs_par.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  Saved {outfile}")


# =========================================================================== #
#  Main
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(
        description="Validate bkg SST & ice against OSTIA")
    parser.add_argument("--ops-dir", default=None,
                        help="Optional base directory for ops data")
    parser.add_argument("--exps-dir", nargs="+", default=["parallel"],
                        help="One or more experiment base directories "
                             "(default: parallel)")
    parser.add_argument("--ostia-dir", default="ostia_raw",
                        help="OSTIA raw data directory (default: ostia_raw)")
    parser.add_argument("--mask-file",
                        default="RECCAP2_region_masks_all.nc",
                        help="RECCAP2 basin mask file")
    parser.add_argument("--output-dir", default="plots_sfc",
                        help="Output directory for plots (default: plots_sfc)")
    parser.add_argument("--variables", default="both",
                        choices=["sst", "ice", "both"],
                        help="Which variables to process: sst, ice, or both "
                             "(default: both)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_specs = []
    used_labels = set()
    if args.ops_dir:
        run_specs.append(("ops", args.ops_dir))
        used_labels.add("ops")
    for idx, exp_dir in enumerate(args.exps_dir, start=1):
        base = os.path.basename(os.path.normpath(exp_dir))
        label = base if base not in ("", ".") else f"exp{idx}"
        label = _safe_tag(label)
        if label in used_labels:
            suffix = 2
            candidate = f"{label}_{suffix}"
            while candidate in used_labels:
                suffix += 1
                candidate = f"{label}_{suffix}"
            label = candidate
        used_labels.add(label)
        run_specs.append((label, exp_dir))

    if not run_specs:
        parser.error("Provide at least one directory with --exps-dir")

    do_sst = args.variables in ("sst", "both")
    do_ice = args.variables in ("ice", "both")

    # --- Load basin masks ---
    if do_sst:
        print(f"Loading basin masks from {args.mask_file} ...")
        basins = load_basin_masks(args.mask_file)
        print(f"  Basins: {sorted(basins.keys())}")

        # --- Basin map ---
        plot_basin_map(basins, args.output_dir)
    else:
        basins = {}

    # --- Discover background files and process runs ---
    days_by_run = {}
    sst_stats_by_run = {}
    ice_stats_by_run = {}
    spatial_by_run = {}

    for label, run_dir in run_specs:
        print(f"\nScanning {run_dir}/ for sfcf006 files ...")
        run_days = discover_sfc_files(run_dir)
        days_by_run[label] = run_days
        print(f"  Found {len(run_days)} days, "
              f"{sum(len(v) for v in run_days.values())} total files")

        if run_days:
            print(f"\n--- {label} vs OSTIA ---")
            sst_stats, ice_stats, spatial = compute_daily_stats(
                run_days, args.ostia_dir, basins)
        else:
            sst_stats, ice_stats, spatial = {}, {}, None

        sst_stats_by_run[label] = sst_stats
        ice_stats_by_run[label] = ice_stats
        spatial_by_run[label] = spatial

    # --- Sea-ice extent on common mask ---
    if do_ice:
        print("\n--- Sea-ice extent (common mask) ---")
        ice_extent = compute_ice_extent(days_by_run, args.ostia_dir)

        # --- Ice edge position error (IIEE) ---
        print("\n--- Integrated Ice Edge Error ---")
        iiee = compute_ice_edge_error(days_by_run, args.ostia_dir)

    # --- Plot time series ---
    print("\nGenerating plots ...")
    if do_sst:
        plot_timeseries(sst_stats_by_run, "sst", args.output_dir)
    if do_ice:
        plot_timeseries(ice_stats_by_run, "ice", args.output_dir)

    if do_sst:
        sst_vmax_b, sst_vmax_r = _shared_spatial_limits(
            list(spatial_by_run.values()), "sst")
        for label, spatial in spatial_by_run.items():
            if spatial is None:
                continue
            plot_spatial_maps(spatial, label, "sst", args.output_dir,
                              vmax_bias=sst_vmax_b, vmax_rmse=sst_vmax_r)

    if do_ice:
        polar_vmax_b, polar_vmax_r = _shared_polar_limits(
            list(spatial_by_run.values()))
        for label, spatial in spatial_by_run.items():
            if spatial is None:
                continue
            plot_ice_spatial_polar(spatial, label, args.output_dir,
                                   vmax_bias=polar_vmax_b,
                                   vmax_rmse=polar_vmax_r)

    # --- Plot sea-ice extent ---
    if do_ice:
        plot_ice_extent(ice_extent, args.output_dir)

        # --- Plot ice edge error ---
        plot_ice_edge_error(iiee, args.output_dir)

        # --- Daily ice-edge maps ---
        print("\n--- Daily ice-edge maps ---")
        plot_daily_ice_edges(days_by_run, args.ostia_dir, args.output_dir)

    print(f"\nAll done. Plots saved to: {os.path.realpath(args.output_dir)}/")


if __name__ == "__main__":
    main()
