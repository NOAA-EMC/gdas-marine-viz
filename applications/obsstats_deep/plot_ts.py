#!/usr/bin/env python3
"""
Generate density plots of OmB (observation minus background) and OmA (observation
minus analysis) versus depth for Argo profiles.

Usage:
    python plot_ts.py --start-date 20260320 --end-date 20260324 \
        --ioda-path './data/gdas.{date}/*/analysis/ocean/diags/insitu_temp_profile_argo.nc' \
        --variable temp --output-dir ./plots

The ioda-path must contain a ``{date}`` placeholder that will be expanded to
every date between start-date and end-date (inclusive).
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ---------------------------------------------------------------------------
# Variable name mapping (reuses the convention from plt_diags_maps)
# ---------------------------------------------------------------------------
VARIABLE_MAP = {
    'temp': ['waterTemperature', 'seaSurfaceTemperature'],
    'salt': ['salinity'],
}

# Default value-axis range for each variable (used in density plots)
VAL_RANGE = {
    'temp': (-1, 1),
    'salt': (-0.5, 0.5),
}

# Fixed color scale for spatial maps (symmetric for mean, max for RMSE)
MAP_VMAX = {
    'temp': 0.5,
    'salt': 0.3,
}

# Fixed x-axis limits for profile plots
PROFILE_XLIM = {
    'temp': 1.2,
    'salt': 0.5,
}

# Ocean basin codes (matches MetaData/oceanBasin in IODA files)
BASIN_NAMES = {
    1: 'atlantic',
    2: 'pacific',
    3: 'indian',
    4: 'arctic',
    5: 'southern',
}

# Spatial binning grid for map plots (4-degree)
LAT_BINS = np.arange(-90.0, 91.0, 4.0)
LON_BINS = np.arange(0.0, 361.0, 4.0)

# Depth layers for map plots: (min_depth, max_depth, label)
DEPTH_LAYERS = [
    (0, 10, '0-10m'),
    (0, 300, '0-300m'),
    (300, 2000, '300-2000m'),
]


# ---------------------------------------------------------------------------
# Data loading – lightweight, reads ombg + oman in a single pass
# ---------------------------------------------------------------------------
def load_argo_ioda(filepath, varname):
    """Load depth, ombg, and oman from a single IODA diag file.

    Returns a dict with 1-D arrays: depth, ombg, oman  (only QC-passed obs).
    Returns ``None`` when the file cannot be read or contains no valid data.
    """
    var_options = VARIABLE_MAP.get(varname)
    if var_options is None:
        raise ValueError(f"Unknown variable '{varname}'. "
                         f"Supported: {list(VARIABLE_MAP)}")

    try:
        ds = nc.Dataset(filepath, 'r')
    except Exception as e:
        print(f"  Warning: cannot open {filepath}: {e}")
        return None

    try:
        # Resolve the NetCDF variable name
        nc_var = None
        for v in var_options:
            if v in ds.groups['ombg'].variables:
                nc_var = v
                break
        if nc_var is None:
            print(f"  Warning: none of {var_options} found in {filepath}")
            ds.close()
            return None

        depth = np.asarray(ds.groups['MetaData'].variables['depth'][:]).flatten()
        ombg = np.asarray(ds.groups['ombg'].variables[nc_var][:]).flatten()

        # Lat/lon for map plots
        lat = np.asarray(ds.groups['MetaData'].variables['latitude'][:]).flatten()
        lon = np.asarray(ds.groups['MetaData'].variables['longitude'][:]).flatten()

        # Ocean basin flag
        if 'oceanBasin' in ds.groups['MetaData'].variables:
            ocean_basin = np.asarray(
                ds.groups['MetaData'].variables['oceanBasin'][:]).flatten()
        else:
            ocean_basin = np.full(len(depth), -1, dtype=np.int32)

        # oman may not exist (e.g. background-only runs)
        has_oman = 'oman' in ds.groups and nc_var in ds.groups['oman'].variables
        if has_oman:
            oman = np.asarray(ds.groups['oman'].variables[nc_var][:]).flatten()
        else:
            oman = None

        # QC mask – use EffectiveQC0 if available
        if 'EffectiveQC0' in ds.groups and nc_var in ds.groups['EffectiveQC0'].variables:
            qc = np.asarray(ds.groups['EffectiveQC0'].variables[nc_var][:]).flatten()
        else:
            qc = np.zeros_like(ombg, dtype=np.int8)

        ds.close()

        # Basic validity + QC filter
        fill_val = -3.368795e+38
        valid = (
            np.isfinite(depth)
            & (depth > fill_val)
            & np.isfinite(ombg)
            & (ombg > fill_val)
            & np.isfinite(lat)
            & np.isfinite(lon)
            & (qc == 0)
        )
        if oman is not None:
            valid &= np.isfinite(oman) & (oman > fill_val)

        if np.sum(valid) == 0:
            return None

        result = {
            'depth': depth[valid],
            'ombg': ombg[valid],
            'lat': lat[valid],
            'lon': np.mod(lon[valid], 360.0),  # ensure [0, 360)
            'ocean_basin': ocean_basin[valid],
        }
        if oman is not None:
            result['oman'] = oman[valid]

        return result

    except Exception as e:
        ds.close()
        print(f"  Warning: error reading {filepath}: {e}")
        return None


# ---------------------------------------------------------------------------
# Date expansion
# ---------------------------------------------------------------------------
def expand_dates(start_str, end_str):
    """Return list of date strings YYYYMMDD between start and end inclusive."""
    fmt = "%Y%m%d"
    start = datetime.strptime(start_str, fmt)
    end = datetime.strptime(end_str, fmt)
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime(fmt))
        d += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# Collect all files matching the pattern across dates
# ---------------------------------------------------------------------------
def collect_files(ioda_path_template, dates):
    """Expand the ``{date}`` placeholder for each date and glob."""
    all_files = []
    for date_str in dates:
        pattern = ioda_path_template.replace('{date}', date_str)
        matched = sorted(glob(pattern))
        all_files.extend(matched)
    return all_files


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_density_vs_depth(depth, values, title, xlabel, output_file,
                          depth_bins=None, val_range=None, vmin_pct=1, vmax_pct=99):
    """Create a 2-D histogram (density) plot of *values* versus *depth*.

    x-axis : values (e.g. OmB or OmA)
    y-axis : depth  (positive down, increasing downward)
    colour : observation count per bin
    """
    if depth_bins is None:
        depth_bins = np.arange(0, 2050, 10)

    # Value bins – use explicit range if provided, otherwise percentile-based
    if val_range is not None:
        val_bins = np.linspace(val_range[0], val_range[1], 200)
    else:
        val_lo = np.percentile(values, vmin_pct)
        val_hi = np.percentile(values, vmax_pct)
        val_abs = max(abs(val_lo), abs(val_hi))
        val_bins = np.linspace(-val_abs, val_abs, 200)

    # 2-D histogram
    H, xedges, yedges = np.histogram2d(values, depth, bins=[val_bins, depth_bins])

    # Normalize each depth row so that it sums to 1 (probability at each depth)
    row_sums = H.sum(axis=0, keepdims=True)  # sum over value bins for each depth
    row_sums[row_sums == 0] = 1  # avoid division by zero
    H = H / row_sums

    # Mask zero counts for a clean plot
    H = np.ma.masked_where(H == 0, H)

    fig, ax = plt.subplots(figsize=(8, 10))
    pcm = ax.pcolormesh(
        xedges, yedges, H.T,
        cmap='jet',
        shading='flat',
    )

    # Overlay contour lines on the density field
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    H_filled = np.where(H.T == 0, np.nan, H.T)  # use nan so contour skips empties
    ax.contour(xc, yc, H_filled, levels=6, colors='k', linewidths=0.6, alpha=0.5)

    ax.invert_yaxis()
    ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    fig.colorbar(pcm, ax=ax, label='Normalized density', shrink=0.6)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_file}")


def plot_profile_stats(depth, ombg, title, output_file,
                       oman=None, depth_bins=None, xlim=None):
    """Plot mean and RMSE profiles of OmB (and optionally OmA) binned by depth.

    When *oman* is provided, both OmB and OmA are shown on the same panels.
    *xlim* sets the x-axis limit: mean uses [-xlim, xlim], RMSE uses [0, xlim].
    """
    if depth_bins is None:
        depth_bins = np.arange(0, 2050, 10)

    bin_centres = 0.5 * (depth_bins[:-1] + depth_bins[1:])

    def _compute_profiles(values):
        bin_idx = np.digitize(values[0], depth_bins) - 1  # uses depth
        means = np.full(len(bin_centres), np.nan)
        rmses = np.full(len(bin_centres), np.nan)
        for i in range(len(bin_centres)):
            mask = bin_idx == i
            n = np.sum(mask)
            if n > 0:
                means[i] = np.mean(values[1][mask])
                rmses[i] = np.sqrt(np.mean(values[1][mask] ** 2))
        return means, rmses

    ombg_means, ombg_rmses = _compute_profiles((depth, ombg))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10), sharey=True)

    # Mean profile
    ax1.plot(ombg_means, bin_centres, 'b-o', markersize=2, linewidth=1.2,
             label='OmB')
    ax1.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax1.set_xlabel('Mean', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.invert_yaxis()
    ax1.set_ylim(2000, 0)
    ax1.grid(True, alpha=0.3)

    # RMSE profile
    ax2.plot(ombg_rmses, bin_centres, 'b-o', markersize=2, linewidth=1.2,
             label='OmB')
    ax2.set_xlabel('RMSE', fontsize=12)
    ax2.grid(True, alpha=0.3)

    if oman is not None:
        oman_means, oman_rmses = _compute_profiles((depth, oman))
        ax1.plot(oman_means, bin_centres, 'r-s', markersize=2, linewidth=1.2,
                 label='OmA')
        ax2.plot(oman_rmses, bin_centres, 'r-s', markersize=2, linewidth=1.2,
                 label='OmA')

    ax1.legend(fontsize=10)
    ax2.legend(fontsize=10)

    if xlim is not None:
        ax1.set_xlim(-xlim, xlim)
        ax2.set_xlim(0, xlim)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_file}")


def plot_binned_map(lat, lon, values, title, output_file, stat='mean',
                    fixed_vmax=None):
    """Plot spatially binned OmB statistics on a 1-degree global map.

    Parameters
    ----------
    lat, lon : 1-D arrays of observation positions (lon in [0, 360)).
    values : 1-D array of OmB (or OmA) values.
    title : plot title.
    output_file : path for the saved PNG.
    stat : 'mean' or 'rmse'.
    fixed_vmax : if provided, use this as the symmetric color scale limit.
    """
    nlat = len(LAT_BINS) - 1
    nlon = len(LON_BINS) - 1

    val_sum = np.zeros((nlat, nlon), dtype=np.float64)
    val_sq_sum = np.zeros((nlat, nlon), dtype=np.float64)
    count = np.zeros((nlat, nlon), dtype=np.float64)

    li = np.clip(np.digitize(lat, LAT_BINS) - 1, 0, nlat - 1)
    lo = np.clip(np.digitize(lon, LON_BINS) - 1, 0, nlon - 1)

    np.add.at(val_sum, (li, lo), values)
    np.add.at(val_sq_sum, (li, lo), values ** 2)
    np.add.at(count, (li, lo), 1.0)

    count_safe = np.where(count > 0, count, np.nan)
    if stat == 'rmse':
        field = np.sqrt(val_sq_sum / count_safe)
    else:
        field = val_sum / count_safe

    lon_c = 0.5 * (LON_BINS[:-1] + LON_BINS[1:])
    lat_c = 0.5 * (LAT_BINS[:-1] + LAT_BINS[1:])
    lon2d, lat2d = np.meshgrid(lon_c, lat_c)

    finite = field[np.isfinite(field)]
    if finite.size == 0:
        print(f"  Skipping map (no data): {output_file}")
        return

    if stat == 'rmse':
        vmax = fixed_vmax if fixed_vmax is not None else float(np.nanpercentile(finite, 95))
        vmin = 0.0
        cmap = 'hot_r'
        cbar_label = 'RMSE'
    else:
        vmax = fixed_vmax if fixed_vmax is not None else max(float(np.nanpercentile(np.abs(finite), 95)), 0.01)
        vmin = -vmax
        cmap = 'RdBu_r'
        cbar_label = 'Mean'

    fig, ax = plt.subplots(
        figsize=(14, 6),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True,
    )
    ax.set_title(title, fontsize=13, fontweight='bold')

    im = ax.pcolormesh(
        lon2d, lat2d, field,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading='auto',
    )
    if stat == 'mean':
        ax.contour(
            lon2d, lat2d, field, levels=[0.0],
            colors='black', linewidths=0.8,
            transform=ccrs.PlateCarree(),
        )
    ax.add_feature(cfeature.LAND, facecolor='0.85', edgecolor='0.5', linewidth=0.4)
    ax.coastlines(linewidth=0.5)
    ax.set_global()

    fig.colorbar(im, ax=ax, shrink=0.75, label=cbar_label)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Density plots of OmB / OmA vs depth for Argo profiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_ts.py --start-date 20260320 --end-date 20260324 \\
      --ioda-path './data/gdas.{date}/*/analysis/ocean/diags/insitu_temp_profile_argo.nc'

  python plot_ts.py --start-date 20260320 --end-date 20260324 \\
      --ioda-path './data/gdas.{date}/00/analysis/ocean/diags/insitu_salt_profile_argo.nc' \\
      --variable salt --output-dir ./salt_plots
        """,
    )
    parser.add_argument('--start-date', required=True,
                        help='Start date YYYYMMDD')
    parser.add_argument('--end-date', required=True,
                        help='End date YYYYMMDD')
    parser.add_argument('--ioda-path', required=True,
                        help='Path template with {date} placeholder, '
                             'e.g. "./data/gdas.{date}/*/analysis/ocean/diags/insitu_temp_profile_argo.nc"')
    parser.add_argument('--variable', default='temp', choices=list(VARIABLE_MAP),
                        help='Variable to plot (default: temp)')
    parser.add_argument('--output-dir', default='.',
                        help='Directory for output plots (default: cwd)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Collect files ----
    dates = expand_dates(args.start_date, args.end_date)
    print(f"Date range: {args.start_date} – {args.end_date}  ({len(dates)} days)")

    files = collect_files(args.ioda_path, dates)
    print(f"Found {len(files)} IODA files")

    if not files:
        print("No files found – check --ioda-path and date range.")
        sys.exit(1)

    # ---- Load data ----
    all_depth = []
    all_ombg = []
    all_oman = []
    all_basin = []
    all_lat = []
    all_lon = []

    for f in files:
        print(f"  Loading {os.path.basename(f)}")
        data = load_argo_ioda(f, args.variable)
        if data is None:
            continue
        all_depth.append(data['depth'])
        all_ombg.append(data['ombg'])
        all_basin.append(data['ocean_basin'])
        all_lat.append(data['lat'])
        all_lon.append(data['lon'])
        if 'oman' in data:
            all_oman.append(data['oman'])

    if not all_depth:
        print("No valid observations loaded.")
        sys.exit(1)

    depth = np.concatenate(all_depth)
    ombg = np.concatenate(all_ombg)
    basin = np.concatenate(all_basin)
    lat = np.concatenate(all_lat)
    lon = np.concatenate(all_lon)
    has_oman = len(all_oman) == len(all_depth)
    if has_oman:
        oman = np.concatenate(all_oman)

    n_obs = len(depth)
    print(f"\nTotal observations: {n_obs:,}")
    print(f"Depth range: {depth.min():.1f} – {depth.max():.1f} m")

    var_label = args.variable.upper()
    period = f"{args.start_date}–{args.end_date}"
    val_range = VAL_RANGE.get(args.variable)
    profile_xlim = PROFILE_XLIM.get(args.variable)

    # ---- Generate plots per basin + global ----
    # Build list of (basin_code, basin_name) pairs; None = global (all basins)
    basins_to_plot = [(None, 'global')]
    for code, name in sorted(BASIN_NAMES.items()):
        if np.any(basin == code):
            basins_to_plot.append((code, name))

    for basin_code, basin_name in basins_to_plot:
        if basin_code is None:
            mask = np.ones(len(depth), dtype=bool)
            basin_label = 'Global'
        else:
            mask = basin == basin_code
            basin_label = basin_name.capitalize()

        b_depth = depth[mask]
        b_ombg = ombg[mask]
        b_oman = oman[mask] if has_oman else None
        b_nobs = len(b_depth)

        if b_nobs == 0:
            print(f"\n  Skipping {basin_label} – no observations.")
            continue

        # Output subdirectory named after the basin
        basin_dir = os.path.join(args.output_dir, basin_name)
        os.makedirs(basin_dir, exist_ok=True)

        print(f"\n  {basin_label}: {b_nobs:,} observations")

        # ---- OmB density plot ----
        plot_density_vs_depth(
            b_depth, b_ombg,
            title=f"OmB density – Argo {var_label} – {basin_label}\n{period}  (n={b_nobs:,})",
            xlabel=f"OmB ({var_label})",
            output_file=os.path.join(basin_dir,
                                     f"argo_{args.variable}_ombg_density_{args.start_date}_{args.end_date}.png"),
            val_range=val_range,
        )

        # ---- Profile stats (OmB + OmA combined) ----
        plot_profile_stats(
            b_depth, b_ombg,
            title=f"OmB/OmA profile – Argo {var_label} – {basin_label}\n{period}  (n={b_nobs:,})",
            output_file=os.path.join(basin_dir,
                                     f"argo_{args.variable}_profile_{args.start_date}_{args.end_date}.png"),
            oman=b_oman,
            xlim=profile_xlim,
        )

        # ---- OmA density plot ----
        if has_oman:
            plot_density_vs_depth(
                b_depth, b_oman,
                title=f"OmA density – Argo {var_label} – {basin_label}\n{period}  (n={b_nobs:,})",
                xlabel=f"OmA ({var_label})",
                output_file=os.path.join(basin_dir,
                                         f"argo_{args.variable}_oman_density_{args.start_date}_{args.end_date}.png"),
                val_range=val_range,
            )

    # ---- Global binned OmB/OmA maps per depth layer ----
    global_dir = os.path.join(args.output_dir, 'global')
    os.makedirs(global_dir, exist_ok=True)
    map_vmax = MAP_VMAX.get(args.variable)

    for d_min, d_max, d_label in DEPTH_LAYERS:
        layer_mask = (depth >= d_min) & (depth < d_max)
        n_layer = int(np.sum(layer_mask))
        if n_layer == 0:
            continue

        d_tag = d_label.replace('-', '_').replace('m', '')

        # Mean OmB map
        plot_binned_map(
            lat[layer_mask], lon[layer_mask], ombg[layer_mask],
            title=(f"Mean OmB – Argo {var_label} [{d_label}]\n"
                   f"{period}  (n={n_layer:,})"),
            output_file=os.path.join(
                global_dir,
                f"argo_{args.variable}_ombg_map_mean_{d_tag}_{args.start_date}_{args.end_date}.png"),
            stat='mean', fixed_vmax=map_vmax,
        )

        # RMSE OmB map
        plot_binned_map(
            lat[layer_mask], lon[layer_mask], ombg[layer_mask],
            title=(f"RMSE OmB – Argo {var_label} [{d_label}]\n"
                   f"{period}  (n={n_layer:,})"),
            output_file=os.path.join(
                global_dir,
                f"argo_{args.variable}_ombg_map_rmse_{d_tag}_{args.start_date}_{args.end_date}.png"),
            stat='rmse', fixed_vmax=map_vmax,
        )

        # OmA maps (if available)
        if has_oman:
            # Mean OmA map
            plot_binned_map(
                lat[layer_mask], lon[layer_mask], oman[layer_mask],
                title=(f"Mean OmA – Argo {var_label} [{d_label}]\n"
                       f"{period}  (n={n_layer:,})"),
                output_file=os.path.join(
                    global_dir,
                    f"argo_{args.variable}_oman_map_mean_{d_tag}_{args.start_date}_{args.end_date}.png"),
                stat='mean', fixed_vmax=map_vmax,
            )

            # RMSE OmA map
            plot_binned_map(
                lat[layer_mask], lon[layer_mask], oman[layer_mask],
                title=(f"RMSE OmA – Argo {var_label} [{d_label}]\n"
                       f"{period}  (n={n_layer:,})"),
                output_file=os.path.join(
                    global_dir,
                    f"argo_{args.variable}_oman_map_rmse_{d_tag}_{args.start_date}_{args.end_date}.png"),
                stat='rmse', fixed_vmax=map_vmax,
            )

    print("\nDone.")


if __name__ == '__main__':
    main()
