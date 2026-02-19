#!/usr/bin/env python3

"""
Plot monthly WOA climatology for ocean temperature and salinity.

This script mirrors the plot style and output pattern used in plot_climatology.py,
but is scoped to Temp and Salt only.
"""

import os

if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process
import xarray as xr
import cartopy.crs as ccrs


climatology_dir = "/scratch3/NCEPDEV/da/Guillaume.Vernieres/data/woa"
grid_file = "/scratch3/NCEPDEV/da/common/validation/vrfy/gdas.t21z.ocngrid.nc"
layer_file = "/scratch3/NCEPDEV/da/common/validation/vrfy/soca_gridspec.bkgerr.nc"
vrfyout = "/scratch3/NCEPDEV/da/Andrew.Eichmann/vrfy/vrfy-dev/climatology_woa"
os.makedirs(vrfyout, exist_ok=True)

DEBUG_TRANSECTS = os.getenv("DEBUG_TRANSECTS", "0").lower() in {"1", "true", "yes", "y"}
DEBUG_MONTH = os.getenv("DEBUG_MONTH", "").strip()


if not os.path.exists(grid_file):
    raise FileNotFoundError(f"Grid file not found: {grid_file}")
if not os.path.exists(layer_file):
    raise FileNotFoundError(f"Layer file not found: {layer_file}")


print(f"Using grid file: {grid_file}")
print(f"Using layer file: {layer_file}")
print(f"Output directory: {vrfyout}")


variable_units = {
    "Temp": "deg C",
    "Salt": "psu",
}


projs = {
    "Global": ccrs.Mollweide(central_longitude=-150),
}


def _debug_log(message):
    if DEBUG_TRANSECTS:
        print(message)


def _to_360(lon_values):
    return np.mod(lon_values, 360.0)


def _circular_lon_diff(lon_values, lon_target):
    return np.abs((_to_360(lon_values) - _to_360(lon_target) + 180.0) % 360.0 - 180.0)


def _find_lat_index(grid_ds, lat_target):
    grid_lat = np.squeeze(np.array(grid_ds.lat))
    lat_axis = np.nanmean(grid_lat, axis=1)
    return int(np.nanargmin(np.abs(lat_axis - lat_target)))


def _find_lon_index(grid_ds, lon_target):
    grid_lon = np.squeeze(np.array(grid_ds.lon))
    lon_radians = np.deg2rad(_to_360(grid_lon))
    complex_lon = np.exp(1j * lon_radians)
    lon_col_mean = np.nanmean(complex_lon, axis=0)
    lon_axis = np.rad2deg(np.angle(lon_col_mean))
    lon_axis = _to_360(lon_axis)
    return int(np.nanargmin(_circular_lon_diff(lon_axis, lon_target)))


def _remap_index(source_index, source_size, target_size):
    if source_size <= 1 or target_size <= 1:
        return 0
    mapped = int(round(source_index * (target_size - 1) / (source_size - 1)))
    return max(0, min(target_size - 1, mapped))


def _get_depth_section(layer_ds, section_type, source_index, source_axis_size, target_size):
    if "h" not in layer_ds.variables:
        raise KeyError("Variable 'h' not found in layer file")

    h_field = np.squeeze(np.array(layer_ds["h"]))
    if h_field.ndim != 3:
        raise ValueError(f"Expected 3D h field, got shape {h_field.shape}")

    nlev, ny_layer, nx_layer = h_field.shape

    if section_type == "zonal":
        layer_index = _remap_index(source_index, source_axis_size, ny_layer)
        thickness = h_field[:, layer_index, :]
    elif section_type == "meridional":
        layer_index = _remap_index(source_index, source_axis_size, nx_layer)
        thickness = h_field[:, :, layer_index]
    else:
        raise ValueError(f"Unsupported section type: {section_type}")

    thickness[np.where(np.abs(thickness) > 10000.0)] = 0.0
    depth = np.cumsum(thickness, axis=0)

    if depth.shape[1] != target_size:
        from scipy.interpolate import interp1d

        src_size = depth.shape[1]
        src = np.linspace(0, 1, src_size)
        dst = np.linspace(0, 1, target_size)
        depth_interp = np.zeros((depth.shape[0], target_size))
        for level_index in range(depth.shape[0]):
            interpolator = interp1d(
                src,
                depth[level_index, :],
                kind="linear",
                fill_value="extrapolate",
            )
            depth_interp[level_index, :] = interpolator(dst)
        depth = depth_interp

    return depth


def _sanitize_slice_data(data_array, slice_data):
    clean = np.array(slice_data, dtype=float)

    fill_value = data_array.attrs.get("_FillValue", None)
    if fill_value is None:
        fill_value = data_array.attrs.get("missing_value", None)

    if fill_value is not None:
        clean[np.isclose(clean, float(fill_value), rtol=0.0, atol=0.0)] = np.nan

    clean[np.abs(clean) > 1.0e10] = np.nan
    clean[~np.isfinite(clean)] = np.nan
    return clean


def plotHorizontalSlice_clim(grid_ds, data_file, variable, bounds, colormap,
                             proj, output_root, exp, pdy, cyc, level=0):
    data = xr.open_dataset(data_file)

    dirname = os.path.join(output_root, variable)
    os.makedirs(dirname, exist_ok=True)

    unit = variable_units.get(variable, "unknown")

    var_da = data[variable]

    if "zaxis_1" in var_da.dims:
        slice_data = np.squeeze(var_da.isel(zaxis_1=level))
    else:
        slice_data = np.squeeze(var_da)

    slice_data = _sanitize_slice_data(var_da, slice_data)

    label_colorbar = f"{variable} ({unit}) Level {level}"
    figname = os.path.join(dirname, f"{variable}_Level_{level}_{proj}")
    title = f"{exp} {pdy} {cyc} {variable} Level {level}"

    slice_data = np.clip(slice_data, bounds[0], bounds[1])

    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": projs[proj]})

    pcolor_plot = ax.pcolormesh(
        np.squeeze(grid_ds.lon),
        np.squeeze(grid_ds.lat),
        slice_data,
        vmin=bounds[0],
        vmax=bounds[1],
        transform=ccrs.PlateCarree(),
        cmap=colormap,
        zorder=0,
    )

    cbar = fig.colorbar(pcolor_plot, ax=ax, shrink=0.75, orientation="horizontal")
    cbar.set_label(label_colorbar)

    contour_levels = np.linspace(bounds[0], bounds[1], 5)
    ax.contour(
        np.squeeze(grid_ds.lon),
        np.squeeze(grid_ds.lat),
        slice_data,
        levels=contour_levels,
        colors="black",
        linewidths=0.1,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    try:
        ax.coastlines()
    except Exception as error:
        print(f"Warning: could not add coastlines. {error}")

    ax.set_title(title)
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plotZonalSlice_clim(grid_ds, layer_ds, data_file, variable, bounds, colormap,
                        lat, output_root, exp, pdy, cyc, max_depth=700.0):
    data = xr.open_dataset(data_file)

    unit = variable_units.get(variable, "unknown")
    grid_lat = np.squeeze(np.array(grid_ds.lat))
    lat_index = _find_lat_index(grid_ds, lat)
    var_da = data[variable]
    slice_data = np.squeeze(np.array(var_da))[:, lat_index, :]
    slice_data = _sanitize_slice_data(var_da, slice_data)

    try:
        depth = _get_depth_section(
            layer_ds,
            "zonal",
            lat_index,
            grid_lat.shape[0],
            slice_data.shape[1],
        )
    except Exception as error:
        num_levels = slice_data.shape[0]
        depth = np.tile(np.linspace(0, max_depth, num_levels), (slice_data.shape[1], 1)).T
        _debug_log(
            f"WARNING zonal fallback depth for {pdy} {variable} lat={lat}: {error}"
        )

    lat_axis = np.nanmean(grid_lat, axis=1)
    selected_lat = float(lat_axis[lat_index])
    depth_max = float(np.nanmax(depth))
    valid_depth = np.where(np.isfinite(slice_data), depth, np.nan)
    max_valid_depth = float(np.nanmax(valid_depth)) if np.any(np.isfinite(valid_depth)) else np.nan
    _debug_log(
        f"DEBUG zonal {pdy} {variable}: target_lat={lat}, idx={lat_index}, "
        f"selected_lat={selected_lat:.2f}, max_cumsum_depth={depth_max:.1f}m, "
        f"max_valid_depth={max_valid_depth:.1f}m, plot_max_depth={max_depth:.1f}m"
    )

    slice_data = np.clip(slice_data, bounds[0], bounds[1])
    lons = np.squeeze(np.array(grid_ds.lon))[lat_index, :]
    x = np.tile(lons, (depth.shape[0], 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    contourf_plot = ax.contourf(
        x,
        -depth,
        slice_data,
        levels=np.linspace(bounds[0], bounds[1], 100),
        vmin=bounds[0],
        vmax=bounds[1],
        cmap=colormap,
    )

    contour_levels = np.linspace(bounds[0], bounds[1], 5)
    ax.contour(x, -depth, slice_data, levels=contour_levels, colors="black", linewidths=0.1)

    cbar = fig.colorbar(contourf_plot, ax=ax, shrink=0.5, orientation="horizontal")
    cbar.set_label(f"{variable} ({unit}) Lat {lat}")
    cbar.set_ticks(contour_levels)
    contourf_plot.set_clim(bounds[0], bounds[1])

    ax.set_ylim(-max_depth, 0)
    ax.set_xlim(lons.min(), lons.max())
    ax.set_title(f"{exp} {pdy} {cyc} {variable} lat {int(lat)}")

    dirname = os.path.join(output_root, variable)
    os.makedirs(dirname, exist_ok=True)
    figname = os.path.join(dirname, f"{variable}_zonal_lat_{int(lat)}_{int(max_depth)}m")
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plotMeridionalSlice_clim(grid_ds, layer_ds, data_file, variable, bounds, colormap,
                             lon, output_root, exp, pdy, cyc, max_depth=700.0):
    data = xr.open_dataset(data_file)

    unit = variable_units.get(variable, "unknown")
    grid_lon = np.squeeze(np.array(grid_ds.lon))
    lon_index = _find_lon_index(grid_ds, lon)
    var_da = data[variable]
    slice_data = np.squeeze(np.array(var_da))[:, :, lon_index]
    slice_data = _sanitize_slice_data(var_da, slice_data)

    try:
        depth = _get_depth_section(
            layer_ds,
            "meridional",
            lon_index,
            grid_lon.shape[1],
            slice_data.shape[1],
        )
    except Exception as error:
        num_levels = slice_data.shape[0]
        depth = np.tile(np.linspace(0, max_depth, num_levels), (slice_data.shape[1], 1)).T
        _debug_log(
            f"WARNING meridional fallback depth for {pdy} {variable} lon={lon}: {error}"
        )

    lon_radians = np.deg2rad(_to_360(grid_lon))
    complex_lon = np.exp(1j * lon_radians)
    lon_col_mean = np.nanmean(complex_lon, axis=0)
    selected_lon = float(np.rad2deg(np.angle(lon_col_mean[lon_index])))
    if selected_lon > 180.0:
        selected_lon -= 360.0
    depth_max = float(np.nanmax(depth))
    valid_depth = np.where(np.isfinite(slice_data), depth, np.nan)
    max_valid_depth = float(np.nanmax(valid_depth)) if np.any(np.isfinite(valid_depth)) else np.nan
    _debug_log(
        f"DEBUG meridional {pdy} {variable}: target_lon={lon}, idx={lon_index}, "
        f"selected_lon={selected_lon:.2f}, max_cumsum_depth={depth_max:.1f}m, "
        f"max_valid_depth={max_valid_depth:.1f}m, plot_max_depth={max_depth:.1f}m"
    )

    slice_data = np.clip(slice_data, bounds[0], bounds[1])
    lats = np.squeeze(grid_ds.lat)[:, lon_index]
    y = np.tile(lats, (depth.shape[0], 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    contourf_plot = ax.contourf(
        y,
        -depth,
        slice_data,
        levels=np.linspace(bounds[0], bounds[1], 100),
        vmin=bounds[0],
        vmax=bounds[1],
        cmap=colormap,
    )

    contour_levels = np.linspace(bounds[0], bounds[1], 5)
    ax.contour(y, -depth, slice_data, levels=contour_levels, colors="black", linewidths=0.1)

    cbar = fig.colorbar(contourf_plot, ax=ax, shrink=0.5, orientation="horizontal")
    cbar.set_label(f"{variable} ({unit}) Lon {lon}")
    cbar.set_ticks(contour_levels)
    contourf_plot.set_clim(bounds[0], bounds[1])

    ax.set_ylim(-max_depth, 0)
    ax.set_xlim(lats.min(), lats.max())
    ax.set_title(f"{exp} {pdy} {cyc} {variable} lon {int(lon)}")

    dirname = os.path.join(output_root, variable)
    os.makedirs(dirname, exist_ok=True)
    figname = os.path.join(dirname, f"{variable}_meridional_lon_{int(lon)}_{int(max_depth)}m")
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_temp_salt_climatology(config):
    try:
        grid_ds = xr.open_dataset(config["grid_file"])
        layer_ds = xr.open_dataset(config["layer_file"])

        for variable, bounds in config["variables_horiz"].items():
            plotHorizontalSlice_clim(
                grid_ds,
                config["data_file"],
                variable,
                bounds,
                config["colormap"],
                "Global",
                config["vrfyout"],
                config["exp"],
                config["PDY"],
                config["cyc"],
            )

        for variable, bounds in config["variables_zonal"].items():
            for lat in config["lats"]:
                for max_depth in config["max_depths"]:
                    plotZonalSlice_clim(
                        grid_ds,
                        layer_ds,
                        config["data_file"],
                        variable,
                        bounds,
                        config["colormap"],
                        lat,
                        config["vrfyout"],
                        config["exp"],
                        config["PDY"],
                        config["cyc"],
                        max_depth,
                    )

        for variable, bounds in config["variables_meridional"].items():
            for lon in config["lons"]:
                for max_depth in config["max_depths"]:
                    plotMeridionalSlice_clim(
                        grid_ds,
                        layer_ds,
                        config["data_file"],
                        variable,
                        bounds,
                        config["colormap"],
                        lon,
                        config["vrfyout"],
                        config["exp"],
                        config["PDY"],
                        config["cyc"],
                        max_depth,
                    )
    except Exception as error:
        print(f"Error plotting {config.get('PDY', 'unknown')}: {error}")
        import traceback
        traceback.print_exc()


configs = []

for month in range(1, 13):
    month_str = str(month).zfill(2)
    if DEBUG_MONTH and month_str != DEBUG_MONTH:
        continue

    ocean_file = os.path.join(climatology_dir, f"woa_on_mom6_layers_{month_str}.nc")

    if not os.path.exists(ocean_file):
        print(f"Warning: {ocean_file} not found, skipping month {month}")
        continue

    print(f"Configuring Temp/Salt WOA plots for month {month}")
    config_ocean = {
        "grid_file": grid_file,
        "layer_file": layer_file,
        "data_file": ocean_file,
        "PDY": f"WOA_Month_{month_str}",
        "cyc": "00",
        "exp": "WOA",
        "lats": np.arange(-60, 60, 10),
        "lons": np.arange(-280, 80, 30),
        "max_depths": [700.0, 5000.0],
        "variables_zonal": {
            "Temp": [-1.8, 34.0],
            "Salt": [32, 40],
        },
        "variables_meridional": {
            "Temp": [-1.8, 34.0],
            "Salt": [32, 40],
        },
        "variables_horiz": {
            "Temp": [-1.8, 34.0],
            "Salt": [32, 40],
        },
        "colormap": "nipy_spectral",
        "vrfyout": os.path.join(vrfyout, f"month_{month_str}", "ocean"),
    }
    configs.append(config_ocean)


print(f"\nTotal configurations to plot: {len(configs)}")
print("Starting plotting process...\n")

processes = []
for config in configs:
    process = Process(target=plot_temp_salt_climatology, args=(config,))
    process.start()
    processes.append(process)

for process in processes:
    process.join()


print("\n==============================================")
print("Temp/Salt climatology plotting completed!")
print(f"Plots saved to: {vrfyout}")
print("==============================================")
