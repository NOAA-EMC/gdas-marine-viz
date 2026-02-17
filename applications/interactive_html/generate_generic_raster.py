#!/usr/bin/env python3
"""
Generic function to generate static raster images from IODA-format observation data
"""
import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d


def lonlat_to_web_mercator(lon, lat):
    """Convert lon/lat to Web Mercator (EPSG:3857) coordinates"""
    x = lon * 20037508.34 / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) * (20037508.34 / np.pi)
    return x, y


def web_mercator_to_lonlat(x, y):
    """Convert Web Mercator coordinates back to lon/lat"""
    lon = x * 180.0 / 20037508.34
    lat = (np.arctan(np.exp(y * np.pi / 20037508.34)) * 360.0 / np.pi) - 90.0
    return lon, lat


def generate_observation_raster(nc_file,
                                output_file,
                                variable_name,
                                vmin=-1.0,
                                vmax=1.0,
                                resolution=0.5):
    """
    Generate a static raster image from IODA-format observation data
    Creates a GLOBAL grid and plots observations on it

    Parameters:
    -----------
    nc_file : str
        Path to NetCDF file in IODA format
    output_file : str
        Path for output PNG file
    variable_name : str
        Name of variable in ombg group
    vmin, vmax : float
        Color scale limits
    resolution : float
        Grid resolution in degrees

    Returns:
    --------
    dict with 'bounds' and 'n_obs' keys, or None on error
    """

    if not os.path.exists(nc_file):
        print(f"Error: {nc_file} not found")
        return None

    try:
        dataset = nc.Dataset(nc_file, 'r')

        # Read data from IODA format
        lons = dataset.groups['MetaData'].variables['longitude'][:]
        lats = dataset.groups['MetaData'].variables['latitude'][:]
        ombg = dataset.groups['ombg'].variables[variable_name][:]

        dataset.close()

        # Convert masked arrays to regular arrays, filling masked values with NaN
        lons = np.ma.filled(lons, np.nan)
        lats = np.ma.filled(lats, np.nan)
        ombg = np.ma.filled(ombg, np.nan)

        # Filter valid data
        valid = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(ombg)

        lons = lons[valid]
        lats = lats[valid]
        ombg = ombg[valid]

        if len(lons) == 0:
            print("No valid observations found")
            return None

        print(f"Read {len(lons)} valid observations")
        print(f"  Lon range: {lons.min():.2f} to {lons.max():.2f}")
        print(f"  Lat range: {lats.min():.2f} to {lats.max():.2f}")
        print(f"  OMB-G range: {ombg.min():.4f} to {ombg.max():.4f}")

        # Define GLOBAL extent in Web Mercator
        # Web Mercator limits: ~85.05°N to ~85.05°S
        global_lat_min, global_lat_max = -85.0, 85.0
        global_lon_min, global_lon_max = -180.0, 180.0

        # Convert global bounds to Web Mercator
        x_min, y_min = lonlat_to_web_mercator(global_lon_min, global_lat_min)
        x_max, y_max = lonlat_to_web_mercator(global_lon_max, global_lat_max)

        # Convert observation coordinates to Web Mercator
        x_obs, y_obs = lonlat_to_web_mercator(lons, lats)

        # Calculate bins for global grid
        resolution_m = resolution * 111320  # degrees to meters at equator
        n_x_bins = int((x_max - x_min) / resolution_m)
        n_y_bins = int((y_max - y_min) / resolution_m)

        print(f"Creating GLOBAL {n_x_bins} x {n_y_bins} grid")

        # Bin the data in Web Mercator space
        ret = binned_statistic_2d(
            x_obs, y_obs, ombg,
            statistic='mean',
            bins=[n_x_bins, n_y_bins],
            range=[[x_min, x_max], [y_min, y_max]]
        )

        grid = ret.statistic.T

        # Mask invalid values for transparency
        grid = np.ma.masked_invalid(grid)

        print(f"Grid contains {np.sum(~grid.mask)} valid cells")

        # Create figure matching grid dimensions
        fig_width_inches = n_x_bins / 100
        fig_height_inches = n_y_bins / 100

        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches),
                         dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Plot in Web Mercator space
        ax.imshow(grid, cmap='RdBu_r', origin='lower',
                  extent=[x_min, x_max, y_min, y_max],
                  vmin=vmin, vmax=vmax, interpolation='nearest',
                  aspect='auto',
                  alpha=1.0)

        # Save with exact dimensions and transparent background
        fig.savefig(output_file, dpi=100, transparent=True, format='png')
        plt.close()

        # Bounds are now global, not data-specific
        print(f"Generated {output_file}")
        print(f"  Global bounds (lat/lon): [{global_lat_min}, {global_lon_min}] "
              f"to [{global_lat_max}, {global_lon_max}]")

        # Return global bounds for Leaflet imageOverlay (in lat/lon)
        return {
            'bounds': [[global_lat_min, global_lon_min], [global_lat_max, global_lon_max]],
            'n_obs': len(lons)
        }

    except Exception as e:
        print(f"Error processing {nc_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # Test with SST
    print("Testing with SST data:")
    result = generate_observation_raster(
        'obs_profiles/sst_abi_g16_l3c.2021070618.nc',
        'test_sst_raster.png',
        'seaSurfaceTemperature',
        vmin=-1.0,
        vmax=1.0,
        resolution=0.5
    )
    if result:
        print(f"Success! {result['n_obs']} observations")
        print(f"Bounds: {result['bounds']}")

    print("\n" + "=" * 60)
    print("Testing with sea ice data:")
    result = generate_observation_raster(
        'obs_profiles/icec_amsr2_north.2021070618.nc',
        'test_seaice_raster.png',
        'seaIceFraction',
        vmin=-0.2,
        vmax=0.2,
        resolution=0.5
    )
    if result:
        print(f"Success! {result['n_obs']} observations")
        print(f"Bounds: {result['bounds']}")
