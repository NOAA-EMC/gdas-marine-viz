#!/usr/bin/env python3

"""
Plot monthly climatology data for ocean (Salt, Temp) and sea ice (hs_h) fields.
This script creates geographic plots of surface values, and meridional and zonal slices.
"""

import os
# Ensure non-interactive backend on headless systems (HPC)
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process
import xarray as xr
import cartopy.crs as ccrs

# Directory containing climatology data
climatology_dir = '/scratch3/NCEPDEV/da/Guillaume.Vernieres/data/woa'

# Output directory for plots
vrfyout = '/scratch3/NCEPDEV/da/Andrew.Eichmann/vrfy/vrfy-dev/climatology_woa'
os.makedirs(vrfyout, exist_ok=True)

# Grid file and layer file paths based on the system's hostname
hpcname = os.getenv('HPCname', 'ursa')  # default to ursa
if hpcname.startswith("hera"):
    grid_file = '/scratch1/NCEPDEV/da/common/validation/vrfy/soca_gridspec.bkgerr.nc'
    layer_file = '/scratch1/NCEPDEV/da/common/validation/vrfy/soca_gridspec.bkgerr.nc'
elif hpcname in ["ursa"]:
    grid_file = '/scratch3/NCEPDEV/da/common/validation/vrfy/gdas.t21z.ocngrid.nc'
    layer_file = '/scratch3/NCEPDEV/da/common/validation/vrfy/soca_gridspec.bkgerr.nc'
elif hpcname in ["hercules", "orion"]:
    grid_file = '/work/noaa/da/marineda/validation/vrfy/soca_gridspec.bkgerr.nc'
    layer_file = '/work/noaa/da/marineda/validation/vrfy/soca_gridspec.bkgerr.nc'
else:
    # Default fallback
    grid_file = '/scratch3/NCEPDEV/da/common/validation/vrfy/gdas.t21z.ocngrid.nc'
    layer_file = '/scratch3/NCEPDEV/da/common/validation/vrfy/soca_gridspec.bkgerr.nc'

# Check if the grid file exists
if not os.path.exists(grid_file):
    print(f"Warning: Grid file {grid_file} not found. Using default.")
    grid_file = '/scratch3/NCEPDEV/da/common/validation/vrfy/gdas.t21z.ocngrid.nc'

# Check if the layer file exists
if not os.path.exists(layer_file):
    print(f"Warning: Layer file {layer_file} not found.")

print(f"Using grid file: {grid_file}")
print(f"Using layer file: {layer_file}")
print(f"Output directory: {vrfyout}")

# Map variable names to their units
variable_units = {
    'Temp': 'deg C',
    'Salt': 'psu',
    'hs_h': 'meter'
}

projs = {'North': ccrs.NorthPolarStereo(),
         'South': ccrs.SouthPolarStereo(),
         'Global': ccrs.Mollweide(central_longitude=-150)}


def plotHorizontalSlice_clim(grid_file, data_file, variable, bounds, colormap, 
                              proj, vrfyout, exp, PDY, cyc, level=0):
    """
    Contourf of a horizontal slice of a climatology field
    """
    grid = xr.open_dataset(grid_file)
    data = xr.open_dataset(data_file)

    dirname = os.path.join(vrfyout, variable)
    os.makedirs(dirname, exist_ok=True)

    unit = variable_units.get(variable, 'unknown')

    if variable in ['Temp', 'Salt']:
        # Ocean variables - select the level
        if 'zaxis_1' in data[variable].dims:
            slice_data = np.squeeze(data[variable].isel(zaxis_1=level))
        else:
            slice_data = np.squeeze(data[variable])
        label_colorbar = f"{variable} ({unit}) Level {level}"
        figname = os.path.join(dirname, variable + '_Level_' + str(level) + '_' + proj)
        title = f"{exp} {PDY} {cyc} {variable} Level {level}"
    else:
        slice_data = np.squeeze(data[variable])
        label_colorbar = f"{variable} ({unit})"
        figname = os.path.join(dirname, variable + '_' + proj)
        title = f"{exp} {PDY} {cyc} {variable}"

    slice_data = np.clip(slice_data, bounds[0], bounds[1])

    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection': projs[proj]})

    # Use pcolor to plot the data
    pcolor_plot = ax.pcolormesh(np.squeeze(grid.lon),
                                np.squeeze(grid.lat),
                                slice_data,
                                vmin=bounds[0], vmax=bounds[1],
                                transform=ccrs.PlateCarree(),
                                cmap=colormap,
                                zorder=0)

    # Add colorbar for filled contours
    cbar = fig.colorbar(pcolor_plot, ax=ax, shrink=0.75, orientation='horizontal')
    cbar.set_label(label_colorbar)

    # Add contour lines with specified linewidths
    contour_levels = np.linspace(bounds[0], bounds[1], 5)
    ax.contour(np.squeeze(grid.lon),
               np.squeeze(grid.lat),
               slice_data,
               levels=contour_levels,
               colors='black',
               linewidths=0.1,
               transform=ccrs.PlateCarree(),
               zorder=2)

    try:
        ax.coastlines()
    except Exception as e:
        print(f"Warning: could not add coastlines. {e}")
    ax.set_title(title)
    if proj == 'South':
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    if proj == 'North':
        ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plotZonalSlice_clim(grid_file, data_file, variable, bounds, colormap, 
                        lat, vrfyout, exp, PDY, cyc, max_depth=700.0, layer_file=None):
    """
    Contourf of a zonal slice of a climatology field
    """
    grid = xr.open_dataset(grid_file)
    data = xr.open_dataset(data_file)
    
    unit = variable_units.get(variable, 'unknown')
    lat_index = np.argmin(np.array(np.abs(np.squeeze(grid.lat)[:, 0] - lat)))

    # Extract the slice data
    slice_data = np.squeeze(np.array(data[variable]))[:, lat_index, :]
    
    # Try to get depth information from the layer file (similar to soca_vrfy.py)
    if layer_file is None:
        layer_file = grid_file
    
    try:
        layer = xr.open_dataset(layer_file)
        if 'h' in layer.variables:
            depth = np.squeeze(np.array(layer['h']))[:, lat_index, :]
            depth[np.where(np.abs(depth) > 10000.0)] = 0.0
            depth = np.cumsum(depth, axis=0)
            
            # Interpolate depth to match the horizontal resolution of slice_data
            if depth.shape[1] != slice_data.shape[1]:
                from scipy.interpolate import interp1d
                # Get the longitude dimension sizes
                n_lons_layer = depth.shape[1]
                n_lons_data = slice_data.shape[1]
                # Create interpolation indices
                x_layer = np.linspace(0, 1, n_lons_layer)
                x_data = np.linspace(0, 1, n_lons_data)
                # Interpolate each vertical level
                depth_interp = np.zeros((depth.shape[0], n_lons_data))
                for k in range(depth.shape[0]):
                    f = interp1d(x_layer, depth[k, :], kind='linear', fill_value='extrapolate')
                    depth_interp[k, :] = f(x_data)
                depth = depth_interp
        else:
            # Create uniform depth levels if no h variable
            num_levels = slice_data.shape[0]
            depth = np.tile(np.linspace(0, max_depth, num_levels), 
                          (slice_data.shape[1], 1)).T
    except:
        # Create uniform depth levels
        num_levels = slice_data.shape[0]
        depth = np.tile(np.linspace(0, max_depth, num_levels), 
                      (slice_data.shape[1], 1)).T
    
    slice_data = np.clip(slice_data, bounds[0], bounds[1])
    lons = grid.lon[:, lat_index]
    x = np.tile(np.squeeze(lons), (np.shape(depth)[0], 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the filled contours
    
    contourf_plot = ax.contourf(x, -depth, slice_data,
                                levels=np.linspace(bounds[0], bounds[1], 100),
                                vmin=bounds[0], vmax=bounds[1],
                                cmap=colormap)

    # Add contour lines
    contour_levels = np.linspace(bounds[0], bounds[1], 5)
    ax.contour(x, -depth, slice_data,
               levels=contour_levels,
               colors='black',
               linewidths=0.1)

    # Add colorbar
    cbar = fig.colorbar(contourf_plot, ax=ax, shrink=0.5, orientation='horizontal')
    cbar.set_label(f"{variable} ({unit}) Lat {lat}")
    cbar.set_ticks(contour_levels)
    contourf_plot.set_clim(bounds[0], bounds[1])

    ax.set_ylim(-max_depth, 0)
    ax.set_xlim(lons.min(), lons.max())
    title = f"{exp} {PDY} {cyc} {variable} lat {int(lat)}"
    ax.set_title(title)
    
    dirname = os.path.join(vrfyout, variable)
    os.makedirs(dirname, exist_ok=True)
    figname = os.path.join(dirname,
        variable + '_zonal_lat_' + str(int(lat)) + '_' + str(int(max_depth)) + 'm')
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plotMeridionalSlice_clim(grid_file, data_file, variable, bounds, colormap, 
                             lon, vrfyout, exp, PDY, cyc, max_depth=700.0, layer_file=None):
    """
    Contourf of a meridional slice of a climatology field
    """
    grid = xr.open_dataset(grid_file)
    data = xr.open_dataset(data_file)
    
    unit = variable_units.get(variable, 'unknown')
    lon_index = np.argmin(np.array(np.abs(np.squeeze(grid.lon)[0, :] - lon)))
    
    # Extract the slice data
    slice_data = np.squeeze(np.array(data[variable]))[:, :, lon_index]
    
    # Try to get depth information from the layer file (similar to soca_vrfy.py)
    if layer_file is None:
        layer_file = grid_file
    
    try:
        layer = xr.open_dataset(layer_file)
        if 'h' in layer.variables:
            depth = np.squeeze(np.array(layer['h']))[:, :, lon_index]
            depth[np.where(np.abs(depth) > 10000.0)] = 0.0
            depth = np.cumsum(depth, axis=0)
            
            # Interpolate depth to match the horizontal resolution of slice_data
            if depth.shape[1] != slice_data.shape[1]:
                from scipy.interpolate import interp1d
                # Get the latitude dimension sizes
                n_lats_layer = depth.shape[1]
                n_lats_data = slice_data.shape[1]
                # Create interpolation indices
                x_layer = np.linspace(0, 1, n_lats_layer)
                x_data = np.linspace(0, 1, n_lats_data)
                # Interpolate each vertical level
                depth_interp = np.zeros((depth.shape[0], n_lats_data))
                for k in range(depth.shape[0]):
                    f = interp1d(x_layer, depth[k, :], kind='linear', fill_value='extrapolate')
                    depth_interp[k, :] = f(x_data)
                depth = depth_interp
        else:
            # Create uniform depth levels if no h variable
            num_levels = slice_data.shape[0]
            depth = np.tile(np.linspace(0, max_depth, num_levels), 
                          (slice_data.shape[1], 1)).T
    except:
        # Create uniform depth levels
        num_levels = slice_data.shape[0]
        depth = np.tile(np.linspace(0, max_depth, num_levels), 
                      (slice_data.shape[1], 1)).T
    
    slice_data = np.clip(slice_data, bounds[0], bounds[1])
    lats = np.squeeze(grid.lat)[:, lon_index]
    y = np.tile(lats, (np.shape(depth)[0], 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the filled contours
    contourf_plot = ax.contourf(y, -depth, slice_data,
                                levels=np.linspace(bounds[0], bounds[1], 100),
                                vmin=bounds[0], vmax=bounds[1],
                                cmap=colormap)

    # Add contour lines
    contour_levels = np.linspace(bounds[0], bounds[1], 5)
    ax.contour(y, -depth, slice_data,
               levels=contour_levels,
               colors='black',
               linewidths=0.1)

    # Add colorbar
    cbar = fig.colorbar(contourf_plot, ax=ax, shrink=0.5, orientation='horizontal')
    cbar.set_label(f"{variable} ({unit}) Lon {lon}")
    cbar.set_ticks(contour_levels)
    contourf_plot.set_clim(bounds[0], bounds[1])

    ax.set_ylim(-max_depth, 0)
    ax.set_xlim(lats.min(), lats.max())
    title = f"{exp} {PDY} {cyc} {variable} lon {int(lon)}"
    ax.set_title(title)
    
    dirname = os.path.join(vrfyout, variable)
    os.makedirs(dirname, exist_ok=True)
    figname = os.path.join(dirname,
        variable + '_meridional_lon_' + str(int(lon)) + '_' + str(int(max_depth)) + 'm')
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_climatology_fields(config):
    """
    Plot a single climatology configuration
    """
    try:
        # Handle ocean fields with vertical slices
        if config['type'] == 'ocean':
            # Horizontal slices
            for variable, bounds in config['variables_horiz'].items():
                plotHorizontalSlice_clim(
                    config['grid_file'], config['data_file'], variable, bounds,
                    config['colormap'], 'Global', config['vrfyout'],
                    config['exp'], config['PDY'], config['cyc']
                )
            
            # Zonal slices
            for variable, bounds in config['variables_zonal'].items():
                for lat in config['lats']:
                    for max_depth in config['max_depths']:
                        plotZonalSlice_clim(
                            config['grid_file'], config['data_file'], variable, bounds,
                            config['colormap'], lat, config['vrfyout'],
                            config['exp'], config['PDY'], config['cyc'], max_depth,
                            layer_file=config.get('layer_file')
                        )
            
            # Meridional slices
            for variable, bounds in config['variables_meridional'].items():
                for lon in config['lons']:
                    for max_depth in config['max_depths']:
                        plotMeridionalSlice_clim(
                            config['grid_file'], config['data_file'], variable, bounds,
                            config['colormap'], lon, config['vrfyout'],
                            config['exp'], config['PDY'], config['cyc'], max_depth,
                            layer_file=config.get('layer_file')
                        )
        
        # Handle ice fields (horizontal only)
        elif config['type'] == 'ice':
            for variable, bounds in config['variables_horiz'].items():
                for proj in config['projs']:
                    plotHorizontalSlice_clim(
                        config['grid_file'], config['data_file'], variable, bounds,
                        config['colormap'], proj, config['vrfyout'],
                        config['exp'], config['PDY'], config['cyc']
                    )
    except Exception as e:
        print(f"Error plotting {config.get('PDY', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()



# Initialize list for all configurations
configs = []

# Loop through all 12 months
for month in range(1, 13):
    month_str = str(month).zfill(2)
    
    ocean_file = os.path.join(climatology_dir, f'woa_on_mom6_layers_{month_str}.nc')
    
    # Check if files exist
    if not os.path.exists(ocean_file):
        print(f"Warning: {ocean_file} not found, skipping ocean plots for month {month}")
    else:
        print(f"Configuring WOA ocean plots for month {month}")
        
        # Ocean climatology plotting configuration
        # Surface plots, zonal and meridional slices for Temp and Salt
        config_ocean = {
            'type': 'ocean',
            'grid_file': grid_file,
            'layer_file': layer_file,
            'data_file': ocean_file,
            'PDY': f'WOA_Month_{month_str}',
            'cyc': '00',
            'exp': 'WOA',
            'lats': np.arange(-60, 60, 10),
            'lons': np.arange(-280, 80, 30),
            'max_depths': [700.0, 5000.0],
            'variables_zonal': {
                'Temp': [-1.8, 34.0],
                'Salt': [32, 40]
            },
            'variables_meridional': {
                'Temp': [-1.8, 34.0],
                'Salt': [32, 40]
            },
            'variables_horiz': {
                'Temp': [-1.8, 34.0],
                'Salt': [32, 40]
            },
            'colormap': 'nipy_spectral',
            'vrfyout': os.path.join(vrfyout, f'month_{month_str}', 'ocean')
        }
        configs.append(config_ocean)

print(f"\nTotal configurations to plot: {len(configs)}")
print("Starting plotting process...\n")

# Create a list to store the processes
processes = []

# Iterate over configs and create parallel processes
for config in configs:
    process = Process(target=plot_climatology_fields, args=(config,))
    process.start()
    processes.append(process)

# Wait for all processes to finish
for process in processes:
    process.join()

print("\n==============================================")
print("Climatology plotting completed!")
print(f"Plots saved to: {vrfyout}")
print("==============================================")
