import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from datetime import datetime, timedelta
from glob import glob
import yaml
import argparse
from tqdm import tqdm


VARIABLE_MAP = {'sst': 'seaSurfaceTemperature',
                'sss': 'seaSurfaceSalinity',
                'adt': 'absoluteDynamicTopography',
                'icec': 'seaIceFraction',
                'temp': ['waterTemperature', 'seaSurfaceTemperature'],
                'salt': ['salinity']}

# Predefined instrument color dictionary for consistent coloring across plots
INSTRUMENT_COLORS = {
    # Temperature instruments (in-situ)
    'argo': '#1f77b4',          # Blue - Argo floats (profile temperature)
    'drifter': '#ff7f0e',       # Orange - Surface drifters (SST)
    'glider': '#2ca02c',        # Green - Gliders (profile temperature)
    'xbtctd': '#d62728',        # Red - XBT/CTD (profile temperature)
    'trkob': '#9467bd',         # Purple - Track obs (surface temperature)

    # SST instruments (satellite/surface)
    'avhrrf_mb': '#17becf',     # Cyan - AVHRR MetOp-B
    'avhrrf_mc': '#bcbd22',     # Olive - AVHRR MetOp-C
    'viirs_n20': '#e377c2',     # Pink - VIIRS NOAA-20
    'viirs_n21': '#7f7f7f',     # Gray - VIIRS NOAA-21
    'viirs_npp': '#8c564b',     # Brown - VIIRS NPP

    # ADT instruments (altimetry)
    'rads_adt_3a': '#1f77b4',   # Blue - RADS 3a
    'rads_adt_3b': '#ff7f0e',   # Orange - RADS 3b
    'rads_adt_6a': '#2ca02c',   # Green - RADS 6a
    'rads_adt_c2': '#d62728',   # Red - RADS c2
    'rads_adt_j3': '#9467bd',   # Purple - RADS j3
    'rads_adt_sa': '#8c564b',   # Brown - RADS sa
    'rads_adt_sw': '#e377c2',   # Pink - RADS sw

    # Salinity instruments
    'smap': '#17becf',          # Cyan - SMAP
    'smos': '#bcbd22',          # Olive - SMOS

    # Sea ice instruments
    'amsr2': '#1f77b4',         # Blue - AMSR2
    'ssmi': '#ff7f0e',          # Orange - SSM/I

    # Default fallback colors for unknown instruments
    'unknown_1': '#ff9999',     # Light red
    'unknown_2': '#66b3ff',     # Light blue
    'unknown_3': '#99ff99',     # Light green
    'unknown_4': '#ffcc99',     # Light orange
    'unknown_5': '#ff99cc',     # Light pink
}


def extract_instrument_name(filepath):
    """Extract instrument name from IODA file path."""
    filename = os.path.basename(filepath)

    # Try to extract instrument name from common patterns
    # Examples: sst_viirs_npp_20210701T120000Z.nc4, adt_jason3_20210701.nc4
    parts = filename.split('_')
    if len(parts) >= 3:
        # Usually format: variable_instrument_platform_datetime
        if len(parts[1]) > 0:
            instrument = parts[1]
            if len(parts) >= 4 and parts[2] not in ['20', '19']:  # Not a date
                instrument = f"{parts[1]}_{parts[2]}_{parts[3]}"
            # Remove file extension if present (e.g., .nc, .nc4)
            instrument = os.path.splitext(instrument)[0]
            return instrument

    # Fallback to filename without extension
    return filename.replace('.nc4', '').replace('.nc', '')


class IODAData:
    def __init__(self, time_data=None, lat=None, lon=None, geovar=None, varname='', obsvalue=None, instruments=None,
                 depth=None, ocean_basin=None):
        self.time_data = time_data if time_data is not None else []
        self.lat = lat if lat is not None else []
        self.lon = lon if lon is not None else []
        self.geovar = geovar if geovar is not None else []
        self.varname = varname
        self.obsvalue = obsvalue if obsvalue is not None else []  # Always store ObsValue for ice-edge detection
        self.instruments = instruments if instruments is not None else []  # Track instrument names for multi-instrument plotting
        self.depth = depth if depth is not None else []  # Depth information for in-situ data
        self.ocean_basin = ocean_basin if ocean_basin is not None else []  # Ocean basin information

    def __str__(self):
        return (f"IODAData(time_data={len(self.time_data)}, lat={len(self.lat)}, "
                f"lon={len(self.lon)}, geovar={len(self.geovar)}, "
                f"obsvalue={len(self.obsvalue)}, instruments={set(self.instruments)}, "
                f"depth={len(self.depth)}, ocean_basin={len(self.ocean_basin)})")

    def __repr__(self):
        return self.__str__()

    def append(self, other):
        # Assert that self.varname == other.varname
        if len(self.time_data) > 0:
            assert self.varname == other.varname
        if len(self.time_data) == 0:
            self.varname = other.varname
        self.time_data.append(other.time_data)
        self.lat.append(other.lat)
        self.lon.append(other.lon)
        self.geovar.append(other.geovar)
        self.obsvalue.append(other.obsvalue)
        self.instruments.append(other.instruments)
        self.depth.append(other.depth)
        self.ocean_basin.append(other.ocean_basin)

    def toarray(self):
        # Filter out empty arrays before concatenating
        try:
            # Only concatenate if we have non-empty arrays
            non_empty_time = [arr for arr in self.time_data if len(arr) > 0]
            non_empty_lat = [arr for arr in self.lat if len(arr) > 0]
            non_empty_lon = [arr for arr in self.lon if len(arr) > 0]
            non_empty_geovar = [arr for arr in self.geovar if len(arr) > 0]
            non_empty_obsvalue = [arr for arr in self.obsvalue if len(arr) > 0]
            non_empty_instruments = [arr for arr in self.instruments if len(arr) > 0]
            non_empty_depth = [arr for arr in self.depth if len(arr) > 0]
            non_empty_ocean_basin = [arr for arr in self.ocean_basin if len(arr) > 0]

            # Handle case where no observations are present
            if not non_empty_time:
                # Return None for both min and max time to indicate empty dataset
                return None, None

            # Concatenate data from all files
            self.time_data = np.concatenate(non_empty_time)
            self.lat = np.concatenate(non_empty_lat)
            self.lon = np.concatenate(non_empty_lon)
            self.geovar = np.concatenate(non_empty_geovar)
            self.obsvalue = np.concatenate(non_empty_obsvalue)
            self.instruments = np.concatenate(non_empty_instruments)
            self.depth = np.concatenate(non_empty_depth)
            self.ocean_basin = np.concatenate(non_empty_ocean_basin)

            # Handle case where no observations are present after concatenation
            if len(self.time_data) == 0:
                # Return None for both min and max time to indicate empty dataset
                return None, None

            min_time, max_time = min(self.time_data), max(self.time_data)
            return min_time, max_time

        except Exception as e:
            # If concatenation fails for any reason, return empty dataset
            print(f"Warning: Failed to concatenate arrays in toarray(): {e}")
            return None, None


def load_ioda_diags(netcdf_file, var_name_short, geovar_group='ObsValue'):
    try:
        # Open NetCDF file
        ds = nc.Dataset(netcdf_file, 'r')

        # Get the full variable name from the mapping
        var_name_options = VARIABLE_MAP[var_name_short]
        if not isinstance(var_name_options, list):
            var_name_options = [var_name_options]

        # Try each variable name option until we find one that exists
        var_name = None
        for var_option in var_name_options:
            if var_option in ds.groups[geovar_group].variables:
                var_name = var_option
                break

        if var_name is None:
            available_vars = list(ds.groups[geovar_group].variables.keys())
            ds.close()
            raise ValueError(f"None of the expected variables {var_name_options} found in group {geovar_group}. "
                             f"Available variables: {available_vars}")

        # Extract instrument name from file path
        instrument = extract_instrument_name(netcdf_file)

    except Exception as e:
        if "truth value of an array" in str(e):
            return IODAData()
        else:
            raise e

    # Read necessary data with robust error handling
    try:
        time_data = ds.groups['MetaData'].variables['dateTime'][:]
        ref_time_str = ds.groups['MetaData'].variables['dateTime'].getncattr('units').split(' ')[-1]
        ref_time = datetime.strptime(ref_time_str, "%Y-%m-%dT%H:%M:%SZ")

        lat = ds.groups['MetaData'].variables['latitude'][:]
        lon = ds.groups['MetaData'].variables['longitude'][:]
        geovar = ds.groups[geovar_group].variables[var_name][:]
        obsvalue = ds.groups['ObsValue'].variables[var_name][:]  # Always load ObsValue for ice-edge detection

        # Try to read QC from EffectiveQC0 group; if missing create zeros with same length as obsvalue
        if 'EffectiveQC0' in ds.groups and var_name in ds.groups['EffectiveQC0'].variables:
            qc = ds.groups['EffectiveQC0'].variables[var_name][:]
        else:
            qc = np.zeros_like(obsvalue, dtype=np.int8)

        # Try to read depth and ocean basin data from MetaData
        if 'depth' in ds.groups['MetaData'].variables:
            depth = ds.groups['MetaData'].variables['depth'][:]
        else:
            depth = np.full(len(lat), np.nan)  # Fill with NaN if not available

        if 'oceanBasin' in ds.groups['MetaData'].variables:
            ocean_basin = ds.groups['MetaData'].variables['oceanBasin'][:]
        else:
            ocean_basin = np.full(len(lat), -1, dtype=np.int32)  # Fill with -1 if not available

        # Convert masked arrays to regular numpy arrays to avoid truth value issues
        time_data = np.asarray(time_data)
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        geovar = np.asarray(geovar)
        obsvalue = np.asarray(obsvalue)
        qc = np.asarray(qc)
        depth = np.asarray(depth)
        ocean_basin = np.asarray(ocean_basin)

    except Exception as e:
        ds.close()
        if "truth value of an array" in str(e) or "invalid index" in str(e) or "cannot broadcast" in str(e):
            print(f"Warning: Array access error in {netcdf_file}: {e}")
            # Return empty data structure to indicate no valid data
            empty_data = IODAData()
            return empty_data
        else:
            raise e

    # Remove fill values - handle potential shape mismatches
    try:
        # Ensure all arrays are 1D and the same length first
        min_len = min(len(time_data), len(lat), len(lon), len(geovar), len(obsvalue), len(qc), len(depth), len(ocean_basin))

        # Truncate all arrays to the same length and ensure they're regular numpy arrays
        time_data = np.asarray(time_data).flatten()[:min_len]
        lat = np.asarray(lat).flatten()[:min_len]
        lon = np.asarray(lon).flatten()[:min_len]
        geovar = np.asarray(geovar).flatten()[:min_len]
        obsvalue = np.asarray(obsvalue).flatten()[:min_len]
        qc = np.asarray(qc).flatten()[:min_len]
        depth = np.asarray(depth).flatten()[:min_len]
        ocean_basin = np.asarray(ocean_basin).flatten()[:min_len]

        # Verify all arrays have valid lengths
        if min_len == 0:
            print(f"Warning: No valid data in {netcdf_file} - all arrays empty")
            ds.close()
            empty_data = IODAData()
            return empty_data

        # Now create the valid mask with consistent shapes
        valid_mask = (geovar > -3.368795e+38) & (obsvalue > -3.368795e+38) & (lat > -3.368795e+38) & (lon > -3.368795e+38) & (qc == 0)

        # Apply the mask
        time_data = time_data[valid_mask]
        lat = lat[valid_mask]
        lon = lon[valid_mask]
        geovar = geovar[valid_mask]
        obsvalue = obsvalue[valid_mask]
        depth = depth[valid_mask]
        ocean_basin = ocean_basin[valid_mask]

    except Exception as e:
        # Catch any error that occurs during array processing
        ds.close()
        empty_data = IODAData()
        return empty_data

    # Convert time to datetime objects
    # Ensure time_data is a 1D array of scalars before conversion
    time_data_flat = np.asarray(time_data).flatten()
    time_data = np.array([ref_time + timedelta(seconds=int(t)) for t in time_data_flat])

    # Create instrument array for all observations
    instruments = np.array([instrument] * len(time_data))

    # Close dataset
    ds.close()

    return IODAData(time_data, lat, lon, geovar, varname=var_name_short, obsvalue=obsvalue,
                    instruments=instruments, depth=depth, ocean_basin=ocean_basin)


def save_statistics_to_netcdf(filename, times, means, stds, n_obs, varname, experiment_id,
                              ice_edge_means=None, ice_edge_stds=None, ice_edge_n_obs=None):
    """Save timeseries statistics to NetCDF file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create NetCDF file
    ds = nc.Dataset(filename, 'w')

    # Create dimensions
    ds.createDimension('time', len(times))

    # Create variables
    time_var = ds.createVariable('time', 'f8', ('time',))
    mean_var = ds.createVariable('mean', 'f4', ('time',))
    std_var = ds.createVariable('std', 'f4', ('time',))
    n_obs_var = ds.createVariable('n_observations', 'i4', ('time',))

    # Add ice-edge variables if provided
    if ice_edge_means is not None:
        ice_edge_mean_var = ds.createVariable('ice_edge_mean', 'f4', ('time',))
        ice_edge_std_var = ds.createVariable('ice_edge_std', 'f4', ('time',))
        ice_edge_n_obs_var = ds.createVariable('ice_edge_n_observations', 'i4', ('time',))

    # Convert datetime objects to seconds since first time
    reference_time = times[0]
    time_seconds = [(t - reference_time).total_seconds() for t in times]

    # Fill variables
    time_var[:] = time_seconds
    mean_var[:] = means
    std_var[:] = stds
    n_obs_var[:] = n_obs

    # Fill ice-edge variables if provided
    if ice_edge_means is not None:
        ice_edge_mean_var[:] = ice_edge_means
        ice_edge_std_var[:] = ice_edge_stds
        ice_edge_n_obs_var[:] = ice_edge_n_obs

    # Add attributes
    time_var.units = f'seconds since {reference_time.strftime("%Y-%m-%d %H:%M:%S")}'
    time_var.long_name = 'time'
    mean_var.long_name = f'mean {varname}'
    std_var.long_name = f'standard deviation {varname}'
    n_obs_var.long_name = 'number of observations'

    # Add ice-edge attributes if variables exist
    if ice_edge_means is not None:
        ice_edge_mean_var.long_name = f'ice-edge mean {varname} (0.1-0.2)'
        ice_edge_std_var.long_name = f'ice-edge standard deviation {varname} (0.1-0.2)'
        ice_edge_n_obs_var.long_name = 'number of ice-edge observations (0.1-0.2)'

    # Global attributes
    ds.title = f'Statistics timeseries for {varname}'
    ds.experiment_id = experiment_id
    ds.variable = varname
    ds.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Close file
    ds.close()


def plot_data(iodaData, frame_idx, time_interval=300, save_dir='./frames', bounds=[-2, 35],
              projection='platecarree', colormap='gist_ncar', scatter_size=0.1, experiment_id='', stats_only=False,
              ice_edge_stats=False, multi_instrument=False, instrument_colormap='tab10'):
    min_time, max_time = iodaData.toarray()
    current_time = min_time

    # Calculate total number of time steps for progress bar
    total_steps = int((max_time - min_time).total_seconds() / time_interval) + 1

    # Lists to store statistics for timeseries
    stats_times = []
    stats_means = []
    stats_stds = []
    stats_n_obs = []

    # Lists for ice-edge statistics (if enabled)
    ice_edge_means = [] if ice_edge_stats and iodaData.varname == 'icec' else None
    ice_edge_stds = [] if ice_edge_stats and iodaData.varname == 'icec' else None
    ice_edge_n_obs = [] if ice_edge_stats and iodaData.varname == 'icec' else None

    # Set up projection based on configuration
    if projection == 'polar_north':
        proj = ccrs.NorthPolarStereo()
        figsize = (8, 8)
    elif projection == 'polar_south':
        proj = ccrs.SouthPolarStereo()
        figsize = (8, 8)
    else:  # Default to PlateCarree
        proj = ccrs.PlateCarree()
        figsize = (10, 5)

    # Create progress bar
    pbar = tqdm(total=total_steps, desc=f"Processing {iodaData.varname}", unit="frames")

    while current_time <= max_time:
        next_time = current_time + timedelta(seconds=time_interval)

        # Get data for current time chunk
        time_min = current_time - timedelta(seconds=3 * time_interval)
        time_max = current_time + timedelta(seconds=3 * time_interval)
        time_mask = (iodaData.time_data >= time_min) & (iodaData.time_data < time_max)
        if np.sum(time_mask) == 0:
            current_time = next_time
            pbar.update(1)
            continue  # Skip empty time bins

        # Calculate statistics for the current time chunk
        data_mean = np.mean(iodaData.geovar[time_mask])
        data_std = np.std(iodaData.geovar[time_mask])
        n_obs = np.sum(time_mask)

        # Store statistics for timeseries output
        stats_times.append(current_time)
        stats_means.append(data_mean)
        stats_stds.append(data_std)
        stats_n_obs.append(n_obs)

        # Calculate ice-edge statistics if enabled for ice concentration
        if ice_edge_means is not None:
            # Ice-edge: ObsValue between 0.05 and 0.6, compute statistics on geovar
            ice_edge_mask = time_mask & (iodaData.obsvalue >= 0.05) & (iodaData.obsvalue <= 0.25)

            if np.sum(ice_edge_mask) > 0:
                ice_edge_mean = np.mean(iodaData.geovar[ice_edge_mask])
                ice_edge_std = np.std(iodaData.geovar[ice_edge_mask])
                ice_edge_count = np.sum(ice_edge_mask)
            else:
                # No ice-edge observations in this time frame
                ice_edge_mean = np.nan
                ice_edge_std = np.nan
                ice_edge_count = 0

            ice_edge_means.append(ice_edge_mean)
            ice_edge_stds.append(ice_edge_std)
            ice_edge_n_obs.append(ice_edge_count)

        # Only create plots if not in stats_only mode
        if not stats_only:
            # Plot
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': proj})

            # Set extent based on projection
            if projection == 'polar_north':
                ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
            elif projection == 'polar_south':
                ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
            else:
                ax.set_global()

            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND, color='lightgray')

            if multi_instrument:
                # Plot by instrument with predefined colors
                unique_instruments = np.unique(iodaData.instruments[time_mask])

                # Plot each instrument separately using predefined colors
                for instrument in unique_instruments:
                    inst_mask = time_mask & (iodaData.instruments == instrument)
                    if np.sum(inst_mask) > 0:
                        color = get_instrument_color(instrument)
                        ax.scatter(iodaData.lon[inst_mask], iodaData.lat[inst_mask],
                                   c=color, label=instrument, s=scatter_size, alpha=0.7,
                                   transform=ccrs.PlateCarree())

                # Add legend for instruments
                ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)

                plt.title(
                    f"{iodaData.varname} Multi-Instrument from {current_time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"to {next_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )
            else:
                # Standard plotting by variable value
                sc = ax.scatter(iodaData.lon[time_mask], iodaData.lat[time_mask], c=iodaData.geovar[time_mask],
                                cmap=colormap, s=scatter_size, transform=ccrs.PlateCarree(),
                                vmin=bounds[0], vmax=bounds[1])

                plt.colorbar(sc, ax=ax, orientation='vertical', label=f'{iodaData.varname}')
                plt.title(
                    f"{iodaData.varname} from {current_time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"to {next_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )

            # Add experiment identifier and statistics in upper left corner
            if experiment_id:
                # Create text with experiment ID and statistics
                stats_text = f"{experiment_id}\nMean: {data_mean:.3f}\nStd: {data_std:.3f}\nN: {n_obs}"

                # Add ice-edge statistics if available and not NaN
                if ice_edge_means is not None and not np.isnan(ice_edge_means[-1]):
                    ice_mean = ice_edge_means[-1]
                    ice_std = ice_edge_stds[-1]
                    ice_count = ice_edge_n_obs[-1]
                    stats_text += f"\nIce-edge Mean: {ice_mean:.3f}\nIce-edge Std: {ice_std:.3f}\nIce-edge N: {ice_count}"

                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, fontweight='bold',
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Save figure
            frame_filename = os.path.join(save_dir, f"{iodaData.varname}_frame_{frame_idx:04d}.png")
            plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Update time and frame index
        current_time = next_time
        frame_idx += 1
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Save statistics to NetCDF file
    if stats_times:
        stats_filename = os.path.join(save_dir, f"{iodaData.varname}_statistics.nc")
        save_statistics_to_netcdf(stats_filename, stats_times, stats_means, stats_stds, stats_n_obs,
                                  iodaData.varname, experiment_id, ice_edge_means, ice_edge_stds, ice_edge_n_obs)


def get_instrument_color(instrument_name):
    """Get color for an instrument from the predefined dictionary.

    Args:
        instrument_name (str): Name of the instrument

    Returns:
        str: Hex color code for the instrument
    """
    # First try exact match
    if instrument_name in INSTRUMENT_COLORS:
        return INSTRUMENT_COLORS[instrument_name]

    # Try partial matches for complex instrument names
    for key in INSTRUMENT_COLORS:
        if key in instrument_name.lower() or instrument_name.lower() in key:
            return INSTRUMENT_COLORS[key]

    # If no match found, generate a fallback color based on hash
    # This ensures consistent colors for the same unknown instrument
    import hashlib
    hash_value = int(hashlib.md5(instrument_name.encode()).hexdigest()[:6], 16)
    # Convert to RGB with good visibility
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    # Ensure colors are not too dark
    r = max(r, 100)
    g = max(g, 100)
    b = max(b, 100)
    return f'#{r:02x}{g:02x}{b:02x}'


def get_instrument_colors_for_list(instrument_list):
    """Get colors for a list of instruments using the predefined dictionary.

    Args:
        instrument_list (list): List of instrument names

    Returns:
        list: List of hex color codes corresponding to the instruments
    """
    return [get_instrument_color(instrument) for instrument in instrument_list]


def main():
    parser = argparse.ArgumentParser(
        description='Generate diagnostic maps from IODA NetCDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plt_diags_maps.py config.yaml
  python plt_diags_maps.py --help

Config YAML file should contain:
  time_interval: 300      # Time interval in seconds
  save_dir: "./frames"    # Output directory for plots
  varname: "sst"          # Variable name (sst, sss, adt, icec)
  geovar_group: "ObsValue"  # NetCDF group name containing the geophysical variable
  bounds: [-2, 35]        # Color scale bounds [min, max]
  data_path: "/path/to/netcdf/files/sst_*.20210701*.nc4"  # Full path pattern to NetCDF files
  projection: "platecarree"  # Map projection: "platecarree", "polar_north", "polar_south"
  colormap: "viridis"     # Matplotlib colormap (e.g., "viridis", "plasma", "coolwarm", "RdBu_r")
  scatter_size: 0.1       # Size of scatter plot markers (float, default: 0.1)
  experiment_id: "EXP001" # Experiment identifier text (displayed in upper left corner)
  stats_only: false       # If true, only generate statistics NetCDF file, no PNG plots
  ice_edge_stats: false   # If true, compute statistics where ObsValue is 0.1-0.2 (ice edge)
  multi_instrument: false # If true, color points by instrument using predefined colors
  instrument_colormap: "tab10"  # Legacy parameter - now uses predefined INSTRUMENT_COLORS dict
        '''
    )

    parser.add_argument('config_file',
                        help='YAML configuration file containing plot parameters')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    time_interval = config['time_interval']
    save_dir = config['save_dir']
    varname = config['varname']
    bounds = config['bounds']
    projection = config.get('projection', 'platecarree')  # Default to PlateCarree if not specified
    geovar_group = config.get('geovar_group', 'ObsValue')  # Default to ObsValue if not specified
    colormap = config.get('colormap', 'gist_ncar')  # Default to gist_ncar if not specified
    scatter_size = config.get('scatter_size', 0.1)  # Default to 0.1 if not specified
    experiment_id = config.get('experiment_id', '')  # Default to empty string if not specified
    stats_only = config.get('stats_only', False)  # Default to False if not specified
    ice_edge_stats = config.get('ice_edge_stats', False)  # Default to False if not specified
    multi_instrument = config.get('multi_instrument', False)  # Default to False if not specified
    instrument_colormap = config.get('instrument_colormap', 'tab10')  # Default to tab10 if not specified

    search_pattern = config['data_path']

    os.makedirs(save_dir, exist_ok=True)
    netcdf_files = glob(search_pattern)
    all_iodaData = IODAData()

    # Load files with progress bar
    for netcdf_file in tqdm(netcdf_files, desc="Loading NetCDF files"):
        iodaData = load_ioda_diags(netcdf_file, varname, geovar_group)
        all_iodaData.append(iodaData)

    # Plot data
    plot_data(all_iodaData, frame_idx=0, time_interval=time_interval, save_dir=save_dir, bounds=bounds,
              projection=projection, colormap=colormap, scatter_size=scatter_size, experiment_id=experiment_id,
              stats_only=stats_only, ice_edge_stats=ice_edge_stats, multi_instrument=multi_instrument,
              instrument_colormap=instrument_colormap)


if __name__ == "__main__":
    main()
