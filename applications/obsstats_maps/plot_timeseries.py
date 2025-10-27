#!/usr/bin/env python3
"""
Enhanced timeseries application that can plot statistics from multiple experiments.
Can read existing NetCDF statistics files or generate them from IODA files.
Supports stats-only processing (no scatter plots) for efficient timeseries generation.
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import sys
from datetime import datetime, timedelta
from glob import glob

# Import functions from the main plt_diags_maps module
try:
    from plt_diags_maps import load_ioda_diags, IODAData, save_statistics_to_netcdf
except ImportError:
    print("Error: Could not import from plt_diags_maps.py")
    print("Make sure plt_diags_maps.py is in the same directory")
    sys.exit(1)


def load_statistics_file(filename):
    """Load statistics from NetCDF file."""
    try:
        ds = nc.Dataset(filename, 'r')

        # Read variables
        time_var = ds.variables['time'][:]
        mean_var = ds.variables['mean'][:]
        std_var = ds.variables['std'][:]
        n_obs_var = ds.variables['n_observations'][:]

        # Get reference time from attributes
        time_units = ds.variables['time'].units
        # Extract reference time from units string like "seconds since 2021-07-01 00:00:00"
        ref_time_str = time_units.split('since ')[-1]
        ref_time = datetime.strptime(ref_time_str, "%Y-%m-%d %H:%M:%S")

        # Convert time to datetime objects
        times = [ref_time + timedelta(seconds=int(t)) for t in time_var]

        # Get metadata
        experiment_id = getattr(ds, 'experiment_id', os.path.basename(filename))
        variable = getattr(ds, 'variable', 'unknown')

        ds.close()

        return {
            'times': times,
            'mean': mean_var,
            'std': std_var,
            'n_obs': n_obs_var,
            'experiment_id': experiment_id,
            'variable': variable,
            'filename': filename
        }
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def plot_timeseries_single(data, output_file=None, title=None):
    """Create 3-panel timeseries plot for a single observation space."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    times = data['times']
    obs_space_name = data['experiment_id']

    # Plot mean
    ax1.plot(times, data['mean'], color='blue', marker='o', markersize=3, linewidth=1.5)
    ax1.set_ylabel(f"Mean ({data['variable']})", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{obs_space_name} - Mean", fontsize=10)

    # Plot standard deviation
    ax2.plot(times, data['std'], color='red', marker='s', markersize=3, linewidth=1.5)
    ax2.set_ylabel(f"Std ({data['variable']})", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"{obs_space_name} - Standard Deviation", fontsize=10)

    # Plot observation count
    ax3.plot(times, data['n_obs'], color='green', marker='^', markersize=3, linewidth=1.5)
    ax3.set_ylabel("Observation Count", fontsize=12)
    ax3.set_xlabel("Time", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f"{obs_space_name} - Observation Count", fontsize=10)

    # Format x-axis
    fig.autofmt_xdate()

    # Add overall title
    if title:
        fig.suptitle(f"{title} - {obs_space_name}", fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f"{obs_space_name} - {data['variable']} Statistics", fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Timeseries plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_timeseries(observation_spaces_data, output_dir=None, title_prefix=None):
    """Create separate 3-panel timeseries plots for each observation space."""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for data in observation_spaces_data:
        obs_space_name = data['experiment_id']

        # Generate output filename for this observation space
        if output_dir:
            # Create a safe filename from the observation space name
            safe_name = "".join(c for c in obs_space_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            output_file = os.path.join(output_dir, f"{safe_name}_timeseries.png")
        else:
            output_file = None

        # Create individual plot
        plot_timeseries_single(data, output_file, title_prefix)


def generate_statistics_from_ioda(data_path, varname, geovar_group='ObsValue',
                                  time_interval=3600, experiment_id='', output_file=None,
                                  ice_edge_stats=False, depth_bins=None, ocean_basins=None):
    """Generate statistics from IODA files when statistics file doesn't exist.

    Args:
        depth_bins: List of depth ranges as tuples, e.g., [(0, 50), (50, 200), (200, float('inf'))]
        ocean_basins: List of ocean basin codes to include, e.g., [1, 2, 3] or None for all basins
    """
    print(f"Generating statistics from IODA files: {data_path}")

    # Load IODA files
    netcdf_files = glob(data_path)
    if not netcdf_files:
        print(f"Error: No files found matching pattern: {data_path}")
        return None

    print(f"Found {len(netcdf_files)} IODA files")

    all_iodaData = IODAData()

    # Load files
    for netcdf_file in netcdf_files:
        print(f"Loading: {os.path.basename(netcdf_file)}")
        try:
            iodaData = load_ioda_diags(netcdf_file, varname, geovar_group)
            all_iodaData.append(iodaData)
        except Exception as e:
            print(f"Warning: Failed to load {netcdf_file}: {e}")
            continue

    if len(all_iodaData.time_data) == 0:
        print("Error: No valid data loaded")
        return None

    # Convert to arrays
    min_time, max_time = all_iodaData.toarray()
    current_time = min_time

    # Print information about depth and ocean basin data availability
    depth_available = not np.all(np.isnan(all_iodaData.depth))
    basin_available = not np.all(all_iodaData.ocean_basin == -1)

    print(f"Depth data available: {depth_available}")
    print(f"Ocean basin data available: {basin_available}")

    if depth_bins and not depth_available:
        print("Warning: Depth binning requested but no depth data available")
        depth_bins = None

    if ocean_basins and not basin_available:
        print("Warning: Ocean basin filtering requested but no basin data available")
        ocean_basins = None

    def create_spatial_mask(depth_range=None, basin_list=None):
        """Create a mask for depth and ocean basin filtering."""
        mask = np.ones(len(all_iodaData.depth), dtype=bool)

        if depth_range and depth_available:
            depth_min, depth_max = depth_range
            if np.isinf(depth_max):
                depth_mask = all_iodaData.depth >= depth_min
            else:
                depth_mask = (all_iodaData.depth >= depth_min) & (all_iodaData.depth < depth_max)
            mask &= depth_mask

        if basin_list and basin_available:
            basin_mask = np.isin(all_iodaData.ocean_basin, basin_list)
            mask &= basin_mask

        return mask

    # Calculate statistics for each time interval
    stats_times = []
    stats_means = []
    stats_stds = []
    stats_n_obs = []

    # Lists for ice-edge statistics (if enabled)
    ice_edge_means = [] if ice_edge_stats and varname == 'icec' else None
    ice_edge_stds = [] if ice_edge_stats and varname == 'icec' else None
    ice_edge_n_obs = [] if ice_edge_stats and varname == 'icec' else None

    print("Computing statistics...")

    # Create spatial mask for depth and ocean basin filtering (applied to all time steps)
    spatial_mask = create_spatial_mask(
        depth_range=depth_bins[0] if depth_bins else None,
        basin_list=ocean_basins
    )

    if np.sum(spatial_mask) == 0:
        print("Warning: No observations pass the spatial filtering criteria")
        return None

    print(f"Spatial filtering: {np.sum(spatial_mask)}/{len(spatial_mask)} observations retained")

    while current_time <= max_time:
        next_time = current_time + timedelta(seconds=time_interval)

        # Get data for current time chunk (use a tighter window to avoid mixing cycles)
        # Use ±1 hour window around each cycle time instead of ±3*time_interval
        window_size = min(3600, time_interval // 2)  # 1 hour or half the interval, whichever is smaller
        time_min = current_time - timedelta(seconds=window_size)
        time_max = current_time + timedelta(seconds=window_size)
        time_mask = (all_iodaData.time_data >= time_min) & (all_iodaData.time_data < time_max)

        # Combine time and spatial masks
        combined_mask = time_mask & spatial_mask

        if np.sum(combined_mask) == 0:
            current_time = next_time
            continue

        # Calculate statistics
        data_mean = np.mean(all_iodaData.geovar[combined_mask])
        data_std = np.std(all_iodaData.geovar[combined_mask])
        n_obs = np.sum(combined_mask)

        stats_times.append(current_time)
        stats_means.append(data_mean)
        stats_stds.append(data_std)
        stats_n_obs.append(n_obs)

        # Calculate ice-edge statistics if enabled
        if ice_edge_means is not None:
            ice_edge_mask = combined_mask & (all_iodaData.obsvalue >= 0.05) & (all_iodaData.obsvalue <= 0.25)

            if np.sum(ice_edge_mask) > 0:
                ice_edge_mean = np.mean(all_iodaData.geovar[ice_edge_mask])
                ice_edge_std = np.std(all_iodaData.geovar[ice_edge_mask])
                ice_edge_count = np.sum(ice_edge_mask)
            else:
                ice_edge_mean = np.nan
                ice_edge_std = np.nan
                ice_edge_count = 0

            ice_edge_means.append(ice_edge_mean)
            ice_edge_stds.append(ice_edge_std)
            ice_edge_n_obs.append(ice_edge_count)

        current_time = next_time

    # Save statistics to file if output_file specified
    if output_file and stats_times:
        save_statistics_to_netcdf(output_file, stats_times, stats_means, stats_stds, stats_n_obs,
                                  varname, experiment_id, ice_edge_means, ice_edge_stds, ice_edge_n_obs)
        print(f"Statistics saved to: {output_file}")

    # Return statistics data
    if stats_times:
        return {
            'times': stats_times,
            'mean': np.array(stats_means),
            'std': np.array(stats_stds),
            'n_obs': np.array(stats_n_obs),
            'experiment_id': experiment_id,
            'variable': varname,
            'filename': output_file or 'generated'
        }
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Plot timeseries statistics for different observation spaces in an experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_timeseries.py config.yaml
  python plot_timeseries.py --help

Config YAML file should contain:
  observation_spaces:  # Each observation space gets its own separate plot
    - name: "SST AVHRR-MB ombg"
      file: "/path/to/sst_statistics.nc"          # Existing statistics file
      ioda_data_path: "/path/to/ioda/*.nc"        # IODA files to process if stats missing
      varname: "sst"                              # Variable name for IODA processing
      geovar_group: "ombg"                        # NetCDF group (default: ObsValue)
      time_interval: 21600                        # Time interval in seconds (default: 3600)
      ice_edge_stats: false                       # Enable ice-edge statistics (default: false)
      stats_output: "/path/to/output_stats.nc"    # Where to save generated statistics
    - name: "Temperature Drifter ombg"
      ioda_data_path: "/path/to/drifter/*.nc"
      varname: "temp"
      geovar_group: "ombg"
      time_interval: 21600
  output_dir: "./timeseries_plots"                # Directory for individual plots
  title: "Experiment Name"                        # Prefix for plot titles

Note: Each observation space will generate a separate 3-panel timeseries plot
      saved as {observation_space_name}_timeseries.png in the output directory.
        '''
    )

    parser.add_argument('config_file',
                        help='YAML configuration file containing experiment files')

    args = parser.parse_args()

    # Load configuration
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Load observation space data (support both new and legacy config formats)
    observation_spaces_data = []

    # Support both 'observation_spaces' (new) and 'experiments' (legacy) keys
    obs_spaces_config = config.get('observation_spaces', config.get('experiments', []))

    for obs_config in obs_spaces_config:
        obs_name = obs_config['name']
        obs_file = obs_config.get('file')

        # New: Support for IODA data processing
        ioda_data_path = obs_config.get('ioda_data_path')
        varname = obs_config.get('varname', 'sst')
        geovar_group = obs_config.get('geovar_group', 'ObsValue')
        time_interval = obs_config.get('time_interval', 3600)
        ice_edge_stats = obs_config.get('ice_edge_stats', False)
        depth_bins = obs_config.get('depth_bins')
        ocean_basins = obs_config.get('ocean_basins')

        data = None

        # Try to load existing statistics file first
        if obs_file:
            # Support glob patterns for files
            if '*' in obs_file:
                files = glob(obs_file)
                if files:
                    obs_file = files[0]  # Take first match
                else:
                    obs_file = None

            if obs_file and os.path.exists(obs_file):
                print(f"Loading existing statistics: {obs_name} from {obs_file}")
                data = load_statistics_file(obs_file)

        # If no statistics file exists or failed to load, try to generate from IODA files
        if not data and ioda_data_path:
            print("Statistics file not found or failed to load. Generating from IODA files...")

            # Generate output filename for statistics
            stats_output = obs_config.get('stats_output')
            if not stats_output and obs_file:
                # Use the intended stats file location
                stats_output = obs_file
            elif not stats_output:
                # Generate a default filename
                safe_name = "".join(c for c in obs_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                stats_output = f"{safe_name}_{varname}_statistics.nc"

            data = generate_statistics_from_ioda(
                data_path=ioda_data_path,
                varname=varname,
                geovar_group=geovar_group,
                time_interval=time_interval,
                experiment_id=obs_name,
                output_file=stats_output,
                ice_edge_stats=ice_edge_stats,
                depth_bins=depth_bins,
                ocean_basins=ocean_basins
            )

        if data:
            # Override experiment ID with config name if provided
            data['experiment_id'] = obs_name
            observation_spaces_data.append(data)
        else:
            print(f"Failed to load or generate data for observation space: {obs_name}")
            print(f"  - Statistics file: {obs_file}")
            print(f"  - IODA data path: {ioda_data_path}")

    if not observation_spaces_data:
        print("Error: No observation space data loaded successfully")
        return

    # Get output settings - now using output_dir instead of single output_file
    output_dir = config.get('output_dir', './timeseries_plots')
    title_prefix = config.get('title')

    # Create separate plots for each observation space
    plot_timeseries(observation_spaces_data, output_dir, title_prefix)


if __name__ == "__main__":
    main()
