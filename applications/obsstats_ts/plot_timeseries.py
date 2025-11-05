#!/usr/bin/env python3
"""
Enhanced timeseries application that can plot statistics from multiple experiments.
Can read existing NetCDF statistics files or generate them from IODA files.
Supports stats-only processing (no scatter plots) for efficient timeseries generation.

NEW FEATURES:
- Incremental statistics updates: Can append new time periods to existing statistics files
  instead of recomputing everything from scratch
- Missing period detection: Automatically identifies time gaps in existing statistics
- Efficient processing: Only processes IODA files for missing time periods
- Chronological ordering: Maintains proper time ordering when appending new data
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
# Add the obsstats_maps directory to the path for importing plt_diags_maps
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'obsstats_maps'))
# Performance optimization: Use cached loader for massive speedup (5-50x)
from fast_loader import load_ioda_diags, IODAData, save_statistics_to_netcdf as _original_save_statistics_to_netcdf


def save_statistics_to_netcdf(filename, *args, **kwargs):
    """
    Wrapper for save_statistics_to_netcdf that ensures output directory exists.
    """
    if filename:
        # Use absolute path to handle relative paths properly
        abs_filename = os.path.abspath(filename)
        output_dir = os.path.dirname(abs_filename)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ Created output directory: {output_dir}")

    return _original_save_statistics_to_netcdf(filename, *args, **kwargs)
def process_observation_spaces_sequentially(obs_spaces_config, exp_name=None):
    """
    Process observation spaces sequentially with progress tracking.
    For multi-experiment compatibility, this processes individual observation spaces
    and returns data that can be grouped later.
    """
    total_spaces = len(obs_spaces_config)
    print(f"Processing {total_spaces} observation spaces...")

    processed_data = []
    processed_count = 0

    for i, obs_config in enumerate(obs_spaces_config, 1):
        try:
            # Progress indicator for large configs
            obs_name = obs_config.get('name', f"Space {i}")
            print(f"[{i:3d}/{total_spaces}] Processing: {obs_name}")

            # Process individual observation space
            data = process_observation_space(obs_config, exp_name)
            processed_data.append(data)
            processed_count += 1

            # Memory management for large configs
            if i % 20 == 0:
                print(f"    Progress: {processed_count}/{total_spaces} spaces completed ({100*processed_count/total_spaces:.1f}%)")

        except Exception as e:
            print(f"ERROR processing space {i}: {e}")
            processed_data.append(None)
            continue

    print(f"Completed processing: {processed_count}/{total_spaces} spaces successful")
    return processed_data


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


def append_statistics_to_netcdf(filename, new_times, new_means, new_stds, new_n_obs,
                                varname, experiment_id, new_ice_edge_means=None,
                                new_ice_edge_stds=None, new_ice_edge_n_obs=None):
    """Append new statistics to existing NetCDF file."""
    try:
        # Load existing data
        existing_data = load_statistics_file(filename)
        if not existing_data:
            # File doesn't exist or failed to load, create new file
            save_statistics_to_netcdf(filename, new_times, new_means, new_stds, new_n_obs,
                                      varname, experiment_id, new_ice_edge_means,
                                      new_ice_edge_stds, new_ice_edge_n_obs)
            return True

        # Combine existing and new data
        try:
            combined_times = list(existing_data['times']) + list(new_times)

            # Ensure arrays are 1D before concatenation
            existing_means = np.asarray(existing_data['mean']).flatten()
            existing_stds = np.asarray(existing_data['std']).flatten()
            existing_n_obs = np.asarray(existing_data['n_obs']).flatten()

            new_means_flat = np.asarray(new_means).flatten()
            new_stds_flat = np.asarray(new_stds).flatten()
            new_n_obs_flat = np.asarray(new_n_obs).flatten()

            combined_means = np.concatenate([existing_means, new_means_flat])
            combined_stds = np.concatenate([existing_stds, new_stds_flat])
            combined_n_obs = np.concatenate([existing_n_obs, new_n_obs_flat])

        except (ValueError, TypeError) as e:
            if "truth value of an array" in str(e) or "could not broadcast" in str(e) or "cannot concatenate" in str(e):
                print(f"Warning: Array concatenation issue, shapes may be incompatible: {e}")
                print(f"  Existing data shapes - mean: {np.asarray(existing_data['mean']).shape}, std: {np.asarray(existing_data['std']).shape}, n_obs: {np.asarray(existing_data['n_obs']).shape}")
                print(f"  New data shapes - mean: {np.asarray(new_means).shape}, std: {np.asarray(new_stds).shape}, n_obs: {np.asarray(new_n_obs).shape}")
                # Fall back to just saving new data (overwrite mode)
                save_statistics_to_netcdf(filename, new_times, new_means, new_stds, new_n_obs,
                                          varname, experiment_id, new_ice_edge_means,
                                          new_ice_edge_stds, new_ice_edge_n_obs)
                return True
            else:
                raise e

        # Handle ice edge data if present
        combined_ice_edge_means = None
        combined_ice_edge_stds = None
        combined_ice_edge_n_obs = None

        # Check if existing file has ice edge data
        try:
            ds = nc.Dataset(filename, 'r')
            has_existing_ice_edge = 'ice_edge_mean' in ds.variables
            ds.close()

            if has_existing_ice_edge or new_ice_edge_means is not None:
                # Load existing ice edge data if it exists
                if has_existing_ice_edge:
                    ds = nc.Dataset(filename, 'r')
                    existing_ice_edge_means = ds.variables['ice_edge_mean'][:]
                    existing_ice_edge_stds = ds.variables['ice_edge_std'][:]
                    existing_ice_edge_n_obs = ds.variables['ice_edge_n_observations'][:]
                    ds.close()
                else:
                    # Create placeholder arrays for existing data
                    existing_ice_edge_means = np.full(len(existing_data['times']), np.nan)
                    existing_ice_edge_stds = np.full(len(existing_data['times']), np.nan)
                    existing_ice_edge_n_obs = np.zeros(len(existing_data['times']), dtype=int)

                # Handle new ice edge data
                if new_ice_edge_means is None:
                    new_ice_edge_means = np.full(len(new_times), np.nan)
                    new_ice_edge_stds = np.full(len(new_times), np.nan)
                    new_ice_edge_n_obs = np.zeros(len(new_times), dtype=int)

                # Combine ice edge data with shape handling
                try:
                    existing_ice_edge_means_flat = np.asarray(existing_ice_edge_means).flatten()
                    existing_ice_edge_stds_flat = np.asarray(existing_ice_edge_stds).flatten()
                    existing_ice_edge_n_obs_flat = np.asarray(existing_ice_edge_n_obs).flatten()

                    new_ice_edge_means_flat = np.asarray(new_ice_edge_means).flatten()
                    new_ice_edge_stds_flat = np.asarray(new_ice_edge_stds).flatten()
                    new_ice_edge_n_obs_flat = np.asarray(new_ice_edge_n_obs).flatten()

                    combined_ice_edge_means = np.concatenate([existing_ice_edge_means_flat, new_ice_edge_means_flat])
                    combined_ice_edge_stds = np.concatenate([existing_ice_edge_stds_flat, new_ice_edge_stds_flat])
                    combined_ice_edge_n_obs = np.concatenate([existing_ice_edge_n_obs_flat, new_ice_edge_n_obs_flat])
                except (ValueError, TypeError) as concat_e:
                    print(f"Warning: Ice edge data concatenation issue: {concat_e}")
                    # Use only new data for ice edge
                    combined_ice_edge_means = np.asarray(new_ice_edge_means).flatten()
                    combined_ice_edge_stds = np.asarray(new_ice_edge_stds).flatten()
                    combined_ice_edge_n_obs = np.asarray(new_ice_edge_n_obs).flatten()

        except Exception as e:
            print(f"Warning: Error handling ice edge data: {e}")

        # Sort by time to maintain chronological order
        try:
            time_indices = np.argsort([t.timestamp() for t in combined_times])
            sorted_times = [combined_times[i] for i in time_indices]
            sorted_means = combined_means[time_indices]
            sorted_stds = combined_stds[time_indices]
            sorted_n_obs = combined_n_obs[time_indices]

            sorted_ice_edge_means = None
            sorted_ice_edge_stds = None
            sorted_ice_edge_n_obs = None
            if combined_ice_edge_means is not None:
                sorted_ice_edge_means = combined_ice_edge_means[time_indices]
                sorted_ice_edge_stds = combined_ice_edge_stds[time_indices]
                sorted_ice_edge_n_obs = combined_ice_edge_n_obs[time_indices]
        except ValueError as e:
            if "truth value of an array" in str(e):
                print(f"Warning: Array indexing issue during sorting, using unsorted data: {e}")
                # Fall back to unsorted data
                sorted_times = combined_times
                sorted_means = combined_means
                sorted_stds = combined_stds
                sorted_n_obs = combined_n_obs
                sorted_ice_edge_means = combined_ice_edge_means
                sorted_ice_edge_stds = combined_ice_edge_stds
                sorted_ice_edge_n_obs = combined_ice_edge_n_obs
            else:
                raise e

        # Save the combined data back to file
        save_statistics_to_netcdf(filename, sorted_times, sorted_means, sorted_stds, sorted_n_obs,
                                  varname, experiment_id, sorted_ice_edge_means,
                                  sorted_ice_edge_stds, sorted_ice_edge_n_obs)
        return True

    except Exception as e:
        print(f"Error appending to statistics file {filename}: {e}")
        return False


def find_missing_time_periods(existing_times, available_ioda_times, time_interval):
    """Find time periods that exist in IODA data but are missing from existing statistics.

    Args:
        existing_times: List of datetime objects from existing statistics
        available_ioda_times: List of datetime objects from available IODA data
        time_interval: Time interval in seconds

    Returns:
        List of datetime objects representing missing time periods
    """
    if not existing_times:
        return available_ioda_times

    # Convert to sets for efficient comparison
    # Round times to nearest interval to handle small time differences
    def round_to_interval(dt, interval_seconds):
        timestamp = dt.timestamp()
        rounded_timestamp = round(timestamp / interval_seconds) * interval_seconds
        return datetime.fromtimestamp(rounded_timestamp)

    existing_rounded = {round_to_interval(t, time_interval) for t in existing_times}
    available_rounded = {round_to_interval(t, time_interval) for t in available_ioda_times}

    # Find missing periods
    missing_periods = available_rounded - existing_rounded

    return sorted(list(missing_periods))


def format_basin_info(ocean_basins):
    """Format ocean basin information for display."""
    if not ocean_basins:
        return "Global"

    basin_names = {1: 'Atlantic', 2: 'Pacific', 3: 'Indian', 4: 'Arctic', 5: 'Southern'}

    if len(ocean_basins) == 1:
        return basin_names.get(ocean_basins[0], f'Basin {ocean_basins[0]}')
    elif len(ocean_basins) < 5:
        basin_list = [basin_names.get(b, f'Basin {b}') for b in ocean_basins]
        return ', '.join(basin_list)
    else:
        return "Global"  # If all or most basins, assume global

def validate_comparable_obs_spaces(experiments_data):
    """Group observation spaces by name for plotting."""
    obs_space_groups = {}

    for exp_name, obs_spaces in experiments_data.items():
        for obs_space_config, data in obs_spaces:
            obs_space_name = obs_space_config['name']
            if obs_space_name not in obs_space_groups:
                obs_space_groups[obs_space_name] = []
            obs_space_groups[obs_space_name].append({
                'experiment': exp_name,
                'config': obs_space_config,
                'data': data
            })

    return obs_space_groups


def plot_timeseries_multi_experiment(experiments_data, output_dir=None, title_prefix=None):
    """Create comparison plots showing multiple experiments on the same figure.

    Args:
        experiments_data: Dictionary where keys are experiment names and values are lists of (config, data) tuples
        output_dir: Directory to save plots
        title_prefix: Prefix for plot titles
    """
    if output_dir:
        abs_output_dir = os.path.abspath(output_dir)
        if not os.path.exists(abs_output_dir):
            os.makedirs(abs_output_dir, exist_ok=True)

    # Group observation spaces by name for plotting
    validated_obs_spaces = validate_comparable_obs_spaces(experiments_data)

    # Create comparison plots for each validated observation space
    colors = plt.cm.Set1(np.linspace(0, 1, len(experiments_data)))

    for obs_space_name, configs in validated_obs_spaces.items():
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        variable_name = None
        plot_variable = None
        basin_info = None

        for i, config_data in enumerate(configs):
            exp_name = config_data['experiment']
            data = config_data['data']

            if data is None:
                print(f"Warning: No data available for {exp_name} - {obs_space_name}")
                continue

            times = data['times']
            color = colors[i % len(colors)]

            # Plot mean
            ax1.plot(times, data['mean'], color=color, marker='o',
                     markersize=3, linewidth=1.5, label=exp_name)

            # Plot standard deviation
            ax2.plot(times, data['std'], color=color, marker='s',
                     markersize=3, linewidth=1.5, label=exp_name)

            # Plot observation count
            ax3.plot(times, data['n_obs'], color=color, marker='^',
                     markersize=3, linewidth=1.5, label=exp_name)

            if variable_name is None:
                variable_name = data['variable']
                plot_variable = data.get('geovar_group', variable_name)
                basin_info = format_basin_info(data.get('ocean_basins'))

        # Set labels and titles
        ax1.set_ylabel(f"Mean {plot_variable} ({variable_name})", fontsize=12)
        ax1.grid(True, alpha=0.3)
        #ax1.set_title(f"{obs_space_name} - Mean {plot_variable}", fontsize=10)
        ax1.legend(fontsize=10)

        ax2.set_ylabel(f"Std {plot_variable} ({variable_name})", fontsize=12)
        ax2.grid(True, alpha=0.3)
        #ax2.set_title(f"{obs_space_name} - Standard Deviation {plot_variable}", fontsize=10)
        ax2.legend(fontsize=10)

        ax3.set_ylabel("Observation Count", fontsize=12)
        ax3.set_xlabel("Time", fontsize=12)
        ax3.grid(True, alpha=0.3)
        #ax3.set_title(f"{obs_space_name} - Observation Count", fontsize=10)
        ax3.legend(fontsize=10)

        # Format x-axis
        fig.autofmt_xdate()

        # Add overall title with basin information
        fig.suptitle(f"{obs_space_name} - {plot_variable} ({variable_name})", fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save plot
        if output_dir:
            safe_name = "".join(c for c in obs_space_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            abs_output_dir = os.path.abspath(output_dir)
            output_file = os.path.join(abs_output_dir, f"{safe_name}_comparison.png")
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def process_observation_space(obs_config, exp_name=None):
    """Process a single observation space configuration and return data."""
    obs_name = obs_config['name']
    obs_file = obs_config.get('file')

    # New: Support for IODA data processing
    ioda_data_path = obs_config.get('ioda_data_path')
    varname = obs_config.get('varname', 'sst')
    geovar_group = obs_config.get('geovar_group', 'ObsValue')
    time_interval = obs_config.get('time_interval', 3600)
    ice_edge_stats = obs_config.get('ice_edge_stats', False)
    depth_bins = obs_config.get('depth_bins')
    # Handle legacy depth_min/depth_max format for backward compatibility
    if not depth_bins and 'depth_min' in obs_config and 'depth_max' in obs_config:
        depth_bins = [[obs_config['depth_min'], obs_config['depth_max']]]

    ocean_basins = obs_config.get('ocean_basins')
    force_regenerate = obs_config.get('force_regenerate', False)  # Option to force full regeneration

    # Determine ocean basin description for display
    basin_description = ""
    if ocean_basins:
        basin_names = {1: 'Atlantic', 2: 'Pacific', 3: 'Indian', 4: 'Arctic', 5: 'Southern'}
        if len(ocean_basins) == 1:
            basin_description = f" - {basin_names.get(ocean_basins[0], f'Basin {ocean_basins[0]}')}"
        elif len(ocean_basins) < 5:
            basin_list = [basin_names.get(b, f'Basin {b}') for b in ocean_basins]
            basin_description = f" - {', '.join(basin_list)}"
        # If all basins (5 or more), don't add basin description (assumed global)

    data = None

    # Check if we need to update existing statistics or generate new ones
    if ioda_data_path:
        # Generate output filename for statistics
        stats_output = obs_config.get('stats_output')
        if not stats_output and obs_file:
            # Use the intended stats file location
            stats_output = obs_file
        elif not stats_output:
            # Generate a default filename
            safe_name = "".join(c for c in obs_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            stats_output = f"{safe_name}_{varname}_statistics.nc"

        # Ensure output directory exists for statistics files
        if stats_output:
            stats_output_dir = os.path.dirname(os.path.abspath(stats_output))
            if stats_output_dir and not os.path.exists(stats_output_dir):
                os.makedirs(stats_output_dir, exist_ok=True)
                print(f"✅ Created output directory: {stats_output_dir}")

        # Try to load existing statistics file first (check both obs_file and stats_output)
        stats_file_to_load = obs_file or stats_output

        if stats_file_to_load:
            # Support glob patterns for files
            if '*' in stats_file_to_load:
                files = glob(stats_file_to_load)
                if files:
                    stats_file_to_load = files[0]  # Take first match
                else:
                    stats_file_to_load = None

            if stats_file_to_load and os.path.exists(stats_file_to_load) and not force_regenerate:
                print(f"Loading existing statistics: {obs_name} from {stats_file_to_load}")
                data = load_statistics_file(stats_file_to_load)
            elif stats_file_to_load and os.path.exists(stats_file_to_load) and force_regenerate:
                print(f"Force regenerate enabled - ignoring existing statistics file: {stats_file_to_load}")
    else:
        # Legacy support: try to load from obs_file if no IODA processing
        if obs_file:
            # Support glob patterns for files
            if '*' in obs_file:
                files = glob(obs_file)
                if files:
                    obs_file = files[0]  # Take first match
                else:
                    obs_file = None

            if obs_file and os.path.exists(obs_file) and not force_regenerate:
                print(f"Loading existing statistics: {obs_name} from {obs_file}")
                data = load_statistics_file(obs_file)
            elif obs_file and os.path.exists(obs_file) and force_regenerate:
                print(f"Force regenerate enabled - ignoring existing statistics file: {obs_file}")

    # Process IODA data if configured
    if ioda_data_path:

        experiment_id = f"{exp_name} - {obs_name}" if exp_name else obs_name

        if data:
            # We have existing statistics, check for missing periods
            print("Checking for missing time periods in existing statistics...")

            # Get available IODA times
            netcdf_files = glob(ioda_data_path)
            if netcdf_files:
                available_ioda_times = []
                for netcdf_file in netcdf_files:
                    try:
                        iodaData = load_ioda_diags(netcdf_file, varname, geovar_group)
                        # Check if the returned data is empty (indicates file had issues)
                        if len(iodaData.time_data) == 0:
                            continue

                        # Safely handle the time data - it could be a list or numpy array
                        if hasattr(iodaData, 'time_data') and iodaData.time_data is not None:
                            if isinstance(iodaData.time_data, (list, tuple)):
                                available_ioda_times.extend(iodaData.time_data)
                            else:
                                # Handle numpy array case
                                try:
                                    if len(iodaData.time_data) > 0:
                                        available_ioda_times.extend(list(iodaData.time_data))
                                except (TypeError, ValueError) as array_error:
                                    print(f"Warning: Could not process time data from {netcdf_file}: {array_error}")
                                    continue
                    except Exception as e:
                        print(f"Warning: Failed to read times from {netcdf_file}: {e}")
                        continue

                if available_ioda_times:
                    # Find missing periods
                    missing_periods = find_missing_time_periods(data['times'], available_ioda_times, time_interval)

                    if missing_periods:
                        print(f"Found {len(missing_periods)} missing time periods. Generating statistics for missing data...")

                        # Generate statistics only for missing periods
                        new_data = generate_statistics_from_ioda_periods(
                            data_path=ioda_data_path,
                            varname=varname,
                            geovar_group=geovar_group,
                            time_interval=time_interval,
                            experiment_id=experiment_id,
                            missing_periods=missing_periods,
                            ice_edge_stats=ice_edge_stats,
                            depth_bins=depth_bins,
                            ocean_basins=ocean_basins
                        )

                        if new_data and new_data['times']:
                            # Append new statistics to existing file
                            success = append_statistics_to_netcdf(
                                stats_output,
                                new_data['times'],
                                new_data['mean'],
                                new_data['std'],
                                new_data['n_obs'],
                                varname,
                                experiment_id,
                                new_data.get('ice_edge_means'),
                                new_data.get('ice_edge_stds'),
                                new_data.get('ice_edge_n_obs')
                            )

                            if success:
                                print(f"Successfully appended {len(new_data['times'])} new time periods to {stats_output}")
                                # Reload the updated statistics file
                                data = load_statistics_file(stats_output)
                            else:
                                print("Failed to append new statistics")
                    else:
                        print("No missing time periods found. Using existing statistics.")
        else:
            # No existing statistics file, generate from scratch
            print("Statistics file not found or failed to load. Generating from IODA files...")

            data = generate_statistics_from_ioda(
                data_path=ioda_data_path,
                varname=varname,
                geovar_group=geovar_group,
                time_interval=time_interval,
                experiment_id=experiment_id,
                output_file=stats_output,
                ice_edge_stats=ice_edge_stats,
                depth_bins=depth_bins,
                ocean_basins=ocean_basins
            )

    if data:
        # Override experiment ID with config name if provided, include basin info
        full_obs_name = f"{obs_name}{basin_description}"
        data['experiment_id'] = f"{exp_name} - {full_obs_name}" if exp_name else full_obs_name
        data['obs_space_name'] = obs_name  # Keep base observation space name for comparison
        data['geovar_group'] = geovar_group  # Store the geovar_group for display
        data['ocean_basins'] = ocean_basins  # Store basin info for reference
        return data
    else:
        print(f"Failed to load or generate data for observation space: {obs_name}")
        print(f"  - Statistics file: {obs_file}")
        print(f"  - IODA data path: {ioda_data_path}")
        return None


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
            # Check if the returned data is empty (indicates file had issues)
            if len(iodaData.time_data) > 0:
                all_iodaData.append(iodaData)
            else:
                print(f"Warning: No valid data in {netcdf_file} (skipping file)")
        except Exception as e:
            # Handle the specific "truth value of an array" error gracefully
            if "truth value of an array" in str(e):
                print(f"Warning: Failed to load {netcdf_file}: Array shape mismatch (skipping file)")
                print(f"  Detailed error: {e}")
            else:
                print(f"Warning: Failed to load {netcdf_file}: {e}")
            continue

    if len(all_iodaData.time_data) == 0:
        print("Error: No valid data loaded")
        return None

    # Convert to arrays
    min_time, max_time = all_iodaData.toarray()

    # Check if we have any data
    if min_time is None or max_time is None:
        print(f"WARNING: No observations found in IODA files. All files may be empty or filtered out.")
        return None

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
            'geovar_group': geovar_group,
            'filename': output_file or 'generated'
        }
    else:
        return None


def generate_statistics_from_ioda_periods(data_path, varname, missing_periods, geovar_group='ObsValue',
                                          time_interval=3600, experiment_id='',
                                          ice_edge_stats=False, depth_bins=None, ocean_basins=None):
    """Generate statistics from IODA files for specific missing time periods only.

    Args:
        missing_periods: List of datetime objects representing missing time periods to compute
    """
    print(f"Generating statistics for {len(missing_periods)} missing time periods...")

    # Load IODA files
    netcdf_files = glob(data_path)
    if not netcdf_files:
        print(f"Error: No files found matching pattern: {data_path}")
        return None

    print(f"Found {len(netcdf_files)} IODA files")

    all_iodaData = IODAData()

    # Load files - only load files that contain data for missing periods
    for netcdf_file in netcdf_files:
        print(f"Loading: {os.path.basename(netcdf_file)}")
        try:
            iodaData = load_ioda_diags(netcdf_file, varname, geovar_group)
            # Check if the returned data is empty (indicates file had issues)
            if len(iodaData.time_data) == 0:
                print(f"Warning: No valid data in {netcdf_file} (skipping file)")
                continue

            # Check if this file has data for any of our missing periods
            file_times = iodaData.time_data
            if len(file_times) > 0 and any(
                any(abs((file_time - missing_period).total_seconds()) < time_interval
                    for missing_period in missing_periods)
                for file_time in file_times
            ):
                all_iodaData.append(iodaData)
        except Exception as e:
            # Handle the specific "truth value of an array" error gracefully
            if "truth value of an array" in str(e):
                print(f"Warning: Failed to load {netcdf_file}: Array shape mismatch (skipping file)")
                print(f"  Detailed error: {e}")
            else:
                print(f"Warning: Failed to load {netcdf_file}: {e}")
            continue

    if len(all_iodaData.time_data) == 0:
        print("Error: No valid data loaded for missing periods")
        return None

    # Convert to arrays
    min_time, max_time = all_iodaData.toarray()

    # Check if we have any data
    if min_time is None or max_time is None:
        print(f"WARNING: No observations found in IODA files. All files may be empty or filtered out.")
        return None

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

    # Calculate statistics only for missing periods
    stats_times = []
    stats_means = []
    stats_stds = []
    stats_n_obs = []

    # Lists for ice-edge statistics (if enabled)
    ice_edge_means = [] if ice_edge_stats and varname == 'icec' else None
    ice_edge_stds = [] if ice_edge_stats and varname == 'icec' else None
    ice_edge_n_obs = [] if ice_edge_stats and varname == 'icec' else None

    print("Computing statistics for missing periods...")

    # Create spatial mask for depth and ocean basin filtering (applied to all time steps)
    spatial_mask = create_spatial_mask(
        depth_range=depth_bins[0] if depth_bins else None,
        basin_list=ocean_basins
    )

    if np.sum(spatial_mask) == 0:
        print("Warning: No observations pass the spatial filtering criteria")
        return None

    print(f"Spatial filtering: {np.sum(spatial_mask)}/{len(spatial_mask)} observations retained")

    # Process only the missing periods
    for current_time in missing_periods:
        # Get data for current time chunk (use a tighter window to avoid mixing cycles)
        window_size = min(3600, time_interval // 2)  # 1 hour or half the interval, whichever is smaller
        time_min = current_time - timedelta(seconds=window_size)
        time_max = current_time + timedelta(seconds=window_size)
        time_mask = (all_iodaData.time_data >= time_min) & (all_iodaData.time_data < time_max)

        # Combine time and spatial masks
        combined_mask = time_mask & spatial_mask

        if np.sum(combined_mask) == 0:
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

    # Return statistics data (don't save to file here, let the caller handle appending)
    if stats_times:
        result = {
            'times': stats_times,
            'mean': np.array(stats_means),
            'std': np.array(stats_stds),
            'n_obs': np.array(stats_n_obs),
            'experiment_id': experiment_id,
            'variable': varname,
            'geovar_group': geovar_group,
            'filename': 'generated_periods'
        }

        # Add ice edge data if computed
        if ice_edge_means is not None:
            result['ice_edge_means'] = np.array(ice_edge_means)
            result['ice_edge_stds'] = np.array(ice_edge_stds)
            result['ice_edge_n_obs'] = np.array(ice_edge_n_obs)

        return result
    else:
        return None


def create_directories_from_config(config):
    """Create all necessary directories from the parsed configuration."""
    directories = set()

    # Add output directory
    output_dir = config.get('output_dir', './timeseries_plots')
    directories.add(output_dir)

    # Extract directories from observation spaces
    if 'experiments' in config and isinstance(config['experiments'], dict):
        # Multi-experiment format
        for exp_config in config['experiments'].values():
            obs_spaces = exp_config.get('observation_spaces', [])
            for obs_space in obs_spaces:
                stats_output = obs_space.get('stats_output')
                if stats_output:
                    stats_dir = os.path.dirname(stats_output)
                    if stats_dir and stats_dir != '.':
                        directories.add(stats_dir)
    else:
        # Single experiment format
        obs_spaces = config.get('observation_spaces', config.get('experiments', []))
        for obs_space in obs_spaces:
            stats_output = obs_space.get('stats_output')
            if stats_output:
                stats_dir = os.path.dirname(stats_output)
                if stats_dir and stats_dir != '.':
                    directories.add(stats_dir)

    # Create directories
    created_count = 0
    for directory in sorted(directories):
        if directory:
            abs_dir = os.path.abspath(directory)
            if not os.path.exists(abs_dir):
                os.makedirs(abs_dir, exist_ok=True)
                print(f"✅ Created directory: {abs_dir}")
                created_count += 1
            else:
                print(f"📁 Directory exists: {abs_dir}")

    if created_count > 0:
        print(f"🎉 Created {created_count} new directories")


def process_observation_space_groups(grouped_obs_spaces, output_dir, title, enable_plotting=True, skip_plots=False):
    """
    Process each group of comparable observation spaces sequentially.
    This is used for batch processing and multi-experiment plotting.
    """
    total_groups = len(grouped_obs_spaces)
    print(f"Processing {total_groups} observation space groups...")

    all_plot_files = []
    processed_count = 0

    for i, group in enumerate(grouped_obs_spaces, 1):
        try:
            # Progress indicator for large configs
            obs_name = group[0]['name'] if group else f"Group {i}"
            print(f"[{i:3d}/{total_groups}] Processing group: {obs_name}")

            if skip_plots:
                # Just process statistics, no plotting
                for obs_config in group:
                    process_observation_space(obs_config)
                processed_count += 1
            elif enable_plotting:
                # Process the group with multi-experiment plotting
                experiments_data = {}

                # Group by experiment for this observation space
                for obs_config in group:
                    exp_name = obs_config.get('experiment_name', 'Unknown')
                    if exp_name not in experiments_data:
                        experiments_data[exp_name] = []

                    data = process_observation_space(obs_config, exp_name)
                    if data:
                        experiments_data[exp_name].append((obs_config, data))

                # Create multi-experiment plot for this observation space
                if experiments_data:
                    plot_files = plot_timeseries_multi_experiment(experiments_data, output_dir, f"{title} - {obs_name}")
                    all_plot_files.extend(plot_files or [])

                processed_count += 1

            # Memory management for large configs
            if i % 10 == 0:
                print(f"    Progress: {processed_count}/{total_groups} groups completed ({100*processed_count/total_groups:.1f}%)")

        except Exception as e:
            print(f"ERROR processing group {i}: {e}")
            continue

    print(f"Completed processing: {processed_count}/{total_groups} groups successful")
    return all_plot_files


def main():
    parser = argparse.ArgumentParser(
        description='Plot timeseries statistics for different observation spaces in an experiment (optimized with file caching)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_timeseries.py config.yaml
  python plot_timeseries.py config.yaml --skip-plots  # Process statistics only, no plots
  python plot_timeseries.py --help

Config YAML file can use either format:

Single Experiment Format:
  observation_spaces:  # Each observation space gets its own separate plot
    - name: "SST AVHRR-MB ombg"
      file: "/path/to/sst_statistics.nc"          # Existing statistics file
      ioda_data_path: "/path/to/ioda/*.nc"        # IODA files to process if stats missing
      varname: "sst"                              # Variable name for IODA processing
      geovar_group: "ombg"                        # NetCDF group (default: ObsValue)
      time_interval: 21600                        # Time interval in seconds (default: 3600)
      ice_edge_stats: false                       # Enable ice-edge statistics (default: false)
      depth_bins: [[0, 50], [50, 200]]            # Depth ranges in meters (optional)
      ocean_basins: [1, 2, 3]                    # Ocean basin codes (optional)
      stats_output: "/path/to/output_stats.nc"    # Where to save generated statistics
      force_regenerate: false                     # Force full regeneration instead of appending (default: false)
  output_dir: "./timeseries_plots"                # Directory for plots
  title: "Experiment Name"                        # Prefix for plot titles

Multi-Experiment Format (for comparisons):
  experiments:
    "Experiment A":
      observation_spaces:
        - name: "Surface Drifters"
          ioda_data_path: "/path/to/drifter/*.nc"
          varname: "temp"
          geovar_group: "ombg"
    "Experiment B":
      observation_spaces:
        - name: "Surface Drifters"  # Same obs space name for comparison
          ioda_data_path: "/path/to/drifter/*.nc"
          varname: "temp"
          geovar_group: "ombg"
          ocean_basins: [2]  # Different filtering
  output_dir: "./multi_experiment_plots"
  title: "Multi-Experiment Comparison"

Notes:
- Multi-experiment format creates comparison plots showing multiple experiments
  for each observation space, plus individual plots for each experiment.
- The application automatically detects missing time periods in existing statistics
  files and appends only new data, making incremental updates efficient.
- Use force_regenerate: true to bypass append functionality and regenerate all statistics.
- File caching provides 5-50x speedup by avoiding duplicate file loading operations.
        '''
    )

    parser.add_argument('config_file',
                        help='YAML configuration file containing experiment files')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip plot generation, only process statistics files')

    args = parser.parse_args()

    # Load configuration
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Create necessary directories from config
    create_directories_from_config(config)

    # Check if this is a multi-experiment configuration
    if 'experiments' in config and isinstance(config['experiments'], dict):
        # Multi-experiment format
        experiments_data = {}

        for exp_name, exp_config in config['experiments'].items():
            print(f"\n🔬 Processing experiment: {exp_name}")

            obs_spaces_config = exp_config.get('observation_spaces', [])

            if not obs_spaces_config:
                print(f"⚠️  No observation spaces found for experiment: {exp_name}")
                continue

            # Process observation spaces sequentially
            processed_data = process_observation_spaces_sequentially(obs_spaces_config, exp_name)
            # Convert to format expected by multi-experiment plotting
            observation_spaces_data = [(obs_spaces_config[i], data)
                                     for i, data in enumerate(processed_data)
                                     if data is not None]

            if observation_spaces_data:
                experiments_data[exp_name] = observation_spaces_data

        # Get output settings
        output_dir = config.get('output_dir', './timeseries_plots')
        title_prefix = config.get('title')

        # Skip plotting if requested
        if not args.skip_plots and experiments_data:
            plot_timeseries_multi_experiment(experiments_data, output_dir, title_prefix)

            # Also create individual plots for each experiment using multi-experiment function
            #print(f"\n🎯 Creating individual plots for each experiment...")
            #for exp_name, obs_space_data in experiments_data.items():
            #    print(f"📊 Creating individual plots for experiment: {exp_name}")
            #    # Create single-experiment data structure for multi-experiment function
            #    single_exp_data = {exp_name: obs_space_data}
            #    plot_timeseries_multi_experiment(single_exp_data, output_dir, f"{title_prefix} - {exp_name}" if title_prefix else exp_name)

#    else:
#        # Single experiment format (backward compatibility)
#        # Support both 'observation_spaces' (new) and 'experiments' (legacy) keys
#        obs_spaces_config = config.get('observation_spaces', config.get('experiments', []))
#
#        if not obs_spaces_config:
#            print("❌ No observation spaces found in configuration")
#            return
#
#        # Process observation spaces sequentially for best performance
#        observation_spaces_data = process_observation_spaces_sequentially(obs_spaces_config)
#
#        if not observation_spaces_data:
#            print("Error: No observation space data loaded successfully")
#            return
#
#        # Get output settings - now using output_dir instead of single output_file
#        output_dir = config.get('output_dir', './timeseries_plots')
#        title_prefix = config.get('title')
#
#        # Skip plotting if requested
#        if args.skip_plots:
#            print(f"\n🚫 Skipping plot generation (--skip-plots flag enabled)")
#            print(f"✅ Statistics processing complete for {len(observation_spaces_data)} observation spaces")
#        else:
#            # Create plots using multi-experiment function (single-experiment case)
#            # Convert to multi-experiment format
#            single_exp_name = title_prefix or "Single Experiment"
#            single_exp_data = {single_exp_name: [({"name": data["obs_space_name"]}, data) for data in observation_spaces_data]}
#            plot_timeseries_multi_experiment(single_exp_data, output_dir, title_prefix)

    print("Processing complete.")

if __name__ == "__main__":
    main()
