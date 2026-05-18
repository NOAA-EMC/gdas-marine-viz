#!/usr/bin/env python3


# creates figures of timeseries from the csv outputs computed by gdassoca_obsstats.x
import argparse
from itertools import product
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import netCDF4 as nc

colors = [
    "lightsteelblue",
    "lightgreen",
    "peachpuff",
    "lightpink",
    "lightgoldenrodyellow",
    "paleturquoise",
    "lightcoral",
    "palegreen",
    "palegoldenrod",
    "mistyrose",
    "lavender",
    "lightsalmon",
]


def get_inst(file_name):
    """Extract the instrument name from file name.
    For CSV: gdas.t00z.ocn.sst_ahi_h08_l3c.stats.csv -> sst_ahi_h08_l3c
    For NetCDF: insitu_temp_profile_argo.nc -> insitu_temp_profile_argo
    """
    if file_name.endswith('.csv'):
        return file_name.split('.')[-3].split('/')[-1]
    elif file_name.endswith('.nc'):
        return os.path.basename(file_name).replace('.nc', '')
    else:
        return os.path.basename(file_name)


def get_ocean_basin_name(basin_code):
    """Convert ocean basin code to name"""
    basin_map = {
        0: 'Arctic',
        1: 'Atlantic',
        2: 'Indian',
        3: 'Pacific',
        4: 'Southern',
        5: 'Global'  # or other codes
    }
    return basin_map.get(basin_code, f'Basin_{basin_code}')


def get_depth_layer_name(depth):
    """Get depth layer name based on depth value"""
    if depth <= 10:
        return '0-10m'
    elif depth <= 50:
        return '10-50m'
    elif depth <= 100:
        return '50-100m'
    elif depth <= 200:
        return '100-200m'
    elif depth <= 500:
        return '200-500m'
    else:
        return '>500m'


class ObsStats:
    def __init__(self):
        self.data = pd.DataFrame()

    def read_csv(self, filepaths):
        # Iterate through the list of file paths and append their data
        for filepath in filepaths:
            new_data = pd.read_csv(filepath)

            # Convert date to datetime for easier plotting
            new_data['date'] = pd.to_datetime(new_data['date'], format='%Y%m%d%H')
            self.data = pd.concat([self.data, new_data], ignore_index=True)
            self.data.sort_values('date', inplace=True)

    def read_netcdf(self, filepaths, exp_name, variable_name, depth_layers=None):
        """Read NetCDF files and compute statistics by depth layer and ocean basin"""
        if depth_layers is None:
            depth_layers = [(0, 10)]  # Default to 0-10m

        all_stats = []

        for filepath in filepaths:
            print(f"Processing NetCDF: {filepath}")

            # Extract date from filepath (assuming format includes date)
            # e.g., gdas.20250820/06/analysis/ocean/diags/insitu_temp_profile_argo.nc
            try:
                path_parts = filepath.split('/')
                for part in path_parts:
                    if (part.startswith('gdas.')  or part.startswith('enkfgdas.')) and len(part) >= 13:
                        date_str = part.split('.')[1]  # Extract YYYYMMDD
                        hour_part = None
                        # Look for hour in next parts
                        idx = path_parts.index(part)
                        if idx + 1 < len(path_parts):
                            hour_part = path_parts[idx + 1]

                        if hour_part and hour_part.isdigit():
                            date_str += hour_part.zfill(2)
                        else:
                            date_str += '00'  # Default hour

                        file_date = pd.to_datetime(date_str, format='%Y%m%d%H')
                        break
                else:
                    print(f"Could not extract date from {filepath}, skipping...")
                    continue

            except Exception as e:
                print(f"Error extracting date from {filepath}: {e}")
                continue

            try:
                with nc.Dataset(filepath, 'r') as dataset:
                    # Read metadata
                    depths = dataset.groups['MetaData']['depth'][:]
                    ocean_basins = dataset.groups['MetaData']['oceanBasin'][:]

                    # Read observation data
                    if variable_name in ['waterTemperature', 'seaSurfaceTemperature', 'salinity']:
                        # Check if variable exists in the file
                        if variable_name not in dataset.groups['ObsValue'].variables:
                            print(f"{variable_name} not found in /ObsValue")
                            continue

                        obs_values = dataset.groups['ObsValue'][variable_name][:]
                        #hofx_values = dataset.groups['hofx0'][variable_name][:]  # background ; letkf only has this for each ens mem, not for ensmean
                        obs_errors = dataset.groups['ObsError'][variable_name][:]

                        ombg = dataset.groups['ombg'][variable_name][:] ## this works for both 3dvar and letkf;  can use OMAN or OMBG (lowercase)

                        # Quality control
                        if 'EffectiveQC0' in dataset.groups and variable_name in dataset.groups['EffectiveQC0'].variables:
                            qc_flags = dataset.groups['EffectiveQC0'][variable_name][:]
                        else:
                            qc_flags = np.zeros_like(obs_values)
                    else:
                        print(f"Unknown variable: {variable_name}")
                        continue

                    # Calculate observation minus background
                    #ombg = obs_values - hofx_values ## this does not work for LETKF

                    # Process each depth layer and ocean basin combination
                    for depth_min, depth_max in depth_layers:
                        # Filter for valid depths and remove fill values
                        valid_depth_mask = (depths >= depth_min) & (depths <= depth_max) & np.isfinite(depths)

                        # Get unique valid basin codes
                        valid_basins = ocean_basins[np.isfinite(ocean_basins) & (ocean_basins >= 0)]
                        unique_basins = np.unique(valid_basins)

                        for basin_code in unique_basins:
                            if np.isnan(basin_code) or basin_code < 0:
                                continue

                            basin_mask = (ocean_basins == basin_code) & np.isfinite(ocean_basins)
                            combined_mask = valid_depth_mask & basin_mask

                            n_obs = np.sum(combined_mask)
                            if n_obs == 0:
                                continue

                            # Optional: print observation counts
                            # print(f"  Found {n_obs} observations for basin {basin_code} "
                            #       f"({get_ocean_basin_name(int(basin_code))}) in {depth_min}-{depth_max}m")

                            # Extract data for this combination
                            ombg_subset = ombg[combined_mask]
                            obs_err_subset = obs_errors[combined_mask]
                            qc_subset = qc_flags[combined_mask]

                            # Remove fill values and invalid data
                            valid_mask = (~np.isnan(ombg_subset)
                                          & ~np.isnan(obs_err_subset)
                                          & np.isfinite(ombg_subset)
                                          & np.isfinite(obs_err_subset))

                            if np.sum(valid_mask) == 0:
                                continue

                            ombg_valid = ombg_subset[valid_mask]
                            obs_err_valid = obs_err_subset[valid_mask]
                            qc_valid = qc_subset[valid_mask]

                            # Compute statistics for all data (no QC)
                            rmse_noqc = np.sqrt(np.mean(ombg_valid**2))
                            bias_noqc = np.mean(ombg_valid)
                            count_noqc = len(ombg_valid)
                            obs_err_mean_noqc = np.mean(obs_err_valid)

                            # Compute statistics for QC'd data (assuming QC=0 is good)
                            qc_good_mask = qc_valid == 0
                            if np.sum(qc_good_mask) > 0:
                                ombg_qc = ombg_valid[qc_good_mask]
                                obs_err_qc = obs_err_valid[qc_good_mask]
                                rmse_qc = np.sqrt(np.mean(ombg_qc**2))
                                bias_qc = np.mean(ombg_qc)
                                count_qc = len(ombg_qc)
                                obs_err_mean_qc = np.mean(obs_err_qc)
                            else:
                                rmse_qc = bias_qc = count_qc = obs_err_mean_qc = np.nan

                            # Create records for both QC and no-QC
                            basin_name = get_ocean_basin_name(int(basin_code))
                            depth_layer = f"{int(depth_min)}-{int(depth_max)}m"

                            # No QC record
                            all_stats.append({
                                'Exp': exp_name,
                                'Variable': 'ombg_noqc',
                                'Ocean': basin_name,
                                'DepthLayer': depth_layer,
                                'date': file_date,
                                'RMSE': rmse_noqc,
                                'Bias': bias_noqc,
                                'ObsErr': obs_err_mean_noqc,
                                'EnsStd': 0.0,  # Not available in this format
                                'Count': count_noqc
                            })

                            # QC record
                            all_stats.append({
                                'Exp': exp_name,
                                'Variable': 'ombg_qc',
                                'Ocean': basin_name,
                                'DepthLayer': depth_layer,
                                'date': file_date,
                                'RMSE': rmse_qc,
                                'Bias': bias_qc,
                                'ObsErr': obs_err_mean_qc,
                                'EnsStd': 0.0,  # Not available in this format
                                'Count': count_qc
                            })

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

        # Convert to DataFrame and combine with existing data
        if all_stats:
            new_data = pd.DataFrame(all_stats)
            self.data = pd.concat([self.data, new_data], ignore_index=True)
            self.data.sort_values('date', inplace=True)

    def plot_timeseries(self, ocean, variable, inst="", dirout="", depth_layer="0-10m"):

        # Filter data for the given ocean, variable, and depth layer
        # Check if data exists
        if self.data.empty:
            print("No data loaded.")
            return []

        if 'DepthLayer' in self.data.columns:
            filtered_data = self.data[(self.data['Ocean'] == ocean)
                                      & (self.data['Variable'] == variable)
                                      & (self.data['DepthLayer'] == depth_layer)]
        else:
            # Fallback for CSV data without depth layers
            filtered_data = self.data[(self.data['Ocean'] == ocean) & (self.data['Variable'] == variable)]
        if filtered_data.empty:
            print("No data available for the given ocean and variable combination.")
            return []

        # Get unique experiments
        experiments = filtered_data['Exp'].unique()
        #experiments.sort()
        print(experiments)

        # Plot settings
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        if 'DepthLayer' in self.data.columns:
            fig.suptitle(f'{inst} {variable} statistics, {ocean} ocean, {depth_layer}',
                         fontsize=18, fontweight='bold')
        else:
            fig.suptitle(f'{inst} {variable} statistics, {ocean} ocean', fontsize=18, fontweight='bold')
        exp_counter = 0
        for exp in experiments:
            exp_data = self.data[
                (self.data['Ocean'] == ocean)
                & (self.data['Variable'] == variable)
                & (self.data['Exp'] == exp)
            ]

            # Plot RMSE, obs error, obs error + spread
            axs[0].plot(exp_data['date'], exp_data['RMSE'], marker='o', linestyle='-',
                        color=colors[exp_counter], linewidth=2, label='RMSE ' + exp)
            #if (exp.endswith("letkf")): ## comment for now
            #    axs[0].plot(exp_data['date'], exp_data['EnsStd'] + exp_data['ObsErr'], marker='x', linestyle='--',
            #                color=colors[exp_counter], linewidth=2, label='EnsStd+ObsErr ' + exp)
            #if (exp.endswith("letkf")):
            #    axs[0].plot(exp_data['date'], exp_data['EnsStd'], marker='s', linestyle='-',
            #                color=colors[exp_counter], linewidth=2, label='EnsStd ' + exp)
            axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
            axs[0].xaxis.set_major_locator(mdates.DayLocator())
            axs[0].tick_params(labelbottom=False)
            axs[0].set_ylabel('RMSE', fontsize=18, fontweight='bold')
            axs[0].legend()
            axs[0].grid(True)

            # Plot Bias
            axs[1].plot(exp_data['date'], exp_data['Bias'], marker='o', linestyle='-',
                        color=colors[exp_counter], linewidth=2, label=exp)
            axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
            axs[1].xaxis.set_major_locator(mdates.DayLocator())
            axs[1].tick_params(labelbottom=False)
            axs[1].set_ylabel('Bias', fontsize=18, fontweight='bold')
            axs[1].grid(True)

            # Plot Count
            axs[2].plot(exp_data['date'], exp_data['Count'], marker='o', linestyle='-',
                        color=colors[exp_counter], linewidth=2, label=exp)
            axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
            axs[2].xaxis.set_major_locator(mdates.DayLocator())
            axs[2].set_ylabel('Count', fontsize=18, fontweight='bold')
            axs[2].grid(True)

            exp_counter += 1

        # Improve layout and show plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if 'DepthLayer' in self.data.columns:
            depth_suffix = depth_layer.replace('-', '_')
            plt.savefig(f'{dirout}/{inst}_{variable}_{ocean}_{depth_suffix}.png')
        else:
            plt.savefig(f'{dirout}/{inst}_{variable}_{ocean}.png')
        # close the figure
        plt.close(fig)

        return experiments


if __name__ == "__main__":
    epilog = [
        "Usage examples:",
        "# CSV mode (default):",
        "./gdassoca_obsstats.py --exps cp1/COMROOT/cp1 --dirout output",
        "./gdassoca_obsstats.py --exps cp1/COMROOT/cp1 --inst sst_abi_g16_l3c --dirout output",
        "# NetCDF mode with depth layers:",
        "./gdassoca_obsstats.py --source netcdf --exps cp1/COMROOT/cp1 --inst 'insitu_temp*' --dirout output",
        "./gdassoca_obsstats.py --source netcdf --depth-layers 0-10 10-50 --dirout output"
    ]
    parser = argparse.ArgumentParser(description="Observation space RMSE's and BIAS's",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=os.linesep.join(epilog))
    parser.add_argument("--exps", nargs='+', required=True,
                        help="Path to the experiment's COMROOT")
    parser.add_argument(
        "--inst",
        required=False,
        help="Optional: The name of the instrument/platform (ex: sst_abi_g16_l3c) or a wild card "
             "(eg sst*). If not provided, all available instruments will be processed."
    )
    parser.add_argument("--dirout", required=True, help="Output directory")
    parser.add_argument("--letkf", action="store_true", help="Generate stats for LETKF diag files")
    parser.add_argument("--both", action="store_true", help="Generate stats for both 3DVar and LETKF diag files")
    parser.add_argument("--source", choices=['csv', 'netcdf'], default='csv',
                        help="Data source: 'csv' for pre-computed stats or 'netcdf' for original data")
    parser.add_argument("--depth-layers", nargs='*', default=['0-10'],
                        help="Depth layers to process (format: 'min-max', e.g., '0-10' '10-50')")
    args = parser.parse_args()

    insts = []
    inst = args.inst if args.inst else "*"  # Use wildcard if no instrument specified
    os.makedirs(args.dirout, exist_ok=True)

    # Parse depth layers
    depth_layers = []
    for layer_str in args.depth_layers:
        try:
            min_depth, max_depth = map(float, layer_str.split('-'))
            depth_layers.append((min_depth, max_depth))
        except ValueError:
            print(f"Warning: Invalid depth layer format '{layer_str}'. Expected 'min-max'.")

    if not depth_layers:
        depth_layers = [(0, 10)]  # Default

    # Get all instruments/obs spaces
    for exp in args.exps:
        if args.source == 'csv':
            wc = exp + f'/*.*/??/analysis/ocean/diags/*{inst}*.stats.csv'
            flist = glob.glob(wc)
            print(f"CSV search: {wc}")
            print(f"Found files: {flist}")
            for fname in flist:
                insts.append(get_inst(fname))
            if (args.letkf):
                wc = exp + f'/*.*/??/analysis/ocean/letkf/diags/*{inst}*.stats.csv'
                flist = glob.glob(wc)
                for fname in flist:
                    insts.append(get_inst(fname))
        else:  # netcdf
            # Look for insitu temperature and salinity NetCDF files
            wc = exp + f'/*.*/??/analysis/ocean/diags/*{inst}*.nc'
            flist = glob.glob(wc)
            print(f"NetCDF search: {wc}")
            print(f"Found files: {flist}")
            for fname in flist:
                inst_name = get_inst(fname)
                # Only process insitu temperature and salinity for now
                if 'insitu_temp' in inst_name or 'insitu_salt' in inst_name:
                    insts.append(inst_name)
    insts = list(set(insts))
    insts.sort()
    print(insts)

    experiments = []
    for inst in insts:
        print(f"Processing {inst}")
        obsStats = ObsStats()
        if args.source == 'csv':
            flist = []
            for exp in args.exps:
                wc = exp + f'/*.*/??/analysis/ocean/diags/*{inst}*.stats.csv'
                flist.append(glob.glob(wc))
                if (args.letkf):
                    wc = exp + f'/*.*/??/analysis/ocean/letkf/diags/*{inst}*.stats.csv'
                    flist.append(glob.glob(wc))
            flist = sum(flist, [])
            obsStats.read_csv(flist)

        else:  # netcdf
            for exp in args.exps:
                # Determine variable name from instrument
                if 'temp_profile' in inst:
                    var_name = 'waterTemperature'
                elif 'temp_surface' in inst:
                    var_name = 'seaSurfaceTemperature'
                elif 'salt' in inst:
                    var_name = 'salinity'
                elif 'sst' in inst:
                    var_name = 'seaSurfaceTemperature' 
                else:
                    print(f"Unknown variable type for {inst}")
                    continue

                if args.both:
                    wc_3dvar = exp + f'/gdas.*/??/analysis/ocean/diags/*{inst}*.nc'
                    exp_name = os.path.basename(exp.rstrip('/')) + '_3dvar'
                    flist = glob.glob(wc_3dvar)
                    obsStats.read_netcdf(flist, exp_name, var_name, depth_layers)

                    wc_letkf = exp + f'/enkfgdas.*/??/ensstat/analysis/ocean/*{inst}*.nc'
                    exp_name = os.path.basename(exp.rstrip('/')) + '_letkf'
                    flist = glob.glob(wc_letkf)
                    print(flist)
                    obsStats.read_netcdf(flist, exp_name, var_name, depth_layers)
                elif args.letkf:
                    wc = exp + f'/enkfgdas.*/??/ensstat/analysis/ocean/*{inst}*.nc'
                    exp_name = os.path.basename(exp.rstrip('/'))
                    flist = glob.glob(wc)
                    obsStats.read_netcdf(flist, exp_name, var_name, depth_layers)
                else: 
                    wc = exp + f'/gdas.*/??/analysis/ocean/diags/*{inst}*.nc'
                    exp_name = os.path.basename(exp.rstrip('/'))
                    flist = glob.glob(wc)
                    obsStats.read_netcdf(flist, exp_name, var_name, depth_layers)

        # Generate plots
        if args.source == 'netcdf':
            # For NetCDF, iterate over depth layers
            for depth_min, depth_max in depth_layers:
                depth_layer = f"{int(depth_min)}-{int(depth_max)}m"
                for var, ocean in product(['ombg_noqc', 'ombg_qc'],
                                          ['Global', 'Atlantic', 'Pacific', 'Indian', 'Arctic', 'Southern']):
                    print(f"OCEAN: {ocean}, DEPTH: {depth_layer}")
                    experiments.extend(obsStats.plot_timeseries(
                        ocean, var, inst=inst, dirout=args.dirout, depth_layer=depth_layer))
        else:
            # For CSV, use original logic
            for var, ocean in product(['ombg_noqc', 'ombg_qc'],
                                      ['Global', 'Atlantic', 'Pacific', 'Indian', 'Arctic', 'Southern']):
                print(f"OCEAN: {ocean}")
                experiments.extend(obsStats.plot_timeseries(ocean, var, inst=inst, dirout=args.dirout))

    # Select unique elements of experiments
    experiments = list(set(experiments))
    experiments.sort()

    print("\nProcessing complete!")
    print(f"Generated plots for {len(insts)} instruments in {args.dirout}")
    print("To create an interactive HTML index, run:")
    print(f"  python generate_index.py {args.dirout} --title 'Custom Title' --experiments 'Experiment Description'")
    if experiments:
        print(f"Experiments processed: {', '.join(experiments)}")
