import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from datetime import datetime, timedelta
from glob import glob
import yaml
import sys


VARIABLE_MAP = {'sst': 'seaSurfaceTemperature',
                'sss': 'seaSurfaceSalinity',
                'adt': 'absoluteDynamicTopography',
                'icec': 'seaIceFraction'}


class IODAData:
    def __init__(self, time_data=[], lat=[], lon=[], geovar=[], varname=''):
        self.time_data = time_data
        self.lat = lat
        self.lon = lon
        self.geovar = geovar
        self.varname = varname

    def __str__(self):
        return f"IODAData(time_data={self.time_data}, lat={self.lat}, lon={self.lon}, geovar={self.geovar})"

    def __repr__(self):
        return self.__str__()

    def append(self, other):
        # Assert that self.varname == other.varname
        print(f"--------- len(other.time_data): {len(other.time_data)}")
        print(f"--------- {self.varname}    {other.varname}")
        if len(self.time_data) > 0:
            assert self.varname == other.varname
        if len(self.time_data) == 0:
            self.varname = other.varname
        print(f"--------- {self.varname}    {other.varname}")
        self.time_data.append(other.time_data)
        self.lat.append(other.lat)
        self.lon.append(other.lon)
        self.geovar.append(other.geovar)

    def toarray(self):
        # Concatenate data from all files
        self.time_data = np.concatenate(self.time_data)
        self.lat = np.concatenate(self.lat)
        self.lon = np.concatenate(self.lon)
        self.geovar = np.concatenate(self.geovar)
        min_time, max_time = min(self.time_data), max(self.time_data)
        return min_time, max_time


def load_ioda_diags(netcdf_file):
    # Open NetCDF file
    ds = nc.Dataset(netcdf_file, 'r')

    # extract the variable name from the file name
    var_name_short = netcdf_file.split('_')[0]
    var_name = VARIABLE_MAP[var_name_short]
    print(f"----------- Processing variable: {var_name}")

    # Read necessary data
    time_data = ds.groups['MetaData'].variables['dateTime'][:]
    ref_time_str = ds.groups['MetaData'].variables['dateTime'].getncattr('units').split(' ')[-1]
    ref_time = datetime.strptime(ref_time_str, "%Y-%m-%dT%H:%M:%SZ")
    print(f"Reference time: {ref_time}")
    lat = ds.groups['MetaData'].variables['latitude'][:]
    lon = ds.groups['MetaData'].variables['longitude'][:]
    geovar = ds.groups['ObsValue'].variables[var_name][:]
    qc = ds.groups['EffectiveQC0'].variables[var_name][:]

    # Remove fill values
    valid_mask = (geovar > -3.368795e+38) & (lat > -3.368795e+38) & (lon > -3.368795e+38) & (qc == 0)
    time_data, lat, lon, geovar = time_data[valid_mask], lat[valid_mask], lon[valid_mask], geovar[valid_mask]

    # Convert time to datetime objects
    time_data = np.array([ref_time + timedelta(seconds=int(t)) for t in time_data])
    print(f"Time range: {min(time_data)} to {max(time_data)}")
    # Close dataset
    ds.close()

    return IODAData(time_data, lat, lon, geovar, varname=var_name_short)


def plot_data(iodaData, frame_idx, time_interval=300, save_dir='./frames', bounds=[-2, 35]):
    min_time, max_time = iodaData.toarray()
    current_time = min_time

    while current_time <= max_time:
        next_time = current_time + timedelta(seconds=time_interval)
        print(f"Processing: {current_time} to {next_time}")

        # Get data for current time chunk
        time_min = current_time - timedelta(seconds=3 * time_interval)
        time_max = current_time + timedelta(seconds=3 * time_interval)
        time_mask = (iodaData.time_data >= time_min) & (iodaData.time_data < time_max)
        if np.sum(time_mask) == 0:
            current_time = next_time
            continue  # Skip empty time bins

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, color='lightgray')

        # ax.set_extent([-85, -60, 25, 50], crs=ccrs.PlateCarree())
        sc = ax.scatter(iodaData.lon[time_mask], iodaData.lat[time_mask], c=iodaData.geovar[time_mask], cmap='gist_ncar',
                        s=0.1, transform=ccrs.PlateCarree(), vmin=bounds[0], vmax=bounds[1])

        plt.colorbar(sc, ax=ax, orientation='vertical', label=f'{iodaData.varname}')
        plt.title(
            f"{iodaData.varname} from {current_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"to {next_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        # Save figure
        frame_filename = os.path.join(save_dir, f"{iodaData.varname}_frame_{frame_idx:04d}.png")
        plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {frame_filename}")

        # Update time and frame index
        current_time = next_time
        frame_idx += 1


def main():
    if len(sys.argv) != 2:
        print("Usage: plt_diags_maps.py <config.yaml>")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    yyyymm = config['yyyymm']
    dd = config['dd']
    cyc = config['cyc']
    time_interval = config['time_interval']
    save_dir = config['save_dir']
    varname = config['varname']
    bounds = config['bounds']

    os.makedirs(save_dir, exist_ok=True)
    netcdf_files = glob(f"{varname}_*.{yyyymm}{dd}{cyc}.nc4")

    all_iodaData = IODAData()

    for netcdf_file in netcdf_files:
        iodaData = load_ioda_diags(netcdf_file)
        all_iodaData.append(iodaData)

    # Plot data
    plot_data(all_iodaData, frame_idx=0, time_interval=time_interval, save_dir=save_dir, bounds=bounds)


if __name__ == "__main__":
    main()
