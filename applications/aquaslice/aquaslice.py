import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from netCDF4 import Dataset
from matplotlib.widgets import RangeSlider
import os


def plot_vertical_profile(ix, iy, lon2d, lat2d, data, depth, ax_profile):
    print(f"Vertical profile at lon={lon2d[iy, ix].values:.2f}, lat={lat2d[iy, ix].values:.2f}")
    print(f"iy, ix: {iy}, {ix}")
    profile = data[:, iy, ix]
    depth_profile = depth[:, iy, ix]
    ax_profile.plot(profile, depth_profile, '-o')
    ax_profile.invert_yaxis()
    ax_profile.set_xlabel('Field Stddev')
    ax_profile.set_ylabel('Depth')
    ax_profile.set_title(f'Vertical Profile at (lon={lon2d[iy, ix].values:.2f}, lat={lat2d[iy, ix].values:.2f})')
    ax_profile.grid()


def load_obsfile(obsfile, longitude_max=None):
    print(f"Loading observation file: {obsfile}")
    print(f"Longitude min: {longitude_max}")
    with Dataset(obsfile, 'r') as f:
        lat_obs = f.groups['MetaData'].variables['latitude'][:]
        lon_obs = f.groups['MetaData'].variables['longitude'][:]
        lon_obs[lon_obs > longitude_max] -= 360.0  # Rotate to the mom6 grid
        depth_obs = f.groups['MetaData'].variables['depth'][:]
        ombg = f.groups['ombg'].variables['waterTemperature'][:]
        oman = f.groups['oman'].variables['waterTemperature'][:]
        obsval = f.groups['ObsValue'].variables['waterTemperature'][:]
        hofx = f.groups['hofx0'].variables['waterTemperature'][:]

        valid = np.isfinite(lat_obs) & np.isfinite(lon_obs) & np.isfinite(depth_obs)
        lat_obs = lat_obs[valid]
        lon_obs = lon_obs[valid]
        depth_obs = depth_obs[valid]
        ombg = ombg[valid]
        oman = oman[valid]
        obsval = obsval[valid]
        hofx = hofx[valid]

        obs = {'lon': lon_obs,
               'lat': lat_obs,
               'depth': depth_obs,
               'ombg': ombg,
               'oman': oman,
               'obsval': obsval,
               'hofx': hofx}
    return obs


def main(hfile, errfile, varname, is_variance, gridfile, obsfile=None, level=0):
    # --- Step 1: Open h file and compute depth
    dsg = xr.open_dataset(hfile)
    # Try both 'time' and 'Time' dimension names
    if 'time' in dsg.dims:
        h = dsg['h'].isel(time=0)        # layer thickness
    elif 'Time' in dsg.dims:
        h = dsg['h'].isel(Time=0)        # layer thickness
    else:
        raise ValueError("Could not find time dimension (tried both 'time' and 'Time')")
    surface_mask = ~np.isnan(h[0, :, :])  # True where surface layer is valid (not NaN)
    # Check which vertical dimension name exists in the dataset
    if 'z_l' in h.dims:
        depth = h.cumsum(dim='z_l')      # cumulative sum along vertical
    elif 'zaxis_1' in h.dims:
        depth = h.cumsum(dim='zaxis_1')  # cumulative sum along vertical
    else:
        print('...')
        #raise ValueError("Could not find vertical dimension (tried both 'z_l' and 'zaxis_1')")
    dsg.close()

    # --- Step 2: Open error file and extract variable
    ds = xr.open_dataset(errfile)
    # Try both 'time' and 'Time' dimension names
    if 'time' in ds.dims:
        data = ds[varname].isel(time=0)  # shape: (z_l, yh, xh)
    elif 'Time' in ds.dims:
        data = ds[varname].isel(Time=0)  # shape: (z_l, yh, xh)
    else:
        raise ValueError("Could not find time dimension (tried both 'time' and 'Time')")
    # Convert variance to stddev if needed
    if is_variance:
        print("--------------------------------- sqrt *************")
        data = np.sqrt(data)

    # --- Step 3: Open grid file and extract lon/lat
    grid = xr.open_dataset(gridfile)
    lon2d = grid['lon'].isel(Time=0)  # shape: (yh, xh)
    lat2d = grid['lat'].isel(Time=0)  # shape: (yh, xh)
    grid.close()

    obs = None
    print("%%%%%%%%%%%%%%%%%%% obsfile:", obsfile)
    if obsfile:
        print(f"Loading observation file: {obsfile}")
        obs = load_obsfile(obsfile, longitude_max=np.max(lon2d.values))

    # Make a simple 2D slice (e.g., surface field)
    # Handle both 2D and 3D data arrays
    if data.ndim == 3:
        surface_field = data[level, :, :]
    elif data.ndim == 2:
        surface_field = data
    else:
        raise ValueError(f"Unexpected number of dimensions for data: {data.ndim}")

    fig, ax = plt.subplots(figsize=(14, 8))
    masked_field = np.ma.masked_where(~surface_mask, surface_field)

    # Plot using pcolormesh for lon/lat with 50% transparency
    # Add color bounds as interactive sliders

    # Initial color bounds
    vmin_init = float(np.nanmin(masked_field))
    vmax_init = float(np.nanmax(masked_field))

    pcm = ax.pcolormesh(lon2d, lat2d, masked_field, vmin=vmin_init, vmax=vmax_init, cmap='gist_ncar', shading='auto', alpha=0.5)

    # Use a RangeSlider for both vmin and vmax

    # Slider axis: [left, bottom, width, height]
    slider_ax = fig.add_axes([0.1, 0.8, 0.12, 0.05])

    range_slider = RangeSlider(
        slider_ax, 'vmin/vmax', vmin_init, vmax_init,
        valinit=(vmin_init, vmax_init)
    )

    def update_color_bounds(val):
        vmin, vmax = range_slider.val
        pcm.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    range_slider.on_changed(update_color_bounds)

    if obs is not None:
        ax.scatter(obs['lon'], obs['lat'], s=2, c='black', label='Observations', alpha=1.0)
        ax.legend(loc='lower left')

    ax.set_title(f'Field from {os.path.basename(errfile)}\nClick on the map to show vertical profile, zonal or meridional slice')
    fig.colorbar(pcm, ax=ax, label='Surface Field (stddev)', shrink=0.3)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()

    # Add menu (RadioButtons) at the top right
    menu_ax = fig.add_axes([0.82, 0.75, 0.15, 0.15])  # [left, bottom, width, height]
    menu = mwidgets.RadioButtons(menu_ax, (
        'Vertical Profile', 'Zonal Slice', 'Meridional Slice', 'Observation Profile'))
    menu_ax.set_title("Plot Type", fontsize=10)

    # Store selected plot type
    plot_type = {'value': 'Vertical Profile'}

    def menu_on_clicked(label):
        plot_type['value'] = label

    menu.on_clicked(menu_on_clicked)

    # Function to find nearest neighbor index
    def find_nearest_2d(lon2d, lat2d, lon0, lat0):
        dist2 = (lon2d - lon0)**2 + (lat2d - lat0)**2
        iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)
        return iy, ix

    # Click event handler
    def onclick(event):
        if event.inaxes is not ax:
            return
        click_lon, click_lat = event.xdata, event.ydata
        iy, ix = find_nearest_2d(lon2d.values, lat2d.values, click_lon, click_lat)

        print(f"Clicked lon/lat: ({click_lon:.2f}, {click_lat:.2f})")
        print(f"Nearest grid indices: (x={ix}, y={iy})")

        if plot_type['value'] == 'Vertical Profile':
            fig_profile, ax_profile = plt.subplots()
            plot_vertical_profile(ix, iy, lon2d, lat2d, data, depth, ax_profile)
            plt.show()

        elif plot_type['value'] == 'Zonal Slice':
            print(f"Zonal slice at lat={lat2d[iy, ix].values:.2f}")
            print(f"iy, ix: {iy}, {ix}")
            zonal_profile = data[:, iy, :]
            zonal_lon = lon2d[iy, :]
            fig_zonal, ax_zonal = plt.subplots()
            ax_zonal.set_xlim(zonal_lon.min(), zonal_lon.max())
            pcm_zonal = ax_zonal.pcolormesh(zonal_lon, depth[:, iy, :], zonal_profile, shading='auto', cmap='gist_ncar')
            ax_zonal.invert_yaxis()
            ax_zonal.set_xlabel('Longitude')
            ax_zonal.set_ylabel('Depth')
            ax_zonal.set_title(f'Zonal Slice at lat={lat2d[iy, ix].values:.2f}')
            fig_zonal.colorbar(pcm_zonal, ax=ax_zonal, label='Field Stddev')
            plt.show()
        elif plot_type['value'] == 'Meridional Slice':
            print(f"Meridional slice at lon={lon2d[iy, ix].values:.2f}")
            print(f"iy, ix: {iy}, {ix}")
            meridional_profile = data[:, :, ix]
            meridional_lat = lat2d[:, ix]
            fig_merid, ax_merid = plt.subplots()
            pcm_merid = ax_merid.pcolormesh(meridional_lat, depth[:, :, ix], meridional_profile, shading='auto', cmap='gist_ncar')
            ax_merid.invert_yaxis()
            ax_merid.set_xlabel('Latitude')
            ax_merid.set_ylabel('Depth')
            ax_merid.set_title(f'Meridional Slice at lon={lon2d[iy, ix].values:.2f}')
            fig_merid.colorbar(pcm_merid, ax=ax_merid, label='Field Stddev')
            plt.show()
        elif plot_type['value'] == 'Observation Profile':
            # Find nearest observation
            dist2 = (obs['lon'] - click_lon)**2 + (obs['lat'] - click_lat)**2
            iobs = np.argmin(dist2)

            # Extract all ombg, oman, and depth values that match the lon/lat
            matching_lon = obs['lon'] == obs['lon'][iobs]
            matching_lat = obs['lat'] == obs['lat'][iobs]
            matching_indices = matching_lon & matching_lat

            ombg_values = obs['ombg'][matching_indices]
            oman_values = obs['oman'][matching_indices]
            depth_values = obs['depth'][matching_indices]
            obs_value = obs['obsval'][matching_indices]

            # Sort by increasing depth
            sort_idx = np.argsort(depth_values)
            depth_values = depth_values[sort_idx]
            ombg_values = ombg_values[sort_idx]
            oman_values = oman_values[sort_idx]
            obs_value = obs_value[sort_idx]

            # Plot the extracted values
            fig_obs, (ax_obs, ax_obsval, ax_model) = plt.subplots(1, 3, figsize=(15, 10), sharey=True)
            # OMBG/OMAN subplot
            ax_obs.plot(ombg_values, depth_values, '.-', color='tab:green', label='ombg')
            ax_obs.plot(oman_values, depth_values, '.-', color='tab:red', label='oman')
            ax_obs.invert_yaxis()
            ax_obs.set_xlabel('O-B (K)')
            ax_obs.set_ylabel('Depth (m)')
            ax_obs.set_title(f'OMB/OMA at lon={obs["lon"][iobs]:.2f}, lat={obs["lat"][iobs]:.2f}')
            ax_obs.legend()
            ax_obs.grid()

            # Model profile subplot (nearest grid point)
            # Find nearest grid point to obs location
            iy_grid, ix_grid = find_nearest_2d(lon2d.values, lat2d.values, obs['lon'][iobs], obs['lat'][iobs])
            model_profile = data[:, iy_grid, ix_grid]
            model_depth = depth[:, iy_grid, ix_grid]
            ax_model.plot(model_profile, model_depth, '.-', color='tab:purple', label='Model')
            ax_model.invert_yaxis()
            ax_model.set_xlabel('Model Value')
            ax_model.set_title(f'Model Profile\n(lon={lon2d[iy_grid, ix_grid].values:.2f}, lat={lat2d[iy_grid, ix_grid].values:.2f})')
            ax_model.legend()
            ax_model.grid()

            # Obs value subplot
            ax_obsval.plot(obs_value, depth_values, '.-', color='tab:blue', label='Obs Value')
            ax_obsval.plot(obs_value - ombg_values, depth_values, '.-', color='tab:green', label='Background')
            ax_obsval.plot(obs_value - oman_values, depth_values, '.-', color='tab:red', label='Analysis')
            ax_obsval.plot(model_profile, model_depth, '.-', color='tab:purple', label="QC'ed Analysis")
            ax_obsval.invert_yaxis()
            ax_obsval.set_xlabel('Obs Value (K)')
            ax_obsval.set_title('Obs Value')
            ax_obsval.legend()
            ax_obsval.grid()

            plt.show()

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot vertical profile from background error file.')
    parser.add_argument('--hfile', required=True, help='NetCDF file containing h variable')
    parser.add_argument('--errfile', required=True, help='NetCDF file containing background error variable')
    parser.add_argument('--varname', required=True, help='Variable name to plot (e.g., Temp)')
    parser.add_argument('--variance', action='store_true', help='Set if the file contains variance instead of standard deviation')
    parser.add_argument('--gridfile', required=True, help='NetCDF file containing 2D lon/lat variables (lon, lat)')
    parser.add_argument('--obsfile', required=False, help='IODA observation file with ombg and oman')
    parser.add_argument('--level', required=False, help='model level to plot (default: 0)', type=int, default=0)
    args = parser.parse_args()
    main(args.hfile, args.errfile, args.varname, args.variance, args.gridfile, obsfile=args.obsfile, level=args.level)
