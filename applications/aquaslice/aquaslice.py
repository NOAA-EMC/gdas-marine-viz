import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from netCDF4 import Dataset
import cartopy.feature as cfeature
import os


def plot_vertical_profile(ix, iy, lon2d, lat2d, data, depth, ax_profile, is_atmos=False):
    # Handle both xarray and numpy arrays for lon2d/lat2d
    lon_val = lon2d[iy, ix].values if hasattr(lon2d[iy, ix], 'values') else lon2d[iy, ix]
    lat_val = lat2d[iy, ix].values if hasattr(lat2d[iy, ix], 'values') else lat2d[iy, ix]

    print(f"Vertical profile at lon={lon_val:.2f}, lat={lat_val:.2f}")
    print(f"iy, ix: {iy}, {ix}")
    profile = data[:, iy, ix]
    depth_profile = depth[:, iy, ix]
    print(f"profile: f{profile}")
    print(f"depth: f{depth_profile}")

    ax_profile.plot(profile, depth_profile, '-o')
    ax_profile.invert_yaxis()
    ax_profile.set_xlabel('Field Value')
    ylabel = 'Pressure (hPa)' if is_atmos else 'Depth (m)'
    ax_profile.set_ylabel(ylabel)
    ax_profile.set_title(f'Vertical Profile at (lon={lon_val:.2f}, lat={lat_val:.2f})')
    ax_profile.grid()


def load_obsfile(obsfile, longitude_max=None, variable='Temp'):
    print(f"Loading observation file: {obsfile}")
    print(f"Longitude min: {longitude_max}")
    var_map = {'Temp': 'waterTemperature', 'Salt': 'salinity'}
    nc_var = var_map[variable]
    with Dataset(obsfile, 'r') as f:
        lat_obs = f.groups['MetaData'].variables['latitude'][:]
        lon_obs = f.groups['MetaData'].variables['longitude'][:]
        lon_obs[lon_obs > longitude_max] -= 360.0  # Rotate to the mom6 grid
        depth_obs = f.groups['MetaData'].variables['depth'][:]
        ombg = f.groups['ombg'].variables[nc_var][:]
        oman = f.groups['oman'].variables[nc_var][:]
        obsval = f.groups['ObsValue'].variables[nc_var][:]
        hofx = f.groups['hofx0'].variables[nc_var][:]

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


def compute_pressure_levels(ak, bk, ps=101325.0):
    """
    Compute pressure levels from hybrid sigma-pressure coordinates.

    Parameters:
    -----------
    ak : array
        Hybrid sigma-pressure coefficient A (Pa)
    bk : array
        Hybrid sigma-pressure coefficient B (dimensionless)
    ps : float or array
        Surface pressure (Pa), default is standard atmosphere 101325 Pa

    Returns:
    --------
    pressure : array
        Pressure at each level (Pa)
    """
    # For layer centers, use mid-point of ak and bk between edges
    ak_mid = 0.5 * (ak[:-1] + ak[1:])
    bk_mid = 0.5 * (bk[:-1] + bk[1:])
    pressure = ak_mid + bk_mid * ps
    return pressure


def load_atmospheric_data(atmfile, varname, level=0):
    """
    Load atmospheric data from Gaussian grid or cube sphere file.

    Parameters:
    -----------
    atmfile : str
        Path to atmospheric NetCDF file
    varname : str
        Variable name to load (e.g., 'T_inc', 'u_inc', 'v_inc', 'tmp', 'ugrd', etc.)
    level : int
        Vertical level to plot for 2D view (default: 0 = top of atmosphere)

    Returns:
    --------
    data : xarray.DataArray
        4D data array (time, lev, lat, lon)
    lon : xarray.DataArray
        1D or 2D longitude array
    lat : xarray.DataArray
        1D or 2D latitude array
    pressure : numpy.ndarray
        Pressure levels (hPa)
    """
    ds = xr.open_dataset(atmfile)

    # Load the variable
    data = ds[varname].isel(time=0)  # shape: (lev, lat, lon) or (pfull, grid_yt, grid_xt)

    # Detect file format and get lon/lat coordinates
    if 'grid_xt' in ds.dims and 'grid_yt' in ds.dims:
        # Gaussian grid format (e.g., gdas.t00z.atm.f006)
        print("Detected Gaussian grid format")

        # Check if lon/lat are 1D or 2D
        if 'lon' in ds.variables and ds['lon'].ndim == 2:
            # 2D lon/lat arrays
            lon = ds['lon'].isel(time=0) if 'time' in ds['lon'].dims else ds['lon']
            lat = ds['lat'].isel(time=0) if 'time' in ds['lat'].dims else ds['lat']
        else:
            # 1D lon/lat arrays - create meshgrid
            lon_1d = ds['grid_xt']
            lat_1d = ds['grid_yt']
            lon, lat = np.meshgrid(lon_1d, lat_1d)

        # Check if pfull variable exists, otherwise use ak/bk
        if 'pfull' in ds.variables:
            pressure = ds['pfull'].values  # Already in hPa (mbar)
            print(f"Using pfull variable for pressure levels: {len(pressure)} levels")
        elif 'ak' in ds.attrs and 'bk' in ds.attrs:
            # Get hybrid coefficients from global attributes
            ak = np.array(ds.attrs['ak'])
            bk = np.array(ds.attrs['bk'])
            # Compute pressure levels (convert Pa to hPa for readability)
            pressure = compute_pressure_levels(ak, bk) / 100.0  # Convert Pa to hPa
            print(f"Computed pressure from ak/bk: {len(pressure)} levels")
        else:
            raise ValueError("Could not determine pressure levels: no 'pfull' variable or 'ak'/'bk' attributes found")

    else:
        # Cube sphere format (e.g., increments files)
        print("Detected cube sphere format")
        lon = ds['lon']
        lat = ds['lat']

        # Check if latitude is descending (North to South) and reverse the data
        # This fixes the issue where Arctic appears in Antarctic and vice versa
        if len(lat) > 1 and lat[0] > lat[-1]:
            print("Reversing atmospheric data along latitude dimension to fix pole placement")
            # Keep lat array as-is (90 to -90), but flip the data
            data = data[:, ::-1, :]  # Reverse latitude dimension in data

        # Get hybrid coefficients from global attributes
        ak = np.array(ds.attrs['ak'])
        bk = np.array(ds.attrs['bk'])

        # Compute pressure levels (convert Pa to hPa for readability)
        pressure = compute_pressure_levels(ak, bk) / 100.0  # Convert Pa to hPa
        print(f"Computed pressure from ak/bk: {len(pressure)} levels")

    ds.close()

    return data, lon, lat, pressure


def main(hfile, oceanfile, atmosfile, oceanvarname, atmosvarname, is_variance, gridfile, obsfile=None, level=None,
         vmin=None, vmax=None, ocean_vmin=None, ocean_vmax=None, atmos_vmin=None, atmos_vmax=None, atmos_to_celsius=False):
    # Determine mode based on which file is provided
    ocean_mode = oceanfile is not None
    atmos_mode = atmosfile is not None

    if atmos_mode and not ocean_mode:
        # Pure atmospheric mode
        data, lon, lat, pressure = load_atmospheric_data(atmosfile, atmosvarname, level=0)

        # Convert to Celsius if requested
        if atmos_to_celsius:
            print("Converting atmospheric temperature from Kelvin to Celsius")
            data = data - 273.15

        # Set default level to last index (near surface) if not specified
        if level is None:
            level = data.shape[0] - 1  # Last vertical level (near surface)

        # Create 2D meshgrid for plotting (if lon/lat are 1D)
        if lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            # Already 2D (Gaussian grid format)
            lon2d = lon
            lat2d = lat

        # Create 3D pressure field (broadcast to match data shape)
        # Shape: (lev, lat, lon)
        pressure_3d = np.broadcast_to(pressure[:, np.newaxis, np.newaxis], data.shape)

        # For 2D plotting, select the specified level
        surface_field = data[level, :, :]
        surface_mask = np.ones_like(surface_field, dtype=bool)  # No masking for atmos

        obs = None
        depth = pressure_3d  # Use pressure as "depth" for vertical profiles
        is_atmos = True

    elif ocean_mode and not atmos_mode:
        # Pure ocean mode
        # Set default level to 0 (surface) if not specified
        if level is None:
            level = 0

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
        mask3d = np.where(h >= 0.01, 1, np.nan)
        h = h * mask3d  # Set very thin layers to NaN
        # Check which vertical dimension name exists in the dataset
        if 'z_l' in h.dims:
            depth = h.cumsum(dim='z_l')      # cumulative sum along vertical
        elif 'zaxis_1' in h.dims:
            depth = h.cumsum(dim='zaxis_1')  # cumulative sum along vertical
        else:
            print('...')
            raise ValueError("Could not find vertical dimension (tried both 'z_l' and 'zaxis_1')")
        dsg.close()

        # --- Step 2: Open ocean file and extract variable
        ds = xr.open_dataset(oceanfile)
        # Try both 'time' and 'Time' dimension names
        if 'time' in ds.dims:
            data = ds[oceanvarname].isel(time=0)  # shape: (z_l, yh, xh)
        elif 'Time' in ds.dims:
            data = ds[oceanvarname].isel(Time=0)  # shape: (z_l, yh, xh)
        else:
            raise ValueError("Could not find time dimension (tried both 'time' and 'Time')")
        data = data * mask3d  # Set invalid points to NaN
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
            obs = load_obsfile(obsfile, longitude_max=np.max(lon2d.values), variable=oceanvarname)

        # Make a simple 2D slice (e.g., surface field)
        # Handle both 2D and 3D data arrays
        if data.ndim == 3:
            surface_field = data[level, :, :]
        elif data.ndim == 2:
            surface_field = data
        else:
            raise ValueError(f"Unexpected number of dimensions for data: {data.ndim}")

        is_atmos = False

    else:
        # Both ocean and atmospheric mode - combined display
        if level is None:
            level = 0  # Use surface for both by default

        # Load atmospheric data
        atmos_data, atmos_lon, atmos_lat, atmos_pressure = load_atmospheric_data(atmosfile, atmosvarname, level=0)

        # Convert to Celsius if requested
        if atmos_to_celsius:
            print("Converting atmospheric temperature from Kelvin to Celsius")
            atmos_data = atmos_data - 273.15

        # Use last level for atmospheric default if level not specified
        atmos_level = atmos_data.shape[0] - 1 if level == 0 else level

        # Create atmospheric 2D meshgrid (if lon/lat are 1D)
        if atmos_lon.ndim == 1:
            atmos_lon2d, atmos_lat2d = np.meshgrid(atmos_lon, atmos_lat)
        else:
            # Already 2D (Gaussian grid format)
            atmos_lon2d = atmos_lon
            atmos_lat2d = atmos_lat
        atmos_pressure_3d = np.broadcast_to(atmos_pressure[:, np.newaxis, np.newaxis], atmos_data.shape)
        atmos_surface_field = atmos_data[atmos_level, :, :]
        atmos_surface_mask = np.ones_like(atmos_surface_field, dtype=bool)

        # Load ocean data
        # --- Step 1: Open h file and compute depth
        dsg = xr.open_dataset(hfile)
        if 'time' in dsg.dims:
            h = dsg['h'].isel(time=0)
        elif 'Time' in dsg.dims:
            h = dsg['h'].isel(Time=0)
        else:
            raise ValueError("Could not find time dimension (tried both 'time' and 'Time')")
        ocean_surface_mask = ~np.isnan(h[0, :, :])
        mask3d = np.where(h >= 0.01, 1, np.nan)
        h = h * mask3d
        if 'z_l' in h.dims:
            ocean_depth = h.cumsum(dim='z_l')
        elif 'zaxis_1' in h.dims:
            ocean_depth = h.cumsum(dim='zaxis_1')
        else:
            raise ValueError("Could not find vertical dimension (tried both 'z_l' and 'zaxis_1')")
        dsg.close()

        # --- Step 2: Open ocean file
        ds = xr.open_dataset(oceanfile)
        if 'time' in ds.dims:
            ocean_data = ds[oceanvarname].isel(time=0)
        elif 'Time' in ds.dims:
            ocean_data = ds[oceanvarname].isel(Time=0)
        else:
            raise ValueError("Could not find time dimension (tried both 'time' and 'Time')")
        ocean_data = ocean_data * mask3d
        if is_variance:
            ocean_data = np.sqrt(ocean_data)

        # --- Step 3: Open grid file
        grid = xr.open_dataset(gridfile)
        ocean_lon2d = grid['lon'].isel(Time=0)
        ocean_lat2d = grid['lat'].isel(Time=0)
        grid.close()

        ocean_level = 0 if level == 0 else level
        if ocean_data.ndim == 3:
            ocean_surface_field = ocean_data[ocean_level, :, :]
        elif ocean_data.ndim == 2:
            ocean_surface_field = ocean_data
        else:
            raise ValueError(f"Unexpected number of dimensions for ocean data: {ocean_data.ndim}")

        obs = None
        if obsfile:
            obs = load_obsfile(obsfile, longitude_max=np.max(ocean_lon2d.values), variable=oceanvarname)

        # Store both datasets for interactive plotting
        combined_data = {
            'atmos': {'data': atmos_data, 'depth': atmos_pressure_3d, 'lon2d': atmos_lon2d, 'lat2d': atmos_lat2d, 'is_atmos': True},
            'ocean': {'data': ocean_data, 'depth': ocean_depth, 'lon2d': ocean_lon2d, 'lat2d': ocean_lat2d, 'is_atmos': False}
        }

    # Create figure - either single plot or combined
    if ocean_mode and atmos_mode:
        # Combined mode - two subplots stacked vertically
        fig, (ax_atmos, ax_ocean) = plt.subplots(2, 1, figsize=(14, 14))

        # Plot atmospheric field on top
        # Plot twice: once at original longitude, once shifted by -360
        # This allows wrapping around the date line to match ocean grid
        atmos_masked = np.ma.masked_where(~atmos_surface_mask, atmos_surface_field)
        vmin_atmos = atmos_vmin if atmos_vmin is not None else float(np.nanmin(atmos_masked))
        vmax_atmos = atmos_vmax if atmos_vmax is not None else float(np.nanmax(atmos_masked))

        # First plot: original longitude
        pcm_atmos = ax_atmos.pcolormesh(atmos_lon2d, atmos_lat2d, atmos_masked,
                                        vmin=vmin_atmos, vmax=vmax_atmos,
                                        cmap='gist_ncar', shading='auto', alpha=0.5)
        # Second plot: shifted by -360 degrees
        ax_atmos.pcolormesh(atmos_lon2d - 360, atmos_lat2d, atmos_masked,
                            vmin=vmin_atmos, vmax=vmax_atmos,
                            cmap='gist_ncar', shading='auto', alpha=0.5)

        # Set axis limits to -180 to 180
        ax_atmos.set_xlim(-180, 180)

        ax_atmos.set_title(f'Atmospheric: {os.path.basename(atmosfile)} - {atmosvarname} (Level {atmos_level})')
        fig.colorbar(pcm_atmos, ax=ax_atmos, label=f'{atmosvarname}', shrink=0.5, pad=0.02)
        ax_atmos.set_xlabel('Longitude')
        ax_atmos.set_ylabel('Latitude')

        # Add coastlines to atmospheric plot using cartopy geometries
        try:
            coastlines = cfeature.COASTLINE.geometries()
            for geom in coastlines:
                if geom.geom_type == 'LineString':
                    x, y = geom.xy
                    ax_atmos.plot(x, y, color='black', linewidth=0.5)
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        x, y = line.xy
                        ax_atmos.plot(x, y, color='black', linewidth=0.5)
        except Exception as e:
            print(f"Warning: Could not add coastlines: {e}")
            print("Continuing without coastlines...")

        # Plot ocean field on bottom
        # Plot twice: once at original longitude, once shifted by +360
        # This ensures complete coverage for longitudes less than -180
        ocean_masked = np.ma.masked_where(~ocean_surface_mask, ocean_surface_field)
        vmin_ocean = ocean_vmin if ocean_vmin is not None else float(np.nanmin(ocean_masked))
        vmax_ocean = ocean_vmax if ocean_vmax is not None else float(np.nanmax(ocean_masked))

        # First plot: original longitude
        pcm_ocean = ax_ocean.pcolormesh(ocean_lon2d, ocean_lat2d, ocean_masked,
                                        vmin=vmin_ocean, vmax=vmax_ocean,
                                        cmap='gist_ncar', shading='auto', alpha=0.5)
        # Second plot: shifted by +360 degrees
        ax_ocean.pcolormesh(ocean_lon2d + 360, ocean_lat2d, ocean_masked,
                            vmin=vmin_ocean, vmax=vmax_ocean,
                            cmap='gist_ncar', shading='auto', alpha=0.5)

        # Set axis limits to -180 to 180
        ax_ocean.set_xlim(-180, 180)

        if obs is not None:
            # Plot observations twice: once at original longitude, once shifted by +360
            # This ensures observations appear correctly across the -180 to 180 longitude range
            ax_ocean.scatter(obs['lon'], obs['lat'], s=2, c='black', label='Observations', alpha=1.0)
            ax_ocean.scatter(obs['lon'] + 360, obs['lat'], s=2, c='black', alpha=1.0)
            ax_ocean.legend(loc='lower left')
        ax_ocean.set_title(f'Ocean: {os.path.basename(oceanfile)} - {oceanvarname} (Level {ocean_level})')
        fig.colorbar(pcm_ocean, ax=ax_ocean, label=f'{oceanvarname}', shrink=0.5, pad=0.02)
        ax_ocean.set_xlabel('Longitude')
        ax_ocean.set_ylabel('Latitude')

        fig.tight_layout(rect=[0, 0, 0.82, 1])  # Leave space on right for menu

        # Store reference to active subplot for click events
        active_domain = {'value': 'ocean'}

    else:
        # Single mode plotting (existing code)
        fig, ax = plt.subplots(figsize=(14, 8))
        masked_field = np.ma.masked_where(~surface_mask, surface_field)

        # Initial color bounds
        vmin_init = vmin if vmin is not None else float(np.nanmin(masked_field))
        vmax_init = vmax if vmax is not None else float(np.nanmax(masked_field))

        pcm = ax.pcolormesh(lon2d, lat2d, masked_field, vmin=vmin_init, vmax=vmax_init, cmap='gist_ncar', shading='auto', alpha=0.5)

        if obs is not None:
            ax.scatter(obs['lon'], obs['lat'], s=2, c='black', label='Observations', alpha=1.0)
            ax.legend(loc='lower left')

        filename = os.path.basename(atmosfile if is_atmos else oceanfile)
        ax.set_title(f'Field from {filename}\nClick on the map to show vertical profile, zonal or meridional slice')
        fig.colorbar(pcm, ax=ax, label='Surface Field (stddev)', shrink=0.3)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # Don't call tight_layout yet - wait until we know if we need menu space

    # Add menu (RadioButtons) at the top right
    menu_options = ['Vertical Profile', 'Zonal Slice', 'Meridional Slice']
    if ocean_mode and atmos_mode:
        menu_options.append('Combined Profile')
        menu_options.append('Combined Zonal Slice')
    if ocean_mode and not atmos_mode and obs is not None:
        menu_options.append('Observation Profile')

    # Add domain selector for combined mode
    if ocean_mode and atmos_mode:
        menu_options_domain = ['Ocean', 'Atmosphere']
        menu_ax_domain = fig.add_axes([0.82, 0.85, 0.15, 0.10])
        menu_domain = mwidgets.RadioButtons(menu_ax_domain, tuple(menu_options_domain))
        menu_ax_domain.set_title("Domain", fontsize=10)

        def domain_on_clicked(label):
            active_domain['value'] = label.lower()

        menu_domain.on_clicked(domain_on_clicked)

    menu_ax = fig.add_axes([0.82, 0.70, 0.15, 0.15])  # [left, bottom, width, height]
    menu = mwidgets.RadioButtons(menu_ax, tuple(menu_options))
    menu_ax.set_title("Plot Type", fontsize=10)

    # Now apply tight_layout with appropriate spacing
    # Reserve space on right for menu in combined mode or when we have observation profiles
    if (ocean_mode and atmos_mode) or (ocean_mode and not atmos_mode and obs is not None):
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        fig.tight_layout()

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
        # Determine which subplot was clicked for combined mode
        if ocean_mode and atmos_mode:
            if event.inaxes == ax_atmos:
                active_domain['value'] = 'atmosphere'
            elif event.inaxes == ax_ocean:
                active_domain['value'] = 'ocean'
            else:
                return

            # Get appropriate data based on clicked domain
            if active_domain['value'] == 'atmosphere':
                data_to_use = combined_data['atmos']['data']
                depth_to_use = combined_data['atmos']['depth']
                lon2d_to_use = combined_data['atmos']['lon2d']
                lat2d_to_use = combined_data['atmos']['lat2d']
                is_atmos_to_use = True
            else:
                data_to_use = combined_data['ocean']['data']
                depth_to_use = combined_data['ocean']['depth']
                lon2d_to_use = combined_data['ocean']['lon2d']
                lat2d_to_use = combined_data['ocean']['lat2d']
                is_atmos_to_use = False
        else:
            # Single mode
            if event.inaxes is not ax:
                return
            data_to_use = data
            depth_to_use = depth
            lon2d_to_use = lon2d
            lat2d_to_use = lat2d
            is_atmos_to_use = is_atmos

        click_lon, click_lat = event.xdata, event.ydata
        # Handle both xarray and numpy arrays
        lon_vals = lon2d_to_use.values if hasattr(lon2d_to_use, 'values') else lon2d_to_use
        lat_vals = lat2d_to_use.values if hasattr(lat2d_to_use, 'values') else lat2d_to_use
        iy, ix = find_nearest_2d(lon_vals, lat_vals, click_lon, click_lat)

        print(f"Clicked lon/lat: ({click_lon:.2f}, {click_lat:.2f})")
        print(f"Nearest grid indices: (x={ix}, y={iy})")
        if ocean_mode and atmos_mode:
            print(f"Domain: {active_domain['value']}")

        if plot_type['value'] == 'Vertical Profile':
            fig_profile, ax_profile = plt.subplots()

            # Plot observations first (behind) if available
            # Works in both ocean-only mode and combined mode (when ocean is selected)
            if obs is not None and not is_atmos_to_use:
                # Find nearest observation to the clicked location
                dist2 = (obs['lon'] - click_lon)**2 + (obs['lat'] - click_lat)**2
                iobs = np.argmin(dist2)

                # Extract all values at this location
                matching_lon = obs['lon'] == obs['lon'][iobs]
                matching_lat = obs['lat'] == obs['lat'][iobs]
                matching_indices = matching_lon & matching_lat

                obs_depths = obs['depth'][matching_indices]
                obs_values = obs['obsval'][matching_indices]

                # Sort by depth
                sort_idx = np.argsort(obs_depths)
                obs_depths = obs_depths[sort_idx]
                obs_values = obs_values[sort_idx]

                # Plot observations on the same axes (will be behind model profile)
                ax_profile.plot(obs_values, obs_depths, 'o-', color='red',
                                label=f'Obs (lon={obs["lon"][iobs]:.2f}, lat={obs["lat"][iobs]:.2f})',
                                markersize=6, linewidth=2, alpha=0.5)

            # Plot model profile on top
            plot_vertical_profile(ix, iy, lon2d_to_use, lat2d_to_use, data_to_use, depth_to_use, ax_profile, is_atmos=is_atmos_to_use)

            # Add legend if observations were plotted
            if obs is not None and not is_atmos_to_use:
                ax_profile.legend()

            plt.show()

        elif plot_type['value'] == 'Zonal Slice':
            lat_val = lat2d_to_use[iy, ix].values if hasattr(lat2d_to_use[iy, ix], 'values') else lat2d_to_use[iy, ix]
            print(f"Zonal slice at lat={lat_val:.2f}")
            print(f"iy, ix: {iy}, {ix}")
            zonal_profile = data_to_use[:, iy, :]
            zonal_lon = lon2d_to_use[iy, :]
            fig_zonal, ax_zonal = plt.subplots()
            ax_zonal.set_xlim(zonal_lon.min(), zonal_lon.max())
            pcm_zonal = ax_zonal.pcolormesh(zonal_lon, depth_to_use[:, iy, :], zonal_profile, shading='auto', cmap='gist_ncar')
            # Always invert y-axis: ocean depth increases down, atmos pressure decreases up
            ax_zonal.invert_yaxis()
            ax_zonal.set_xlabel('Longitude')
            ylabel = 'Pressure (hPa)' if is_atmos_to_use else 'Depth (m)'
            ax_zonal.set_ylabel(ylabel)
            ax_zonal.set_title(f'Zonal Slice at lat={lat_val:.2f}')
            fig_zonal.colorbar(pcm_zonal, ax=ax_zonal, label='Field Value')
            plt.show()
        elif plot_type['value'] == 'Meridional Slice':
            lon_val = lon2d_to_use[iy, ix].values if hasattr(lon2d_to_use[iy, ix], 'values') else lon2d_to_use[iy, ix]
            print(f"Meridional slice at lon={lon_val:.2f}")
            print(f"iy, ix: {iy}, {ix}")
            meridional_profile = data_to_use[:, :, ix]
            meridional_lat = lat2d_to_use[:, ix]
            fig_merid, ax_merid = plt.subplots()
            pcm_merid = ax_merid.pcolormesh(meridional_lat, depth_to_use[:, :, ix], meridional_profile,
                                            shading='auto', cmap='gist_ncar')
            # Always invert y-axis: ocean depth increases down, atmos pressure decreases up
            ax_merid.invert_yaxis()
            ax_merid.set_xlabel('Latitude')
            ylabel = 'Pressure (hPa)' if is_atmos_to_use else 'Depth (m)'
            ax_merid.set_ylabel(ylabel)
            ax_merid.set_title(f'Meridional Slice at lon={lon_val:.2f}')
            fig_merid.colorbar(pcm_merid, ax=ax_merid, label='Field Value')
            plt.show()
        elif plot_type['value'] == 'Combined Profile':
            # Only available in combined mode
            if not (ocean_mode and atmos_mode):
                print("Combined Profile only available in combined ocean/atmospheric mode")
                return

            # Get profiles from both domains at the clicked location
            # Ocean profile
            ocean_data = combined_data['ocean']['data']
            ocean_depth = combined_data['ocean']['depth']
            ocean_lon2d = combined_data['ocean']['lon2d']
            ocean_lat2d = combined_data['ocean']['lat2d']

            # Atmospheric profile
            atmos_data = combined_data['atmos']['data']
            atmos_depth = combined_data['atmos']['depth']  # This is pressure
            atmos_lon2d = combined_data['atmos']['lon2d']
            atmos_lat2d = combined_data['atmos']['lat2d']

            # Find nearest grid points in each domain
            # Handle longitude wrapping for proper distance calculation
            ocean_lon_vals = ocean_lon2d.values if hasattr(ocean_lon2d, 'values') else ocean_lon2d
            ocean_lat_vals = ocean_lat2d.values if hasattr(ocean_lat2d, 'values') else ocean_lat2d

            # Wrap click_lon to match ocean grid longitude range
            ocean_lon_min = np.min(ocean_lon_vals)
            ocean_lon_max = np.max(ocean_lon_vals)
            click_lon_ocean = click_lon
            if click_lon < ocean_lon_min:
                click_lon_ocean = click_lon + 360.0
            elif click_lon > ocean_lon_max:
                click_lon_ocean = click_lon - 360.0

            iy_ocean, ix_ocean = find_nearest_2d(ocean_lon_vals, ocean_lat_vals, click_lon_ocean, click_lat)

            # Convert atmos arrays to numpy if they're xarray DataArrays
            atmos_lon_vals = atmos_lon2d.values if hasattr(atmos_lon2d, 'values') else atmos_lon2d
            atmos_lat_vals = atmos_lat2d.values if hasattr(atmos_lat2d, 'values') else atmos_lat2d

            # Wrap click_lon to match atmospheric grid longitude range
            atmos_lon_min = np.min(atmos_lon_vals)
            atmos_lon_max = np.max(atmos_lon_vals)
            click_lon_atmos = click_lon
            if click_lon < atmos_lon_min:
                click_lon_atmos = click_lon + 360.0
            elif click_lon > atmos_lon_max:
                click_lon_atmos = click_lon - 360.0

            iy_atmos, ix_atmos = find_nearest_2d(atmos_lon_vals, atmos_lat_vals, click_lon_atmos, click_lat)

            # Extract profiles
            ocean_profile = ocean_data[:, iy_ocean, ix_ocean]
            ocean_depth_profile = ocean_depth[:, iy_ocean, ix_ocean]
            ocean_lon_val = (ocean_lon2d[iy_ocean, ix_ocean].values if
                             hasattr(ocean_lon2d[iy_ocean, ix_ocean], 'values') else
                             ocean_lon2d[iy_ocean, ix_ocean])
            ocean_lat_val = (ocean_lat2d[iy_ocean, ix_ocean].values if
                             hasattr(ocean_lat2d[iy_ocean, ix_ocean], 'values') else
                             ocean_lat2d[iy_ocean, ix_ocean])

            atmos_profile = atmos_data[:, iy_atmos, ix_atmos]
            atmos_depth_profile = atmos_depth[:, iy_atmos, ix_atmos]
            atmos_lon_val = (atmos_lon2d[iy_atmos, ix_atmos].values if
                             hasattr(atmos_lon2d[iy_atmos, ix_atmos], 'values') else
                             atmos_lon2d[iy_atmos, ix_atmos])
            atmos_lat_val = (atmos_lat2d[iy_atmos, ix_atmos].values if
                             hasattr(atmos_lat2d[iy_atmos, ix_atmos], 'values') else
                             atmos_lat2d[iy_atmos, ix_atmos])

            # Create a single continuous profile with normalized vertical coordinate
            # Normalize atmospheric pressure: 0 (top of atmosphere) to 0.5 (surface)
            # Normalize ocean depth: 0.5 (surface) to 1.0 (bottom)

            # Atmospheric data: index 0 = top of atmosphere (low pressure), last index = surface (high pressure)
            # We want to display: top at y=0 (low pressure), surface at y=0.5 (high pressure)
            # So we don't reverse - use original order
            atmos_profile_to_plot = atmos_profile
            atmos_pressure_to_plot = atmos_depth_profile

            # Normalize atmospheric vertical coordinate (0 to 0.5)
            # Top of atmosphere (low pressure) at 0, surface (high pressure) at 0.5
            atmos_norm = np.linspace(0, 0.5, len(atmos_profile_to_plot))

            # Get actual depth values for proper scaling
            # Use valid (non-NaN) ocean depths only
            valid_ocean_mask = ~np.isnan(ocean_depth_profile)
            valid_ocean_depths = ocean_depth_profile[valid_ocean_mask]
            valid_ocean_profile = ocean_profile[valid_ocean_mask]

            # If no valid depths, fall back to original behavior
            if len(valid_ocean_depths) == 0:
                valid_ocean_depths = ocean_depth_profile
                valid_ocean_profile = ocean_profile

            max_ocean_depth = np.nanmax(valid_ocean_depths)

            # Normalize ocean vertical coordinate (0.5 to 1.0) based on actual depths
            # This ensures observations will align correctly
            ocean_norm = 0.5 + (valid_ocean_depths / max_ocean_depth) * 0.5

            # Create single plot
            fig_combined, ax_combined = plt.subplots(figsize=(10, 10))

            # Plot observations first (behind) if available (ocean part only)
            if obs is not None:
                # Find nearest observation to the clicked location
                dist2 = (obs['lon'] - click_lon)**2 + (obs['lat'] - click_lat)**2
                iobs = np.argmin(dist2)

                # Extract all values at this location
                matching_lon = obs['lon'] == obs['lon'][iobs]
                matching_lat = obs['lat'] == obs['lat'][iobs]
                matching_indices = matching_lon & matching_lat

                obs_depths = obs['depth'][matching_indices]
                obs_values = obs['obsval'][matching_indices]

                # Sort by depth
                sort_idx = np.argsort(obs_depths)
                obs_depths = obs_depths[sort_idx]
                obs_values = obs_values[sort_idx]

                # Normalize observation depths to 0.5-1.0 range using the same scaling as model
                # This ensures observations align with the depth axis
                obs_norm = 0.5 + (obs_depths / max_ocean_depth) * 0.5
                # Clip to valid range (but allow observations deeper than model grid)
                obs_norm = np.clip(obs_norm, 0.5, 1.0)

                # Plot observations on the same axes (will be behind model profiles)
                ax_combined.plot(obs_values, obs_norm, 'o-', color='red',
                                 label=f'Obs (lon={obs["lon"][iobs]:.2f}, lat={obs["lat"][iobs]:.2f})',
                                 markersize=6, linewidth=2, alpha=0.5)

            # Plot atmospheric profile (top half: 0 to 0.5) - on top
            ax_combined.plot(atmos_profile_to_plot, atmos_norm, '-o', color='tab:red',
                             label=f'{atmosvarname} (Atmosphere)', markersize=3, linewidth=1.5)

            # Plot ocean profile (bottom half: 0.5 to 1.0) - on top
            ax_combined.plot(valid_ocean_profile, ocean_norm, '-o', color='tab:blue',
                             label=f'{oceanvarname} (Ocean)', markersize=3, linewidth=1.5)

            # Add horizontal line at the interface
            ax_combined.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                                label='Interface')

            # Set up dual y-axes with pressure on left (top half) and depth on right (bottom half)
            ax_combined.set_ylim(0, 1)
            ax_combined.invert_yaxis()  # Invert so 0 is at top, 1 is at bottom
            ax_combined.set_ylabel('Normalized Vertical Coordinate', fontsize=11)
            ax_combined.set_xlabel('Field Value', fontsize=11)

            # Create custom y-tick labels
            # Top half: atmosphere (low pressure at top, high pressure at bottom/surface)
            # Bottom half: ocean (depth increases downward)
            yticks = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
            yticklabels = []

            for yt in yticks:
                if yt < 0.5:
                    # Atmosphere: interpolate pressure from original (not reversed) array
                    # yt=0 corresponds to idx=0 (low pressure, top of atmosphere)
                    # yt=0.5 corresponds to idx=-1 (high pressure, surface)
                    idx = int((yt / 0.5) * (len(atmos_pressure_to_plot) - 1))
                    if idx < len(atmos_pressure_to_plot):
                        yticklabels.append(f'{atmos_pressure_to_plot[idx]:.0f} hPa')
                    else:
                        yticklabels.append('')
                elif yt == 0.5:
                    yticklabels.append('Surface')
                else:
                    # Ocean: interpolate depth using the actual valid depths that are plotted
                    # yt=0.5 corresponds to surface (depth=0), yt=1.0 corresponds to max depth
                    depth_fraction = (yt - 0.5) / 0.5  # 0 to 1
                    actual_depth = depth_fraction * max_ocean_depth
                    yticklabels.append(f'{actual_depth:.0f} m')

            ax_combined.set_yticks(yticks)
            ax_combined.set_yticklabels(yticklabels, fontsize=9)

            # Add labels for atmosphere/ocean regions
            ax_combined.text(0.02, 0.75, 'ATMOSPHERE', transform=ax_combined.transAxes,
                             fontsize=10, color='tab:red', fontweight='bold', alpha=0.7,
                             rotation=90, va='center')
            ax_combined.text(0.02, 0.25, 'OCEAN', transform=ax_combined.transAxes,
                             fontsize=10, color='tab:blue', fontweight='bold', alpha=0.7,
                             rotation=90, va='center')

            ax_combined.grid(True, alpha=0.3)
            ax_combined.legend(loc='best', fontsize=9)

            # Add title
            title_text = (f'Combined Vertical Profile\nAtmos: (lon={atmos_lon_val:.2f}, '
                          f'lat={atmos_lat_val:.2f}) | Ocean: (lon={ocean_lon_val:.2f}, '
                          f'lat={ocean_lat_val:.2f})')
            ax_combined.set_title(title_text, fontsize=11, pad=10)

            plt.tight_layout()
            plt.show()

        elif plot_type['value'] == 'Observation Profile':
            # Only available in ocean-only mode
            if not (ocean_mode and not atmos_mode and obs is not None):
                print("Observation Profile only available in ocean-only mode with observations loaded")
                return

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

            print(f"model: f{model_profile}")
            print(f"depth: f{model_depth}")

            ax_model.plot(model_profile, model_depth, '.-', color='tab:purple', label='Model')
            ax_model.invert_yaxis()
            ax_model.set_xlabel('Model Value')
            ax_model.set_title(f'Model Profile\n(lon={lon2d[iy_grid, ix_grid].values:.2f}, lat={lat2d[iy_grid, ix_grid].values:.2f})')
            ax_model.legend()
            ax_model.grid()

            # Obs value subplot
            np.set_printoptions(threshold=np.inf)

            # Print analysis values (temperatures)
            T_ana_valid = obs_value - oman_values
            z_ana_valid = depth_values
            print("T_ana_valid = [")
            print(", ".join(f"{v:.6f}" for v in T_ana_valid))
            print("]")

            # Print analysis depths
            print("z_ana_valid = [")
            print(", ".join(f"{v:.1f}" for v in z_ana_valid))
            print("]")

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

        elif plot_type['value'] == 'Combined Zonal Slice':
            # Only available in combined mode
            if not (ocean_mode and atmos_mode):
                print("Combined Zonal Slice only available in combined ocean/atmospheric mode")
                return

            # Get data from both domains at the clicked location
            ocean_data = combined_data['ocean']['data']
            ocean_depth = combined_data['ocean']['depth']
            ocean_lon2d = combined_data['ocean']['lon2d']
            ocean_lat2d = combined_data['ocean']['lat2d']

            atmos_data = combined_data['atmos']['data']
            atmos_depth = combined_data['atmos']['depth']  # This is pressure
            atmos_lon2d = combined_data['atmos']['lon2d']
            atmos_lat2d = combined_data['atmos']['lat2d']

            # Find nearest grid points in each domain
            ocean_lon_vals = ocean_lon2d.values if hasattr(ocean_lon2d, 'values') else ocean_lon2d
            ocean_lat_vals = ocean_lat2d.values if hasattr(ocean_lat2d, 'values') else ocean_lat2d

            # Wrap click_lon to match ocean grid longitude range
            ocean_lon_min = np.min(ocean_lon_vals)
            ocean_lon_max = np.max(ocean_lon_vals)
            click_lon_ocean = click_lon
            if click_lon < ocean_lon_min:
                click_lon_ocean = click_lon + 360.0
            elif click_lon > ocean_lon_max:
                click_lon_ocean = click_lon - 360.0

            iy_ocean, ix_ocean = find_nearest_2d(ocean_lon_vals, ocean_lat_vals, click_lon_ocean, click_lat)

            # Convert atmos arrays to numpy if they're xarray DataArrays
            atmos_lon_vals = atmos_lon2d.values if hasattr(atmos_lon2d, 'values') else atmos_lon2d
            atmos_lat_vals = atmos_lat2d.values if hasattr(atmos_lat2d, 'values') else atmos_lat2d

            # Wrap click_lon to match atmospheric grid longitude range
            atmos_lon_min = np.min(atmos_lon_vals)
            atmos_lon_max = np.max(atmos_lon_vals)
            click_lon_atmos = click_lon
            if click_lon < atmos_lon_min:
                click_lon_atmos = click_lon + 360.0
            elif click_lon > atmos_lon_max:
                click_lon_atmos = click_lon - 360.0

            iy_atmos, ix_atmos = find_nearest_2d(atmos_lon_vals, atmos_lat_vals, click_lon_atmos, click_lat)

            # Extract zonal slices (all longitudes at the selected latitude)
            ocean_lat_val = (ocean_lat2d[iy_ocean, ix_ocean].values if
                             hasattr(ocean_lat2d[iy_ocean, ix_ocean], 'values') else
                             ocean_lat2d[iy_ocean, ix_ocean])
            atmos_lat_val = (atmos_lat2d[iy_atmos, ix_atmos].values if
                             hasattr(atmos_lat2d[iy_atmos, ix_atmos], 'values') else
                             atmos_lat2d[iy_atmos, ix_atmos])

            # Get zonal slices
            ocean_zonal = ocean_data[:, iy_ocean, :]  # shape: (z_levels, x)
            ocean_zonal_lon = ocean_lon2d[iy_ocean, :]  # shape: (x,)
            # Convert to numpy if xarray
            if hasattr(ocean_zonal_lon, 'values'):
                ocean_zonal_lon = ocean_zonal_lon.values

            atmos_zonal = atmos_data[:, iy_atmos, :]  # shape: (lev, lon)
            atmos_zonal_pressure = atmos_depth[:, iy_atmos, :]  # shape: (lev, lon)
            atmos_zonal_lon = atmos_lon2d[iy_atmos, :]  # shape: (lon,)
            # Convert to numpy if xarray
            if hasattr(atmos_zonal_lon, 'values'):
                atmos_zonal_lon = atmos_zonal_lon.values

            # Create meshgrids for pcolormesh
            # For atmospheric data: lon vs normalized vertical (0 to 0.5)
            # For ocean data: lon vs normalized vertical (0.5 to 1.0)

            # Get dimensions
            n_atmos_lev = atmos_zonal.shape[0]
            n_atmos_lon = atmos_zonal.shape[1]
            n_ocean_lev = ocean_zonal.shape[0]

            # Create normalized vertical coordinates
            atmos_norm_vert = np.linspace(0, 0.5, n_atmos_lev)
            ocean_norm_vert = np.linspace(0.5, 1.0, n_ocean_lev)

            # Create meshgrids
            atmos_lon_mesh, atmos_norm_mesh = np.meshgrid(atmos_zonal_lon, atmos_norm_vert)
            ocean_lon_mesh, ocean_norm_mesh = np.meshgrid(ocean_zonal_lon, ocean_norm_vert)

            # Create combined plot
            fig_combined, ax_combined = plt.subplots(figsize=(14, 10))

            # Set black background to better see ocean topography
            ax_combined.set_facecolor('black')

            # Determine color bounds for atmospheric data
            vmin_atmos_slice = atmos_vmin if atmos_vmin is not None else float(np.nanmin(atmos_zonal))
            vmax_atmos_slice = atmos_vmax if atmos_vmax is not None else float(np.nanmax(atmos_zonal))

            # Determine color bounds for ocean data
            vmin_ocean_slice = ocean_vmin if ocean_vmin is not None else float(np.nanmin(ocean_zonal))
            vmax_ocean_slice = ocean_vmax if ocean_vmax is not None else float(np.nanmax(ocean_zonal))

            # Plot atmospheric zonal slice (top half: 0 to 0.5)
            # Plot twice: once at original longitude, once shifted by -360
            pcm_atmos = ax_combined.pcolormesh(atmos_lon_mesh, atmos_norm_mesh, atmos_zonal,
                                               vmin=vmin_atmos_slice, vmax=vmax_atmos_slice,
                                               shading='auto', cmap='RdBu_r', alpha=1.0)
            ax_combined.pcolormesh(atmos_lon_mesh - 360, atmos_norm_mesh, atmos_zonal,
                                   vmin=vmin_atmos_slice, vmax=vmax_atmos_slice,
                                   shading='auto', cmap='RdBu_r', alpha=1.0)

            # Add contour lines to atmospheric slice (3 levels, ensure 0 is included if in range)
            if vmin_atmos_slice <= 0 <= vmax_atmos_slice:
                # Include 0 as the middle contour
                contour_levels_atmos = [vmin_atmos_slice * 0.5, 0.0, vmax_atmos_slice * 0.5]
            else:
                contour_levels_atmos = np.linspace(vmin_atmos_slice, vmax_atmos_slice, 3)

            # Plot contours with different styles: dashed for negative, solid for positive, thick for zero
            for level in contour_levels_atmos:
                if abs(level) < 1e-10:  # Zero contour
                    linewidth = 1.5
                    linestyle = '-'
                else:
                    linewidth = 0.5
                    linestyle = '--' if level < 0 else '-'
                ax_combined.contour(atmos_lon_mesh, atmos_norm_mesh, atmos_zonal,
                                    levels=[level], colors='black', linewidths=linewidth,
                                    linestyles=linestyle, alpha=0.5)
                ax_combined.contour(atmos_lon_mesh - 360, atmos_norm_mesh, atmos_zonal,
                                    levels=[level], colors='black', linewidths=linewidth,
                                    linestyles=linestyle, alpha=0.5)

            # Plot ocean zonal slice (bottom half: 0.5 to 1.0)
            # Plot twice: once at original longitude, once shifted by +360
            pcm_ocean = ax_combined.pcolormesh(ocean_lon_mesh, ocean_norm_mesh, ocean_zonal,
                                               vmin=vmin_ocean_slice, vmax=vmax_ocean_slice,
                                               shading='auto', cmap='RdBu_r', alpha=1.0)
            ax_combined.pcolormesh(ocean_lon_mesh + 360, ocean_norm_mesh, ocean_zonal,
                                   vmin=vmin_ocean_slice, vmax=vmax_ocean_slice,
                                   shading='auto', cmap='RdBu_r', alpha=1.0)

            # Add contour lines to ocean slice (3 levels, ensure 0 is included if in range)
            if vmin_ocean_slice <= 0 <= vmax_ocean_slice:
                # Include 0 as the middle contour
                contour_levels_ocean = [vmin_ocean_slice * 0.5, 0.0, vmax_ocean_slice * 0.5]
            else:
                contour_levels_ocean = np.linspace(vmin_ocean_slice, vmax_ocean_slice, 3)

            # Plot contours with different styles: dashed for negative, solid for positive, thick for zero
            for level in contour_levels_ocean:
                if abs(level) < 1e-10:  # Zero contour
                    linewidth = 1.5
                    linestyle = '-'
                else:
                    linewidth = 0.5
                    linestyle = '--' if level < 0 else '-'
                ax_combined.contour(ocean_lon_mesh, ocean_norm_mesh, ocean_zonal,
                                    levels=[level], colors='black', linewidths=linewidth,
                                    linestyles=linestyle, alpha=0.5)
                ax_combined.contour(ocean_lon_mesh + 360, ocean_norm_mesh, ocean_zonal,
                                    levels=[level], colors='black', linewidths=linewidth,
                                    linestyles=linestyle, alpha=0.5)
            ax_combined.contour(ocean_lon_mesh, ocean_norm_mesh, ocean_zonal,
                                levels=contour_levels_ocean, colors='black', linewidths=0.5, alpha=0.5)
            ax_combined.contour(ocean_lon_mesh + 360, ocean_norm_mesh, ocean_zonal,
                                levels=contour_levels_ocean, colors='black', linewidths=0.5, alpha=0.5)

            # Add horizontal line at the interface
            ax_combined.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7,
                                label='Surface')

            # Set up axes
            ax_combined.set_xlim(-180, 180)  # Zoom to -180 to 180
            ax_combined.set_ylim(0, 1)
            ax_combined.invert_yaxis()  # Invert so 0 is at top, 1 is at bottom
            ax_combined.set_ylabel('Normalized Vertical Coordinate', fontsize=11)
            ax_combined.set_xlabel('Longitude', fontsize=11)

            # Create custom y-tick labels
            yticks = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
            yticklabels = []

            # Sample pressure and depth values from a valid location
            # For atmosphere: use middle longitude
            mid_lon_idx_atmos = n_atmos_lon // 2

            # For ocean: use the clicked location (ix_ocean) to get depth profile
            # This gives us the actual depth values at that specific location
            ocean_depth_for_labels = ocean_depth[:, iy_ocean, ix_ocean]

            for yt in yticks:
                if yt < 0.5:
                    # Atmosphere: interpolate pressure
                    idx = int((yt / 0.5) * (n_atmos_lev - 1))
                    if idx < n_atmos_lev:
                        pressure_val = atmos_zonal_pressure[idx, mid_lon_idx_atmos]
                        yticklabels.append(f'{pressure_val:.0f} hPa')
                    else:
                        yticklabels.append('')
                elif yt == 0.5:
                    yticklabels.append('Surface')
                else:
                    # Ocean: interpolate depth
                    idx = int(((yt - 0.5) / 0.5) * (n_ocean_lev - 1))
                    if idx < n_ocean_lev:
                        depth_val = ocean_depth_for_labels[idx]
                        if not np.isnan(depth_val):
                            yticklabels.append(f'{depth_val:.0f} m')
                        else:
                            yticklabels.append('')
                    else:
                        yticklabels.append('')

            ax_combined.set_yticks(yticks)
            ax_combined.set_yticklabels(yticklabels, fontsize=9)

            # Add labels for atmosphere/ocean regions
            ax_combined.text(0.01, 0.25, 'OCEAN', transform=ax_combined.transAxes,
                             fontsize=11, color='darkblue', fontweight='bold', alpha=0.7,
                             rotation=90, va='center')
            ax_combined.text(0.01, 0.75, 'ATMOSPHERE', transform=ax_combined.transAxes,
                             fontsize=11, color='darkred', fontweight='bold', alpha=0.7,
                             rotation=90, va='center')

            # Add colorbars
            fig_combined.colorbar(pcm_atmos, ax=ax_combined, orientation='horizontal',
                                  pad=0.08, aspect=50, shrink=0.6, label=f'{atmosvarname}')
            fig_combined.colorbar(pcm_ocean, ax=ax_combined, orientation='horizontal',
                                  pad=0.15, aspect=50, shrink=0.6, label=f'{oceanvarname}')

            # Add title
            title_text = (f'Combined Zonal Slice\nAtmos at lat={atmos_lat_val:.2f}° | '
                          f'Ocean at lat={ocean_lat_val:.2f}°')
            ax_combined.set_title(title_text, fontsize=12, pad=10)

            ax_combined.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.show()

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot vertical profile from ocean or atmospheric fields.')
    parser.add_argument('--hfile', required=False, help='NetCDF file containing h variable (ocean mode only)')
    parser.add_argument('--oceanfile', required=False,
                        help='NetCDF file containing ocean variable (e.g., background error, analysis, etc.)')
    parser.add_argument('--atmosfile', required=False,
                        help='NetCDF file containing atmospheric variable (e.g., increments, analysis, etc.)')
    parser.add_argument('--oceanvarname', required=False, help='Ocean variable name to plot (e.g., Temp, Salt)')
    parser.add_argument('--atmosvarname', required=False, help='Atmospheric variable name to plot (e.g., T_inc, u_inc, v_inc)')
    parser.add_argument('--variance', action='store_true', help='Set if the file contains variance instead of standard deviation')
    parser.add_argument('--gridfile', required=False, help='NetCDF file containing 2D lon/lat variables (ocean mode only)')
    parser.add_argument('--obsfile', required=False, help='IODA observation file with ombg and oman (ocean mode only)')
    parser.add_argument('--level', required=False, type=int, default=None,
                        help='Model level to plot (default: 0 for ocean [surface], last level for atmos [near surface])')
    parser.add_argument('--bounds', required=False, type=str, default=None,
                        help='Colorbar bounds as vmin,vmax. Use = or quotes for negatives: --bounds="-0.5,0.5"')
    parser.add_argument('--ocean_bounds', required=False, type=str, default=None,
                        help='Ocean colorbar bounds as vmin,vmax. Use = or quotes: --ocean_bounds="-0.3,0.3"')
    parser.add_argument('--atmos_bounds', required=False, type=str, default=None,
                        help='Atmospheric colorbar bounds as vmin,vmax. Use = or quotes: --atmos_bounds="-2,2"')
    parser.add_argument('--atmos_to_celsius', action='store_true',
                        help='Convert atmospheric temperature from Kelvin to Celsius')
    args = parser.parse_args()

    # Parse bounds arguments
    vmin, vmax = None, None
    if args.bounds:
        try:
            vmin, vmax = map(float, args.bounds.split(','))
        except ValueError:
            parser.error("--bounds must be in format 'vmin,vmax' (e.g., -0.5,0.5)")

    ocean_vmin, ocean_vmax = None, None
    if args.ocean_bounds:
        try:
            ocean_vmin, ocean_vmax = map(float, args.ocean_bounds.split(','))
        except ValueError:
            parser.error("--ocean_bounds must be in format 'vmin,vmax' (e.g., -0.3,0.3)")

    atmos_vmin, atmos_vmax = None, None
    if args.atmos_bounds:
        try:
            atmos_vmin, atmos_vmax = map(float, args.atmos_bounds.split(','))
        except ValueError:
            parser.error("--atmos_bounds must be in format 'vmin,vmax' (e.g., -2,2)")

    # Validate arguments
    if args.oceanfile and args.atmosfile:
        # Both modes - future feature
        if not args.oceanvarname:
            parser.error("--oceanvarname is required when using --oceanfile")
        if not args.atmosvarname:
            parser.error("--atmosvarname is required when using --atmosfile")
        if not args.hfile or not args.gridfile:
            parser.error("--hfile and --gridfile are required for combined ocean/atmospheric mode")
    elif args.oceanfile:
        # Ocean mode only
        if not args.oceanvarname:
            parser.error("--oceanvarname is required when using --oceanfile")
        if not args.hfile or not args.gridfile:
            parser.error("--hfile and --gridfile are required for ocean mode")
    elif args.atmosfile:
        # Atmospheric mode only
        if not args.atmosvarname:
            parser.error("--atmosvarname is required when using --atmosfile")
        if args.hfile or args.gridfile or args.obsfile:
            parser.error("--hfile, --gridfile, and --obsfile are not used in atmospheric mode (when using only --atmosfile)")
    else:
        parser.error("Must specify either --oceanfile, --atmosfile, or both")

    main(args.hfile, args.oceanfile, args.atmosfile, args.oceanvarname, args.atmosvarname,
         args.variance, args.gridfile, obsfile=args.obsfile, level=args.level,
         vmin=vmin, vmax=vmax, ocean_vmin=ocean_vmin, ocean_vmax=ocean_vmax,
         atmos_vmin=atmos_vmin, atmos_vmax=atmos_vmax, atmos_to_celsius=args.atmos_to_celsius)
