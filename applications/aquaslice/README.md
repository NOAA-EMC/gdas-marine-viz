# AquaSlice - Interactive Ocean and Atmospheric Field Viewer

Interactive visualization tool for exploring ocean and atmospheric model fields with vertical profiles, zonal/meridional slices, and observation overlays.

## Features

- **Interactive 2D Map**: Click to explore vertical structure at any location
- **Multiple Plot Types**: Vertical profiles, zonal slices, meridional slices
- **Observation Integration**: Overlay and compare with IODA observations (ocean mode)
- **Dynamic Color Scaling**: Interactive sliders to adjust colorbar bounds
- **Dual Mode Support**:
  - Ocean mode: MOM6 grid with depth coordinates
  - Atmospheric mode: Gaussian grid with pressure coordinates

## Usage

### Ocean Mode (Default)

Visualize ocean fields from MOM6 models:

```bash
python aquaslice.py \
    --hfile ocean.h.nc \
    --oceanfile ocean_bgvar.nc \
    --gridfile ocean_static.nc \
    --oceanvarname Temp \
    --variance \
    --level 0 \
    --obsfile ioda_obs.nc
```

**Required Arguments (Ocean Mode):**
- `--hfile`: NetCDF file containing layer thickness (`h` variable)
- `--oceanfile`: NetCDF file with the field to visualize
- `--gridfile`: NetCDF file with 2D longitude/latitude grids
- `--oceanvarname`: Variable name to plot (e.g., `Temp`, `Salt`)

**Optional Arguments:**
- `--variance`: Set if the file contains variance (will compute sqrt for stddev)
- `--level`: Model level for 2D surface plot (default: 0 = ocean surface)
- `--obsfile`: IODA observation file to overlay on map

### Atmospheric Mode

Visualize atmospheric fields from Gaussian grid models (e.g., FV3-based systems):

```bash
python aquaslice.py \
    --atmosfile atmos_inc.nc \
    --atmosvarname T_inc \
    --level 60
```

**Required Arguments (Atmospheric Mode):**
- `--atmosfile`: NetCDF file with atmospheric field (e.g., increments, analysis)
- `--atmosvarname`: Variable name to plot (e.g., `T_inc`, `u_inc`, `v_inc`, `sphum_inc`)

**Optional Arguments:**
- `--level`: Vertical level for 2D horizontal plot (default: last level = near surface)

**Note:** In atmospheric mode, `--hfile`, `--gridfile`, and `--obsfile` are not used and will trigger an error if specified.

### Combined Ocean and Atmospheric Mode

Display both ocean and atmospheric fields on the same figure (stacked vertically):

```bash
python aquaslice.py \
    --hfile ocean.h.nc \
    --oceanfile ocean_bgvar.nc \
    --atmosfile atmos_inc.nc \
    --gridfile ocean_static.nc \
    --oceanvarname Temp \
    --atmosvarname T_inc \
    --level 0
```

**Required Arguments:**
- `--hfile`: NetCDF file containing ocean layer thickness
- `--oceanfile`: NetCDF file with ocean field
- `--atmosfile`: NetCDF file with atmospheric field
- `--gridfile`: NetCDF file with 2D ocean grid
- `--oceanvarname`: Ocean variable name
- `--atmosvarname`: Atmospheric variable name

**Features:**
- Atmospheric field displayed on top subplot
- Ocean field displayed on bottom subplot
- Click on either subplot to generate vertical profiles/slices for that domain
- Domain selector menu to switch between ocean and atmosphere
- Observations overlaid on ocean subplot if `--obsfile` provided

## Atmospheric File Format

The atmospheric mode expects NetCDF files with the following structure:

**Dimensions:**
- `lat`: Gaussian grid latitude dimension
- `lon`: Gaussian grid longitude dimension
- `lev`: Number of vertical levels
- `time`: Time dimension

**Variables:**
- `lat(lat)`: 1D latitude array (degrees_north)
- `lon(lon)`: 1D longitude array (degrees_east)
- `<varname>(time, lev, lat, lon)`: 4D field to visualize

**Global Attributes:**
- `ak`: Hybrid sigma-pressure coefficient A (Pa) - array of length `lev+1`
- `bk`: Hybrid sigma-pressure coefficient B (dimensionless) - array of length `lev+1`

The pressure at each level is computed as: `p = ak + bk * ps` where `ps` is surface pressure (default: 101325 Pa).

## Interactive Controls

Once the visualization window opens:

1. **Plot Type Menu** (top right): Select between:
   - Vertical Profile
   - Zonal Slice (constant latitude)
   - Meridional Slice (constant longitude)
   - Observation Profile (ocean mode only, if observations provided)

2. **Color Range Slider** (top left): Adjust vmin/vmax for the 2D map

3. **Click on Map**: Generate the selected plot type at the clicked location

## Examples

### Ocean Background Error Stddev
```bash
python aquaslice.py \
    --hfile MOM.res.nc \
    --oceanfile ocn_bkg_stddev.nc \
    --gridfile ocean_static.nc \
    --oceanvarname Temp \
    --variance \
    --level 0
```

### Atmospheric Temperature Increment
```bash
python aquaslice.py \
    --atmosfile atminc.nc \
    --atmosvarname T_inc
```

Note: Without `--level`, defaults to last vertical level (near surface)

### Ocean Analysis with Observations
```bash
python aquaslice.py \
    --hfile MOM.res.nc \
    --oceanfile ocn_ana.nc \
    --gridfile ocean_static.nc \
    --oceanvarname Temp \
    --level 0 \
    --obsfile argo_profile_2024010100.nc
```

### Atmospheric Wind Increment
```bash
python aquaslice.py \
    --atmosfile atminc.nc \
    --atmosvarname u_inc \
    --level 60
```

Note: Specifying `--level 60` to view mid-troposphere instead of near-surface default

### GDAS Ocean Temperature with Argo Observations
```bash
python aquaslice.py \
    --oceanfile gdas.t00z.inst.f006.nc \
    --gridfile soca_gridspec_025.nc \
    --oceanvarname Temp \
    --hfile gdas.t00z.inst.f006.nc \
    --ocean_bounds="-2,31" \
    --obsfile insitu_temp_profile_argo.nc
```

### GDAS Combined Ocean and Atmospheric Temperature with Argo Observations
```bash
python aquaslice.py \
    --atmosfile gdas.t00z.atm.f006.nc \
    --atmosvarname tmp \
    --oceanfile gdas.t00z.inst.f006.nc \
    --gridfile soca_gridspec_025.nc \
    --oceanvarname Temp \
    --hfile gdas.t00z.inst.f006.nc \
    --ocean_bounds="-2,31" \
    --atmos_bounds="-10,31" \
    --atmos_to_celsius \
    --obsfile insitu_temp_profile_argo.nc
```

Note: `--atmos_to_celsius` converts atmospheric temperature from Kelvin to Celsius for easier comparison with ocean temperature. Combined mode displays both domains with a unified vertical profile view.

### Batch Mode: Generate Observation Profiles

Create PNG plots for all observation locations without interactive mode:

```bash
python aquaslice.py \
    --oceanfile gdas.t00z.inst.f006.nc \
    --gridfile soca_gridspec_025.nc \
    --oceanvarname Salt \
    --hfile gdas.t00z.inst.f006.nc \
    --obsfile insitu_salt_profile_argo.nc \
    --batch_obs_profiles \
    --no_plot_background
```

This creates a `obs_profiles/` directory with PNG files and a `obs_profiles.tar.gz` archive.

### Batch Mode: Generate Zonal Sections

Create zonal (constant latitude) section plots for a range of latitudes:

```bash
python aquaslice.py \
    --oceanfile gdas.t00z.inst.f006.nc \
    --gridfile soca_gridspec_025.nc \
    --oceanvarname Temp \
    --hfile gdas.t00z.inst.f006.nc \
    --batch_zonal_sections \
    --lat_start -60 \
    --lat_end 60 \
    --lat_step 10 \
    --ocean_bounds="-2,31" \
    --sections_output_dir zonal_sections
```

This creates zonal section plots at latitudes: -60°, -50°, ..., 50°, 60°.

### Batch Mode: Generate Meridional Sections

Create meridional (constant longitude) section plots for a range of longitudes:

```bash
python aquaslice.py \
    --oceanfile gdas.t00z.inst.f006.nc \
    --gridfile soca_gridspec_025.nc \
    --oceanvarname Temp \
    --hfile gdas.t00z.inst.f006.nc \
    --batch_meridional_sections \
    --lon_start -180 \
    --lon_end 180 \
    --lon_step 30 \
    --ocean_bounds="-2,31" \
    --sections_output_dir meridional_sections
```

This creates meridional section plots at longitudes: -180°, -150°, ..., 150°, 180°.

**Batch Section Options:**
- `--lat_start`, `--lat_end`, `--lat_step`: Define latitude range for zonal sections (step must be integer ≥ 1°)
- `--lon_start`, `--lon_end`, `--lon_step`: Define longitude range for meridional sections (step must be integer ≥ 1°)
- `--sections_output_dir`: Output directory for section plots (default: `sections`)
- `--ocean_bounds` or `--bounds`: Color scale bounds for the plots

**Notes:**
- Step sizes (`--lat_step` and `--lon_step`) must be integers of 1 degree or more
- Latitude/longitude values are rounded to the nearest integer in the output
- Filenames include variable name and use integer format: `zonal_section_Temp_lat+045.png`, `meridional_section_Salt_lon-120.png`

## Output

The tool creates interactive matplotlib figures with:
- Main window: 2D horizontal slice with interactive controls
- Secondary windows: Vertical profiles/slices based on user clicks

**Batch modes** create PNG files saved to disk instead of interactive plots.

## Requirements

- Python 3.7+
- xarray
- numpy
- matplotlib
- netCDF4

## Notes

- Ocean mode requires structured grids (MOM6 format)
- Atmospheric mode assumes Gaussian grids with hybrid coordinates
- Vertical axis is inverted for ocean (depth increases downward)
- Vertical axis is normal for atmosphere (pressure decreases upward)
- Color ranges are auto-scaled to data min/max but can be adjusted interactively
