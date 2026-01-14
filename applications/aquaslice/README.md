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

## Output

The tool creates interactive matplotlib figures with:
- Main window: 2D horizontal slice with interactive controls
- Secondary windows: Vertical profiles/slices based on user clicks

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
