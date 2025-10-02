# GDAS Ocean Observation Statistics Time Series Visualizer

This application creates time series visualizations from ocean observation data. It supports both pre-computed CSV statistics and direct NetCDF file processing to generate comprehensive plots showing RMSE, Bias, and observation counts over time for different ocean regions and depth layers.

## Features

- **Dual data sources**: Process both CSV statistics files and original NetCDF observation files
- **Depth stratification**: For NetCDF mode, generate statistics by depth layers (e.g., 0-10m, 10-50m, 50-100m)
- **Ocean basin analysis**: Automatically classifies observations by ocean basin using built-in basin flags
- **Automatic instrument discovery**: Processes all available instruments when no specific instrument is specified
- **Multi-ocean analysis**: Generates plots for Global, Atlantic, Pacific, Indian, Arctic, and Southern oceans
- **Multiple variables**: Analyzes both quality-controlled (`ombg_qc`) and non-quality-controlled (`ombg_noqc`) observations
- **HTML summary**: Creates an interactive HTML index page to browse all generated plots
- **Flexible instrument selection**: Supports specific instruments, wildcards, or automatic discovery
- **LETKF support**: Optional processing of LETKF diagnostic files

## Requirements

- Python 3.6+
- Required packages:
  ```bash
  pip install pandas matplotlib jinja2 netcdf4 numpy
  ```

## Usage

### CSV Mode (Pre-computed Statistics)

#### Basic Usage (Auto-discovery mode)

Process all available instruments automatically:

```bash
./gdassoca_obsstats.py --exps /path/to/experiment/COMROOT --dirout output_directory
```

#### Specific Instrument

Process a single instrument:

```bash
./gdassoca_obsstats.py --exps /path/to/experiment/COMROOT --inst sss_smap_l2 --dirout output_directory
```

#### Wildcard Pattern

Process instruments matching a pattern:

```bash
./gdassoca_obsstats.py --exps /path/to/experiment/COMROOT --inst "sss*" --dirout output_directory
```

### NetCDF Mode (Original Data Processing)

#### Process In-Situ Temperature Profiles by Depth

Generate statistics for 0-10m depth layer:

```bash
./gdassoca_obsstats.py --source netcdf --exps /path/to/experiment/COMROOT --inst "insitu_temp_profile*" --depth-layers 0-10 --dirout output_directory
```

#### Multiple Depth Layers

Process multiple depth layers simultaneously:

```bash
./gdassoca_obsstats.py --source netcdf --exps /path/to/experiment/COMROOT --inst "insitu_temp_profile_argo" --depth-layers 0-10 10-50 50-100 100-200 --dirout output_directory
```

#### Process Salinity Data

Generate statistics for salinity observations:

```bash
./gdassoca_obsstats.py --source netcdf --exps /path/to/experiment/COMROOT --inst "insitu_salt*" --depth-layers 0-10 --dirout output_directory
```

#### Auto-discovery with NetCDF

Process all available in-situ instruments:

```bash
./gdassoca_obsstats.py --source netcdf --exps /path/to/experiment/COMROOT --inst "insitu*" --depth-layers 0-10 10-50 --dirout output_directory
```

### Multiple Experiments

Compare multiple experiments:

```bash
./gdassoca_obsstats.py --exps exp1/COMROOT exp2/COMROOT --dirout comparison_output
```

### Include LETKF Diagnostics

Process both standard and LETKF diagnostic files:

```bash
./gdassoca_obsstats.py --exps /path/to/experiment/COMROOT --dirout output_directory --letkf
```

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--exps` | Yes | Path(s) to experiment COMROOT directory(ies) |
| `--inst` | No | Instrument name or wildcard pattern (if not provided, all instruments are processed) |
| `--dirout` | Yes | Output directory for plots and HTML index |
| `--letkf` | No | Include LETKF diagnostic files in processing |

## Input Data Structure

The application expects CSV files in the following directory structure:

```
COMROOT/
├── gdas.YYYYMMDD/
│   └── HH/
│       └── analysis/
│           └── ocean/
│               ├── diags/
│               │   └── gdas.tHHz.ocn.{instrument}.stats.csv
│               └── letkf/diags/  (optional, with --letkf)
│                   └── gdas.tHHz.ocn.{instrument}.stats.csv
```

### CSV File Format

Expected CSV columns:
- `Exp`: Experiment name
- `Variable`: Variable type (ombg_noqc, ombg_qc)
- `Ocean`: Ocean region (Global, Atlantic, Pacific, Indian, Arctic, Southern)
- `date`: Date in YYYYMMDDHH format
- `RMSE`: Root Mean Square Error
- `Bias`: Observation bias
- `ObsErr`: Observation error
- `EnsStd`: Ensemble standard deviation (for LETKF)
- `Count`: Number of observations

## Output

The script generates **time series plots** as PNG files with RMSE, bias, and observation count statistics for each instrument, ocean basin, and quality control combination.

### Creating an Interactive HTML Index

After generating plots, create an organized web interface using the separate `generate_index.py` tool:

```bash
# Basic HTML index generation
python generate_index.py /path/to/output/directory

# With custom title and description
python generate_index.py /path/to/output/directory --title "My Experiment Results" --experiments "Experiment vs Control"
```

The HTML index provides:
- Dropdown selectors for observation types and ocean basins
- Depth layer controls for depth-stratified data
- Side-by-side QC vs No-QC comparisons
- Responsive design for desktop and mobile viewing

See `README_generate_index.md` for detailed documentation on the HTML index generator.

## Supported Instruments

The application automatically detects available instruments, including:

- **Sea Surface Temperature (SST)**: VIIRS, AVHRR sensors
- **Sea Surface Salinity (SSS)**: SMAP L2, SMOS L2
- **Sea Ice Concentration**: AMSR2 North/South
- **In-situ observations**: Argo profiles, surface drifters
- **Radar altimetry**: Various satellite altimeters (RADS data)

## Examples

### Example 1: Process all instruments from a single experiment
```bash
./gdassoca_obsstats.py \
    --exps /home/user/experiments/gfs17-sss/COMROOT/gfs17-sss \
    --dirout /home/user/output/all_instruments
```

### Example 2: Compare SSS instruments between two experiments
```bash
./gdassoca_obsstats.py \
    --exps /path/to/exp1/COMROOT /path/to/exp2/COMROOT \
    --inst "sss*" \
    --dirout /home/user/output/sss_comparison
```

### Example 3: Process specific instrument with LETKF data
```bash
./gdassoca_obsstats.py \
    --exps /path/to/experiment/COMROOT \
    --inst sst_viirs_npp_l3u \
    --dirout /home/user/output/sst_analysis \
    --letkf
```

## Plot Interpretation

Each generated plot contains three panels:

1. **RMSE Panel (top)**:
   - Shows observation-minus-background RMSE over time
   - For LETKF experiments: also shows ensemble spread and ensemble spread + observation error

2. **Bias Panel (middle)**:
   - Shows observation-minus-background bias over time
   - Ideally should be close to zero

3. **Count Panel (bottom)**:
   - Shows number of observations assimilated over time
   - Useful for understanding data availability

## Troubleshooting

### Common Issues

1. **No CSV files found**: Check that the experiment path is correct and contains the expected directory structure
2. **Empty plots**: Verify that the CSV files contain data for the requested ocean regions and variables
3. **Missing template file**: Ensure the HTML template exists at `../../templates/gdassoca_obsstats_template.html`

### Debug Information

The application prints useful debug information including:
- File paths being searched
- List of files found
- Instruments discovered
- Processing status for each ocean region

## File Naming Convention

The instrument names are automatically extracted from CSV filenames using the pattern:
```
gdas.t{HH}z.ocn.{instrument_name}.stats.csv → {instrument_name}
```

For example:
- `gdas.t00z.ocn.sss_smap_l2.stats.csv` → `sss_smap_l2`
- `gdas.t06z.ocn.sst_viirs_npp_l3u.stats.csv` → `sst_viirs_npp_l3u`

## Workflow Integration

This application integrates well with GDAS ocean data assimilation workflows:

1. Run your GDAS experiment with ocean observation processing
2. Use `gdassoca_obsstats.py` to generate statistics plots
3. Use `generate_index.py` to create an interactive HTML interface
4. Compare different experiments or configurations
5. Share results via the organized web interface

**Recommended Workflow:**
```bash
# Step 1: Generate plots from NetCDF data
python gdassoca_obsstats.py --source netcdf --exps /path/to/experiment --inst "insitu_temp*" --depth-layers 0-10 10-50 --dirout results

# Step 2: Create interactive HTML index
python generate_index.py results --title "Ocean Temperature Analysis" --experiments "Experiment vs Control"

# Step 3: Open results/index.html in your browser
```

## Files in this Directory

- **`gdassoca_obsstats.py`** - Main statistics generation script (CSV and NetCDF processing)
- **`generate_index.py`** - Standalone HTML index generator for organizing figure galleries
- **`README.md`** - This documentation
- **`README_generate_index.md`** - Detailed documentation for the HTML index generator
