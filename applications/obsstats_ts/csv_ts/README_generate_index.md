# HTML Index Generator for Observation Statistics Figures

This script automatically generates an interactive HTML index page to organize and display ocean observation statistics figures.

## Features

- 📊 **Automatic Organization**: Scans directories for PNG files and organizes them by observation type, ocean basin, and QC status
- 🌊 **Depth Stratification**: Supports depth-layered data with automatic depth layer detection
- 🎨 **Interactive Interface**: Dropdown selectors for observation types, ocean basin buttons, and depth layer controls
- 📱 **Responsive Design**: Modern, mobile-friendly interface with hover effects
- 🔍 **Smart Parsing**: Automatically parses filenames to extract metadata (instrument, ocean, QC status, depth)

## Filename Format

The script expects PNG files with this naming convention:
```
instrument_ombg_[qc|noqc]_Ocean[_depth].png
```

**Examples:**
- `sst_avhrrf_mb_l3u_ombg_noqc_Atlantic.png` - Surface SST data
- `insitu_temp_profile_argo_ombg_qc_Pacific_0_10m.png` - Temperature profile with depth layer
- `sss_smap_l2_ombg_qc_Global.png` - Sea surface salinity data

**Supported elements:**
- **Instruments**: Any instrument name (e.g., `sst_avhrrf_mb_l3u`, `insitu_temp_profile_argo`)
- **QC Status**: `qc` (quality controlled) or `noqc` (no quality control)
- **Ocean Basins**: `Arctic`, `Atlantic`, `Indian`, `Pacific`, `Southern`, `Global`
- **Depth Layers**: Optional depth specification (e.g., `0_10m`, `10_50m`, `50_100m`)

## Usage

### Basic Usage
```bash
python generate_index.py /path/to/figures/directory
```

### With Custom Title and Description
```bash
python generate_index.py gfsv17sss --title "GFS v17 Ocean Statistics" --experiments "GFS v17 with SSS vs without SSS assimilation"
```

### Custom Output Filename
```bash
python generate_index.py /path/to/figures --output custom_index.html
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `directory` | Directory containing PNG figure files | (required) |
| `--title` | Title for the HTML page | "Time Series of (Observation - Background) Statistics" |
| `--experiments` | Experiment description subtitle | "Marine Observation Statistics" |
| `--output` | Output HTML filename | `index.html` |

## Output

The script generates an interactive HTML file with:

1. **Header Section**: Custom title and experiment description
2. **Control Panel**:
   - Observation type dropdown (auto-populated from files)
   - Depth layer selector (appears for instruments with depth data)
     - "0-bottom (all depths)" - includes all observations from surface to seafloor
     - Specific depth ranges (e.g., "0_10m", "10_50m") - observations within those depth bounds
   - Ocean basin buttons (Global, Arctic, Atlantic, Indian, Pacific, Southern)
3. **Image Gallery**: Side-by-side comparison of QC vs No-QC statistics
4. **Smart Navigation**: Images update dynamically based on selections

## Example Directory Structure

```
experiment_output/
├── sst_avhrrf_mb_l3u_ombg_noqc_Arctic.png
├── sst_avhrrf_mb_l3u_ombg_qc_Arctic.png
├── insitu_temp_profile_argo_ombg_noqc_Atlantic_0_10m.png
├── insitu_temp_profile_argo_ombg_qc_Atlantic_0_10m.png
├── insitu_temp_profile_argo_ombg_noqc_Atlantic_10_50m.png
├── insitu_temp_profile_argo_ombg_qc_Atlantic_10_50m.png
└── ... (more PNG files)
```

After running the script:
```
experiment_output/
├── index.html  ← Generated HTML index
├── sst_avhrrf_mb_l3u_ombg_noqc_Arctic.png
├── sst_avhrrf_mb_l3u_ombg_qc_Arctic.png
└── ... (other PNG files)
```

## Features in Detail

### Automatic Detection
- **Instruments**: Detects all unique instrument types from filenames
- **Ocean Basins**: Identifies which oceans have data for each instrument
- **Depth Layers**: Automatically detects depth-stratified data and creates depth selectors

### Interactive Interface
- **Type Selection**: Dropdown with all available observation types
- **Depth Control**: Appears only for instruments with depth-layered data
- **Ocean Navigation**: Buttons for quick ocean basin selection
- **Visual Feedback**: Active state highlighting and hover effects

### Error Handling
- Validates directory existence and PNG file availability
- Shows helpful error messages for missing data combinations
- Graceful handling of partial data (missing QC or No-QC versions)

## Requirements

- Python 3.6+
- Standard library only (no external dependencies)

## Integration

This script works seamlessly with the `gdassoca_obsstats.py` tool:

1. Generate figures with `gdassoca_obsstats.py`
2. Run `generate_index.py` on the output directory
3. Open the generated `index.html` in a browser

## Browser Compatibility

The generated HTML works with all modern browsers:
- Chrome/Chromium
- Firefox
- Safari
- Edge

The interface is responsive and works on desktop and mobile devices.
