# Marine GDAS Verification Tools

## Unit Testing
This `oceanview` application relies on older interactive tools and probably requires a different Python environment. Test it separately, from the root directory of the repository:
```console
conda activate oceanview  # Does not work on HPC yet
pytest --disable-warnings -v tests/oceanview/test_oceanview.py
```

The other application require the `EVA` modules to be loaded to work:
```console
module use modulefiles
module load EVA/orion
pytest --disable-warnings -v tests/test_coding_style.py
pytest --disable-warnings -v tests/obsstats_ts/
```

## Simple Observation Space Statistics
A quick way to generate o-b stats and compare experiments
```
python gdassoca_obsstats.py --exps .../COMROOT/cp1 .../COMROOT/cp2 --inst '*' --dirout cp1vscp2
```
The above will generate time series of o-b RMSEs, Bias, and obs count for multiple regions. It will also generate an HTML document to facilitate viewing figures.

---

## Advanced Timeseries Analysis with `plot_timeseries.py`

The `plot_timeseries.py` tool provides comprehensive timeseries analysis capabilities for marine data assimilation diagnostics. It supports both single-experiment analysis and multi-experiment comparisons with advanced spatial filtering options.

### Features
- **Flexible Data Processing**: Can read existing NetCDF statistics files or generate them from IODA files
- **Spatial Filtering**: Support for depth binning and ocean basin filtering for in-situ observations
- **Multi-Experiment Comparisons**: Compare multiple experiments with validation for statistical compatibility
- **Stats-Only Processing**: Efficient generation of statistics without scatter plots
- **Configuration Validation**: Ensures observation spaces are comparable across experiments

### Basic Usage
```bash
cd applications/obsstats_maps
python plot_timeseries.py config.yaml
```

### Configuration Formats

#### Single Experiment Configuration
Create timeseries plots for different observation spaces in one experiment:

```yaml
# single_experiment_config.yaml
observation_spaces:
  - name: "SST AVHRR-MB ombg"
    ioda_data_path: "/path/to/gdas.*/*/analysis/ocean/diags/sst_avhrrf_mb_l3u.nc"
    varname: "sst"
    geovar_group: "ombg"
    time_interval: 21600  # 6 hours
    stats_output: "sst_avhrrf_mb_ombg_statistics.nc"

  - name: "Argo Temperature - Surface"
    ioda_data_path: "/path/to/gdas.*/*/analysis/ocean/diags/insitu_temp_profile_argo.nc"
    varname: "temp"
    geovar_group: "ombg"
    time_interval: 21600
    depth_bins: [[0, 50]]  # Surface observations only (0-50m)
    stats_output: "argo_surface_ombg_statistics.nc"

output_dir: "./single_experiment_plots"
title: "GDAS Marine DA Analysis"
```

#### Multi-Experiment Comparison Configuration
Compare the same observation spaces across multiple experiments:

```yaml
# multi_experiment_config.yaml
experiments:
  "Control Run":
    observation_spaces:
      - name: "Surface Drifters"
        ioda_data_path: "/path/to/control/gdas.*/*/analysis/ocean/diags/insitu_temp_surface_drifter.nc"
        varname: "temp"
        geovar_group: "ombg"
        time_interval: 21600
        stats_output: "control_drifters_ombg_statistics.nc"

  "Test Run":
    observation_spaces:
      - name: "Surface Drifters"  # Same name for comparison
        ioda_data_path: "/path/to/test/gdas.*/*/analysis/ocean/diags/insitu_temp_surface_drifter.nc"
        varname: "temp"
        geovar_group: "ombg"
        time_interval: 21600
        stats_output: "test_drifters_ombg_statistics.nc"

output_dir: "./multi_experiment_plots"
title: "Control vs Test Comparison"
```

### Advanced Spatial Filtering

#### Depth Binning
Filter in-situ observations by depth ranges:
```yaml
- name: "Argo Temperature - Mid-depth"
  ioda_data_path: "/path/to/argo.nc"
  varname: "temp"
  geovar_group: "ombg"
  depth_bins: [[50, 200]]  # Only observations between 50-200m depth
```

#### Ocean Basin Filtering
Filter observations by ocean basins (1=Atlantic, 2=Pacific, 3=Indian, 4=Arctic, 5=Southern):
```yaml
- name: "Drifters - Pacific Only"
  ioda_data_path: "/path/to/drifters.nc"
  varname: "temp"
  geovar_group: "ombg"
  ocean_basins: [2]  # Pacific Ocean only
```

### Output
The tool generates:

**Single Experiment Mode:**
- Individual 3-panel timeseries plots for each observation space
- Plots show mean, standard deviation, and observation count over time

**Multi-Experiment Mode:**
- Comparison plots with all experiments on the same figure
- Configuration validation ensures statistical comparability
- Different colors for each experiment with legends

### Examples

#### Example 1: Basic OMBG Analysis
```bash
# Analyze observation minus background (ombg) statistics
python plot_timeseries.py ombg_config.yaml
```

#### Example 2: Depth-Stratified Analysis
```bash
# Compare surface vs deep observations
python plot_timeseries.py depth_analysis_config.yaml
```

#### Example 3: Multi-Experiment Comparison
```bash
# Compare control vs experimental runs
python plot_timeseries.py comparison_config.yaml
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Observation space identifier | Required |
| `ioda_data_path` | Glob pattern for IODA files | Required |
| `varname` | Variable name (sst, temp, salt, etc.) | 'sst' |
| `geovar_group` | NetCDF group (ObsValue, ombg, etc.) | 'ObsValue' |
| `time_interval` | Time interval in seconds | 3600 |
| `depth_bins` | Depth ranges [[min,max], ...] | None |
| `ocean_basins` | Ocean basin codes [1,2,3,4,5] | None |
| `stats_output` | Output NetCDF statistics file | Auto-generated |

### Tips for Multi-Experiment Comparisons
- Ensure observation spaces have identical configurations except for `ioda_data_path` and `stats_output`
- Use meaningful experiment names for clear legends
- The tool validates configuration compatibility and shows clear error messages for incompatible setups

---

## Using `oceanview`
An interactive tool, mostly meant for insitu obs.
```console
./oceanview -v waterTemperature -i insitu_profile_argo.2021070412.nc4
```

---

## How to generate the EVA and State space figures

#### Create a scratch place to run `run_vrfy.py`. This script will generate a bunch of sbatch scripts and logs.
```
mkdir /somewhere/scratch
cd /somewhere/scratch
ln -s /path/to/run_vrfy.py .   # to be sorted out properly in the future
cp /path/to/vrfy_config.yaml .
module use ...
module load EVA/....
```
---
#### Edit `vrfy_config.yaml`
It's actually read as a jinja template to render `pslot` if necessary. Anything that is a templated variable in `vrfy_jobcard.sh.j2` can be added to the yaml below.
```yaml
pslot: "nomlb"
start_pdy: '20210701'
end_pdy: '20210701'
cycs: ["00", "06", "12", "18"]
run: "gdas"
homegdasmarineviz: "/path/to/gdas-marine-viz"
base_exp_path: "/path/to/comroot/{{ pslot }}/COMROOT/{{ pslot }}"
plot_ensemble_b: "OFF"
plot_parametric_b: "OFF"
plot_background: "OFF"
plot_increment: "ON"
plot_analysis: "OFF"
plot_letkf_ensemble: "OFF"
eva_plots: "ON"
eva_letkf_plots: "OFF"
qos: "batch"
hpc: "hercules"
eva_module: "EVA/orion"
```

---
#### Run the application
```python run_vrfy.py vrfy_config.yaml```
This will generate and submit the job cards for all the **cycles** defined by `cycs`, from `start_pdy` to `end_pdy`.

---
#### View the results
The script above will create figures according to the specified configuration. An `index.html` file is also generated to facilitate browsing through the results.

##### On hera
`X2GO` is probably the simplest option to start a browser and view the results.

##### On MSU
Two options that works:
- Use the MSU `Dashboard` application
- Tar the results and copy to your local machine
