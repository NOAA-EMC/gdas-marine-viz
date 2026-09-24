## Applications

This directory contains the main diagnostic applications used in this repository. Each application is organized into its own subdirectory with relevant scripts, configuration files, and documentation.

### 1. Oceanview
An interactive map-based diagnostic tool for visualizing ocean observations and simulated observations.

- **Directory:** `oceanview/`
- **Main Components:**
  - `obsview` (executable)

### 2. Aquaslice
This application will eventually supersede `Oceanview`. It allows plotting slices of 3D fields along with in situ OMA and OMB values (currently only works for temperature).


### 3. Obs Stats Time Series
A tool for generating time series statistics of o-b.

- **Directory:** `obsstats_ts/`
- **Main Components:**
  - `gdassoca_obsstats.py` (main script for generating statistics)

### 4. Cycle Diagnostics
A diagnostic application for generating figures related to state-space and obs-space verification for a specific cycle.

- **Directory:** `state_space_vrfy/`
- **Main Components:**
  - `run_vrfy.py` (main script for running verification diagnostics)
  - `soca_vrfy.py` (supporting script for verification)
  - `vrfy_script.py` (additional verification utilities)
  - `gen_eva_obs_yaml.py` (generates configuration file for EVA)
  - `marine_eva_post.py` (run EVA)

### 5. Obs Stats Deep
Bins observation minus background (OMB) and observation minus analysis (OMA) statistics for in situ vertical profiles (Argo temperature and salinity), stratified by ocean basin and depth layer.

- **Directory:** `obsstats_deep/`
- **Main Components:**
  - `plot_ts.py` (density plots, mean/RMSE profiles, and spatial maps)

### 6. State Stats OSTIA
Compares GFS background SST and sea-ice concentration against OSTIA L4 analyses, supporting multiple experiment runs side-by-side.

- **Directory:** `statestats_ostia/`
- **Main Components:**
  - `compare_sfc_ostia.py` (per-basin time series, spatial maps, ice extent, and IIEE)

### 7. Obs Maps
Creates multiple frames containing maps of obs values for now.

- **Directory:** `obsstats_maps/`
- **Main Components:**
  - `plt_diags_maps.py` (script that generates multiple obs maps)

Usage:
```
python plt_diags_maps.py config.yaml
```
where `config.yaml` would look like this:
```yaml
yyyymm: '202107'
dd: '05'
cyc: '00'
time_interval: 300
save_dir: './test-frames'
varname: 'sst'
bounds: [-2, 35]
```

### 8. LETKF Verification
Scores any number of marine DA experiments — LETKF or 3DVar, in any mix — against a chosen reference, and assembles a single self-contained HTML report covering observation space (common-sample O-B/O-A, Desroziers, rank histograms, CRPS), state space (increment and ensemble-spread profiles and maps, applied inflation), and the surface state against CMEMS ADT, CMEMS SSS and OSTIA SST.

Work is split in two: an expensive per-experiment precompute that can run as independent parallel jobs, and a cheap step that builds the comparison page from those caches.

- **Directory:** `letkf_verif/`
- **Main Components:**
  - `preflight.py` (check the environment and inputs before a long job)
  - `precompute_experiment.py` (per-experiment cache; the expensive step)
  - `build_comparison.py` (rejoins the observations, then builds the page)

Overlaps `statestats_ostia` on SST only, and on a different grid — see `letkf_verif/README.md`.
