## Applications

This directory contains the main diagnostic applications used in this repository. Each application is organized into its own subdirectory with relevant scripts, configuration files, and documentation.

### 1. Oceanview
An interactive map-based diagnostic tool for visualizing ocean observations and simulated observations.

- **Directory:** `oceanview/`
- **Main Components:**
  - `obsview` (executable)

### 2. Obs Stats Time Series
A tool for generating time series statistics of o-b.

- **Directory:** `obsstats_ts/`
- **Main Components:**
  - `gdassoca_obsstats.py` (main script for generating statistics)

### 3. Cycle Diagnostics
A diagnostic application for generating figures related to state-space and obs-space verification for a specific cycle.

- **Directory:** `state_space_vrfy/`
- **Main Components:**
gen_eva_obs_yaml.py  marine_eva_post.py  run_vrfy.py  soca_vrfy.py  vrfy_script.py
  - `run_vrfy.py` (main script for running verification diagnostics)
  - `soca_vrfy.py` (supporting script for verification)
  - `vrfy_script.py` (additional verification utilities)
  - `gen_eva_obs_yaml.py` (generates configuration file for EVA)
  - `marine_eva_post.py` (run EVA)


