# Obs Stats Deep

Generates density plots, mean/RMSE profiles, and spatial maps of OmB (observation minus background) and OmA (observation minus analysis) for in situ vertical profiles (Argo temperature and salinity). Results are stratified by ocean basin and depth layer.

## Outputs

- **Density plots** — 2-D histograms of OmB / OmA vs depth (per basin)
- **Profile stats** — mean and RMSE profiles of OmB and OmA (per basin)
- **Spatial maps** — gridded mean and RMSE on a 4° global map for three depth layers (0–10 m, 0–300 m, 300–2000 m)

Output PNGs are written to `<output-dir>/<basin>/` and `<output-dir>/global/`.

## Usage

```bash
python plot_ts.py \
    --start-date 20260320 \
    --end-date   20260324 \
    --ioda-path  './data/gdas.{date}/*/analysis/ocean/diags/insitu_temp_profile_argo.nc'
```

| Argument | Default | Description |
|---|---|---|
| `--start-date` | required | Start date `YYYYMMDD` |
| `--end-date` | required | End date `YYYYMMDD` |
| `--ioda-path` | required | Path template with `{date}` placeholder |
| `--variable` | `temp` | Variable to plot (`temp` or `salt`) |
| `--output-dir` | `.` | Directory for output plots |

## Input files

IODA diagnostic NetCDF files produced by SOCA/GSI, expected to contain groups `MetaData`, `ombg`, and optionally `oman` and `EffectiveQC0`. Only QC-passed observations (`EffectiveQC0 == 0`) are used.
