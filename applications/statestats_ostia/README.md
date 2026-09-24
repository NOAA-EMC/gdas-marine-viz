# State Stats OSTIA

Validates GFS background SST (foundation temperature) and sea-ice concentration against OSTIA L4 analyses. Supports one or more experiment runs compared side-by-side on a common 0.5° grid.

## Outputs

| File | Description |
|---|---|
| `sst_ts.png` | Per-basin SST bias & RMSE time series |
| `ice_ts.png` | Per-hemisphere ice concentration bias & RMSE time series |
| `sst_maps_<run>.png` | Time-mean SST bias & RMSE spatial maps |
| `ice_maps_<run>_north/south.png` | Polar stereographic ice bias & RMSE maps |
| `ice_extent.png` | Arctic/Antarctic sea-ice extent time series |
| `ice_edge_error_north/south.png` | Integrated Ice Edge Error (IIEE) time series |
| `ice_edge/ice_edge_<hemi>_<date>.png` | Daily polar ice-edge contour maps |
| `basin_map.png` | RECCAP2 basin mask diagnostic |

## Usage

```bash
python3 compare_sfc_ostia.py \
    --ops-dir  /path/to/ops \
    --exps-dir /path/to/exp_a /path/to/exp_b \
    --ostia-dir /path/to/ostia_raw \
    --mask-file RECCAP2_region_masks_all.nc \
    --output-dir plots_sfc
```

| Argument | Default | Description |
|---|---|---|
| `--ops-dir` | *(optional)* | Base directory for operational run |
| `--exps-dir` | `parallel` | One or more experiment directories |
| `--ostia-dir` | `ostia_raw` | Directory containing OSTIA L4 NetCDF files |
| `--mask-file` | `RECCAP2_region_masks_all.nc` | RECCAP2 ocean basin mask |
| `--output-dir` | `plots_sfc` | Directory for output plots |
| `--variables` | `both` | `sst`, `ice`, or `both` |

## Input files

**Background:** `gdas.t{HH}z.sfcf006.nc` (ops) or `gdas.t{HH}z.sfc.f006.nc` (parallel), discovered recursively under each run directory. All available cycles for a given day are averaged before comparison.

**OSTIA:** Daily L4 files named `{YYYYMMDD}120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc`, organised as `<ostia-dir>/YYYY/MM/`.

**Basin mask:** RECCAP2 `RECCAP2_region_masks_all.nc` (1° grid, interpolated to 0.5°).
