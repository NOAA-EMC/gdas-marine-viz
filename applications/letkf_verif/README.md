# LETKF Verification

## TL;DR

```bash
# One directory per comparison: the config plus a symlink to the driver
mkdir /my/verifs/realtime && cd /my/verifs/realtime
cp /path/to/letkf_verif/experiments.yaml .                          # edit it
ln -s /path/to/letkf_verif/run_verif.sh .

# Precompute every experiment, then build the comparison page
sbatch run_verif.sh                       # sbatch --time=02:00:00 run_verif.sh to resize
#   -> precompute-<exp>/, page/letkf_verification.html, letkf_verification.tar, slurm-<jobid>.log
```

Env knobs: `PRECOMPUTE_OPTS=--force`, `BUILD_OPTS`, `HOURS`, `GRIDSPEC`, `VENV`.
Details in [Running it](#running-it-one-directory-per-comparison); cron-driven
updates in [Daily updates](#daily-updates).

Scores any number of marine DA experiments — LETKF or 3DVar, in any mix —
against a chosen reference, per 6-hourly cycle, and assembles a single
self-contained HTML report. Three kinds of evidence go into it:

- **Observation space** — O&minus;B and O&minus;A per obs type, on a *common
  sample* so thinning and QC differences cannot flatter one experiment; plus
  Desroziers ratios, rank histograms, CRPS and spread&ndash;skill.
- **State space** — increment magnitude and ensemble spread against depth,
  globally and by region; increment, spread-reduction and applied-inflation
  maps; the three stages of the ensemble spread (background, analysis,
  post-inflation).
- **Gridded analyses** — the surface state against three daily L4 products
  produced outside the system: CMEMS ADT, CMEMS SSS and OSTIA SST.
- **Monthly means against GREP** — optional. Three independent ocean
  reanalyses of the *same month*, full depth: heat and salt content through
  time, monthly-mean T/S/u/v sections and difference maps. The spread between
  the members is the yardstick a model&minus;GREP difference is read against.
  2020&ndash;2024 only.
- **Frontal-current placement** — optional, single-cycle regional maps of
  geostrophic speed against Copernicus ADT. A common absolute speed threshold
  within each box shows broad and branching-current footprints without forcing
  them into one artificial axis.

## Two stages

The expensive work is per experiment and is done once; everything the page needs
afterwards is read from that cache.

```bash
# 1. Precompute, one experiment per job. Give each its own --outdir so they
#    can run in parallel.
python3 precompute_experiment.py my_experiments.yaml \
        --experiment 3dvar_ctl --outdir /scratch/me/verif/3dvar &
python3 precompute_experiment.py my_experiments.yaml \
        --experiment letkf_v1  --outdir /scratch/me/verif/letkf &
wait

# 2. Build the comparison page across the caches.
python3 build_comparison.py my_experiments.yaml \
        --cache /scratch/me/verif/3dvar/cache \
        --cache /scratch/me/verif/letkf/cache \
        --outdir /scratch/me/verif/page
```

`preflight.py my_experiments.yaml` checks the python stack, the grid file, the
cartopy coastline data and whether every experiment's files actually resolve,
before you commit to a long job. `--experiment` scopes it to one.

**The config is always explicit**, and every relative path inside it — `grid:`,
an experiment `root:`, `outdir:` — resolves against the config file's own
directory. Nothing is ever relative to this application directory, so nothing is
ever written into the checkout. Copy `experiments.example.yaml` somewhere of
your own and edit it.

### Why step 2 re-reads the observations

A cache precomputed for one experiment alone has no cross-experiment common
sample: for it, "common" is just its own observations. Comparing two such caches
directly would score each experiment on a *different* set of observations —
which is the confound the common sample exists to remove. At the sample cycle
the LETKF thins sea-ice observations to ~9% of the 3DVar sample, and the subset
it keeps is the easier one, so the own-sample comparison reports **+0.9%** where
the like-for-like one reports **+5.5%**.

So `build_comparison.py` first rebuilds a genuine join across every experiment on
show. That re-reads only the observation files — seconds per cycle, against
minutes for a full pass. `--no-rejoin` skips it when those files are not
reachable; the scorecard and report then say plainly that the numbers are
own-sample. The per-experiment caches are never modified: the rejoined cycle is
written into the page's own output directory.

## Outputs

Everything lands under `--outdir`.

| File | Description |
|---|---|
| `letkf_verification.html` | The report, eleven linked pages (`_02_usage` … `_11_grep`); figures are linked from `figs/`, not embedded (`build_report.py --embed` inlines them) |
| `letkf_verification.tar` | The pages plus every figure they link, laid out so the links hold once unpacked; written on every run |
| `scorecard.md` | Every experiment against the reference, with a paired Wilcoxon test over cycles |
| `figs/obs_*.png` | Departures, consistency, rank histograms, spread&ndash;skill, profiles by region |
| `figs/atmos_maps_<cycle>.png`, `figs/atmos_region_<region>.png` | Atmospheric forcing over the ocean (10 m wind, stress, net heat flux, precipitation, 2 m temperature) from the coupled atmosphere history: per-cycle maps and regional means against cycle (section 10) |
| `figs/obsbins_{map,reg,sec}_<type>_<cycle\|all>.png` | Binned O&minus;B / O&minus;A / count / obs-error maps, observation-vs-model regressions, depth&times;latitude sections; per `--hours` cycle and pooled over every cycle |
| `figs/state_*.png` | Increment and spread profiles, increment / spread-reduction / inflation maps, vertical sections of the increment |
| `figs/verif_maps_<product>_<cycle>.png` | Product beside each experiment's background and analysis, one per rendered cycle (`--hours`, 00z plus the latest by default; the background-state and WOA figures are tagged the same way and the report puts a date menu over them) |
| `figs/verif_diff_<product>.png` | Model minus product |
| `figs/front_strong_<region>_<cycle>.png` | Optional geostrophic-speed footprint for one configured current box, one per rendered cycle (`--hours`) |
| `figs/seq_<realm>_incr_<field>_<cycle\|all>.png` | One increment map per field and cached cycle, all dates of a field on one colour scale; `_all` is the mean increment over every cached cycle (the systematic correction) on its own scale, the default in the report's date menu |
| `figs/cycle_*.png` | Everything against cycle: obs fit, obs counts, gridded-analysis scores, background drift |
| `figs/grep_content_<band>_region_<region>.png` | Heat and salt content through time over a depth band: the model at every cycle against each GREP reanalysis member's monthly mean, as the equivalent band-mean temperature and salinity (section 11) |
| `figs/grep_sections_<var>[_0-Nm]_<YYYYMM>.png`, `figs/grep_bias_{heat,salt}_<band>_<YYYYMM>.png` | Monthly-mean T/S/u/v sections with one column per GREP member beside each experiment, and the model-minus-member band-mean difference maps |
| `grep_content_<exp>.csv`, `grep_content_grep.csv`, `grep/*.npz` | The content series behind those figures, and the cached monthly reductions |
| `figs/cycle_ssh_stability_<exp>.png`, `ssh_stability_<exp>.{csv,json}` | SSH cycling-stability metrics per cycle and region (increment size, persistence, rejection, small/grid-scale variance, shock, on-track index) with fitted trends and a flagged verdict (section 07) |
| `cache/` | The per-cycle reductions (`<cycle>.json`, `_maps.npz`, `_obsbins.npz`); the report never re-reads model output |
| `figs/.fresh-<stage>.json` | What each plot stage drew from which inputs; a rerun skips units whose inputs have not changed (`--force` redraws) |

## Configuration

`experiments.example.yaml` is the template. The parts worth knowing:

| Key | What it does |
|---|---|
| `experiments:` | The registry. `kind: var` or `kind: letkf`; `stem` is a template in `{Ymd}` and `{HH}` and the suite globs beneath it |
| `cycles:` | An explicit list, or `{start, stop, step}` with an inclusive stop |
| `grid:` | soca gridspec, or a ~7 MB slim copy made by `make_gridfile.py` |
| `regions:` | Named lat/lon boxes for the regional profiles; `global` is always included |
| `verification:` | One directory per gridded product — they come from unrelated archives, so each names its own. `adt`, `sss`, `sst` and `icec` (OSTIA sea-ice fraction against `aice_h`); an experiment with no ocean history is scored for SST/SSS from the `sst_h`/`sss_h` in its sea-ice history |
| `obs_bins:` | `deg` (bin spacing), `limits` (colour-scale bounds per obs type or prefix), `layers` (depth layers the profile types' maps are also binned in; default 0–10 m, 0–300 m, 300 m–bottom) for the binned departures (section 09) |
| `frontal_analysis:` | Optional ADT-gradient diagnostic for the `--hours` cycles; configures current boxes and shared absolute speed thresholds. Each experiment also needs `analysis_pattern:` for its written ocean analysis state. |
| `depth_max:` | Cut every depth panel at N metres; a view setting, so it needs no recompute |
| `sections:` | `zonal` latitudes and `meridional` longitudes to cut vertical sections of the increment and background along. A zonal line is one grid row, a true latitude circle only to ~64&deg;N; anything above 65&deg;N is drawn but flagged by `preflight.py`, by `plot_statespace.py` and on the panel |
| `grep:` | Optional. Monthly means against the GREP multi-reanalysis ensemble (section 11). Presence of the block turns it on; GREP covers **2020&ndash;2024 only**, and a run outside that window is skipped silently. The field averaged is the ocean background valid at each cycle (`Experiment.background()`, f006 of cycle&minus;6 for `kind: var`), so a per-experiment `background:` override is honoured. `bands` (default 0&ndash;300 m and 300&ndash;1000 m), `members`, `section_hour` (cycles entering the section means, default one a day), `min_cycles`/`min_coverage` (what counts as a month). Velocity panels need `cos_rot`/`sin_rot` in the grid file &mdash; without them they are limited to transects south of the tripolar fold |
| `map_limits:` | Fixed (vmin, vmax) per state variable/level, so a map figure's color scale holds across cycles and experiments instead of being recomputed per figure; optional, per-hemisphere override for ice thickness/snow depth |

## Running it: one directory per comparison

The code stays in this checkout; a comparison lives in a directory of its own
holding the config, the grid and two symlinks:

```bash
mkdir /my/verifs/realtime && cd /my/verifs/realtime
cp /path/to/letkf_verif/experiments.example.yaml experiments.yaml   # and edit it
ln -s /path/to/letkf_verif/run_verif.sh /path/to/letkf_verif/run_daily.sh .
sbatch run_verif.sh                       # one build; slurm-<jobid>.log lands here
```

`run_verif.sh` reads the `experiments.yaml` in the directory it was submitted
from and writes everything beside it (`precompute-<exp>/`, `page/`, the
tarball); it finds the application through the symlink, so no path needs
editing. Slurm sizing beyond its header goes on the command line
(`sbatch --time=02:00:00 run_verif.sh`). `PRECOMPUTE_OPTS=--force`,
`BUILD_OPTS`, `HOURS`, `GRIDSPEC` and `VENV` are read from the environment.

### Daily updates

`./run_daily.sh` in that directory is the cron entry point. It advances the
config's cycle window to the newest complete cycle on disk (`newest_cycle.py`:
analysis directory, ocean increment and the following forecast all present, for
any configured experiment; `cycles:` must be the `{start, stop, step}` form,
`start` is yours, `stop` is what the script advances), submits `run_verif.sh`
with `sbatch --wait`, and on success stamps `daily/LAST_SUCCESS` and runs
`$RSYNC_HOOK` if set — the place for the rsync to the web server. A lock keeps
two runs from overlapping; logs and the Slurm output land under `daily/`.
`SBATCH_OPTS` carries per-comparison Slurm sizing; `SBATCH_CMD="echo sbatch"`
dry-runs the driver.

```
30 2 * * * cd /my/verifs/realtime && ./run_daily.sh >> daily/cron.log 2>&1
```

Because every stage caches per cycle and per figure, a daily run costs the new
cycles (about 4 min each in stage 1, in one parallel wave) plus a few minutes of
stage 2; a cycle that arrives late (an archive fetched afterwards) is picked up
by the next run, since absent cycles are never cached.

## Relationship to `statestats_ostia`

Both compare a model surface state to OSTIA, and both read the same
`ostia_raw/YYYY/MM/` layout. They answer different questions:

| | `statestats_ostia` | this application |
|---|---|---|
| source | GFS `sfc` f006 | soca ocean state |
| grid | regridded to 0.5° | model-native tripolar |
| fields | SST, ice concentration | SST, SSS, ADT |
| states | background | background **and** analysis |
| regions | RECCAP2 basins | config lat/lon boxes |
| also | IIEE, ice-edge maps | obs-space and state-space scoring in the same report |

Note that ADT and SSS are genuinely independent of a system assimilating ice
concentration, Argo profiles, drifter SST and AVHRR/VIIRS SST — no altimetry and
no satellite salinity go in. OSTIA is built from the same AVHRR and VIIRS
radiances that *are* assimilated, so the SST number here is a consistency check
rather than independent validation.

## Requirements

numpy, netCDF4, PyYAML, matplotlib, and **cartopy** for the maps; **scipy** is
optional and only the scorecard's significance test uses it, degrading to
reporting differences unmarked. On HPC these come from the EVA module
environment, as for the other applications in this repository.
