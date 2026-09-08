# LETKF Verification

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
| `letkf_verification.html` | The report — every figure embedded, stands alone |
| `scorecard.md` | Every experiment against the reference, with a paired Wilcoxon test over cycles |
| `figs/obs_*.png` | Departures, consistency, rank histograms, spread&ndash;skill, profiles by region |
| `figs/state_*.png` | Increment and spread profiles, increment / spread-reduction / inflation maps, vertical sections of the increment |
| `figs/verif_maps_<product>.png` | Product beside each experiment's background and analysis |
| `figs/verif_diff_<product>.png` | Model minus product |
| `figs/cycle_*.png` | Everything against cycle: obs fit, obs counts, gridded-analysis scores, background drift |
| `cache/` | The per-cycle reductions; the report never re-reads model output |

## Configuration

`experiments.example.yaml` is the template. The parts worth knowing:

| Key | What it does |
|---|---|
| `experiments:` | The registry. `kind: var` or `kind: letkf`; `stem` is a template in `{Ymd}` and `{HH}` and the suite globs beneath it |
| `cycles:` | An explicit list, or `{start, stop, step}` with an inclusive stop |
| `grid:` | soca gridspec, or a ~7 MB slim copy made by `make_gridfile.py` |
| `regions:` | Named lat/lon boxes for the regional profiles; `global` is always included |
| `verification:` | One directory per gridded product — they come from unrelated archives, so each names its own |
| `depth_max:` | Cut every depth panel at N metres; a view setting, so it needs no recompute |
| `sections:` | `zonal` latitudes and `meridional` longitudes to cut vertical sections of the increment and background along. A zonal line is one grid row, a true latitude circle only to ~64&deg;N; anything above 65&deg;N is drawn but flagged by `preflight.py`, by `plot_statespace.py` and on the panel |
| `map_limits:` | Fixed (vmin, vmax) per state variable/level, so a map figure's color scale holds across cycles and experiments instead of being recomputed per figure; optional, per-hemisphere override for ice thickness/snow depth |

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
