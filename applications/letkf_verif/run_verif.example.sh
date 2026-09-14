#!/bin/bash
#SBATCH --job-name=letkf_verif_2exp
#SBATCH --account=da-cpu
#SBATCH --qos=batch
##SBATCH --partition=hera
#SBATCH --nodes=1
#SBATCH --ntasks=96
##SBATCH --cpus-per-task=60
#SBATCH --mem=300GB
#SBATCH --time=01:30:00
#SBATCH --output=letkf_verif_2exp.%j.log

# Example sbatch script: precompute + build the LETKF-verif comparison
# report for every experiment in experiments.yaml.
#
# Copy this (and experiments.example.yaml -> your own experiments.yaml)
# somewhere of your own and edit CFG/OUT/GRIDSPEC below. Everything else --
# the experiment list, the cycle count, the per-experiment worker counts and
# the --cache list for stage 2 -- is read from the yaml at run time, so
# adding an experiment or a cycle there is the whole edit.
#
# `grid:` in experiments.yaml is expected to already exist next to it
# (lv_grid.nc, resolved relative to the config file) -- this script slims
# GRIDSPEC down to it with make_gridfile.py if it is not there yet. Only ever
# reads GRIDSPEC; never regenerates an lv_grid.nc that already exists.
#
# Parallelism, all through existing CLI options:
#   stage 1  every experiment precomputed CONCURRENTLY (independent roots and
#            cache dirs), each with --jobs = the number of cycles it actually
#            resolves on disk (one wave, bounded by the slowest cycle), capped
#            so the sum stays within the cores this job was given.
#   stage 2  rejoin first (it rewrites the page cache the rest read, --jobs
#            over cycles), then the figure stages -- obs-space, state-space,
#            binned departures, time series, fronts -- and the scorecard
#            CONCURRENTLY (they only
#            share the read-only cache and write different files), then the
#            report once all of them are in. State-space draws its cycles
#            --jobs-way in parallel too.
#   caching  every stage skips work whose inputs have not changed: the
#            rejoin when the page-cache file is newer than its sources, and
#            each plot stage per cycle (figs/.fresh-<stage>.json records
#            what was drawn from which cache files). A rerun that added one
#            cycle redraws that cycle and the across-date sequences only.
#            BUILD_OPTS="--force" redoes everything.
#
# Sizing: worst case is sum(cycles per experiment) precompute workers at
# once; --ntasks=96 covers two experiments x ~30 cycles with margin, and
# the cap below shrinks the per-experiment --jobs if the yaml grows past
# that. A measured worker peaks around 3 GB RSS (sacct on a real run), so
# --mem=300GB is ~5 GB x 60 workers. OMP/BLAS threads are pinned to 1 so
# worker processes do not also fan out into threads.
#
# NOT `set -e`: build_comparison.py's report stage returns a non-zero,
# advisory exit code when it can't find a figure it expected (e.g. ensemble
# spread panels for a var-kind experiment) -- it still writes the report and
# scorecard either way. A hard exit here would throw away a finished report
# over a status flag that just means "go check what's flagged."
set -uo pipefail

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/gdas-marine-viz/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

APP=/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gfs-dev/gdas-marine-viz/applications/letkf_verif
OUT=/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gfs-dev/gdas-marine-viz/applications/letkf_verif/compare-exps
CFG=$OUT/experiments.yaml
# UTC hours of the cycles to draw per-date state / background / gridded-
# product / frontal figures for (the report puts a date menu over them); the
# latest cycle is always included. "all" for every cycle.
HOURS=00
# Extra build_comparison.py options, e.g. "--force" (redo everything).
BUILD_OPTS=""
# Extra precompute_experiment.py options: "--force" recomputes every cached
# cycle (needed once after a change to what the cache holds -- e.g. a new
# verification product or observation bins; otherwise old cycles keep the
# old content and only new cycles get the new fields).
PRECOMPUTE_OPTS="--force"
# Full soca_gridspec.nc (~190 MB) to slim down to $OUT/lv_grid.nc. Any
# cycle's works -- lon/lat/area/mask2d are the static model grid, not a
# per-cycle field. Point this at your own experiment's bmatrix output.
GRIDSPEC=/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gfs-dev/cp06.torchbalance/COMROOT/cp06.torchbalance/gdas.20251219/06/bmatrix/ocean/soca_gridspec.nc

cd "$APP"

# ---------------------------------------------------------------------------
# Read the yaml: experiment names, cycle count, and how many of the cycles
# each experiment has a directory for (the number of precompute workers it
# can actually keep busy). Uses the same loader the python stages use, so
# relative paths and stems resolve identically.
# ---------------------------------------------------------------------------
eval "$(python3 - "$CFG" <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from lv_common import load_config
cfg = load_config(sys.argv[1], None, None, None)
cycles = [str(c) for c in cfg['cycles']]
print('NCYC=%d' % len(cycles))
print('EXPS="%s"' % ' '.join(e.name for e in cfg['experiments']))
print('declare -A CYCLES_FOR')
for e in cfg['experiments']:
    n = sum(os.path.isdir(e.dir_for(c)) for c in cycles)
    print('CYCLES_FOR[%s]=%d' % (e.name, n))
PY
)" || { echo "!! could not parse $CFG"; exit 1; }

NPROC=${SLURM_CPUS_ON_NODE:-$(nproc)}
NEXP=$(wc -w <<<"$EXPS")
echo "== config: $CFG"
echo "   $NCYC cycle(s), $NEXP experiment(s): $EXPS"
echo "   $NPROC core(s) available to this job"

if [ ! -f "$OUT/lv_grid.nc" ]; then
    echo "== grid: slimming $GRIDSPEC -> $OUT/lv_grid.nc =="
    python3 make_gridfile.py --grid "$GRIDSPEC" --out "$OUT/lv_grid.nc"
fi

echo "== preflight =="
python3 preflight.py "$CFG" || true

# ---------------------------------------------------------------------------
# stage 1: precompute, every experiment concurrently
# ---------------------------------------------------------------------------
echo
echo "== stage 1: precompute, every experiment concurrently =="
declare -A PIDS
CACHES=()
per_exp_cap=$(( NPROC / (NEXP > 0 ? NEXP : 1) ))
[ "$per_exp_cap" -lt 1 ] && per_exp_cap=1
for exp in $EXPS; do
    CACHES+=(--cache "$OUT/precompute-$exp/cache")
    n=${CYCLES_FOR[$exp]}
    if [ "$n" -eq 0 ]; then
        echo "-- $exp: none of the $NCYC cycles resolve on disk -- skipping precompute"
        continue
    fi
    jobs_exp=$(( n < per_exp_cap ? n : per_exp_cap ))
    python3 precompute_experiment.py "$CFG" \
        --experiment "$exp" \
        --outdir "$OUT/precompute-$exp" \
        --skip-preflight \
        --jobs "$jobs_exp" $PRECOMPUTE_OPTS \
        > "$OUT/precompute-$exp.log" 2>&1 &
    PIDS[$exp]=$!
    echo "-- launched $exp (pid ${PIDS[$exp]}, $n of $NCYC cycles on disk, ${jobs_exp}-way, log: $OUT/precompute-$exp.log)"
done

fail=0
for exp in "${!PIDS[@]}"; do
    if wait "${PIDS[$exp]}"; then
        echo "-- precompute ok: $exp"
    else
        echo "!! precompute FAILED: $exp -- see $OUT/precompute-$exp.log"
        fail=1
    fi
done
if [ $fail -ne 0 ]; then
    echo "!! at least one experiment's precompute failed -- stopping before stage 2"
    exit 1
fi

# ---------------------------------------------------------------------------
# stage 2: build the comparison. build_comparison.py runs its sub-stages in
# sequence; --skip lets one invocation run exactly one of them, which is how
# the independent ones are run side by side here. (Each of those invocations
# ends its log with build_comparison.py's "the observation join was skipped"
# note -- that is about the invocation, not the run: the rejoin happened in
# 2a, and the scorecard and report read the joined sample from the cache.)
# ---------------------------------------------------------------------------
ALL_STAGES="rejoin obsspace obsbins statespace timeseries fronts stability scorecard report"
stage() {
    # stage <name>: run only that sub-stage of build_comparison.py
    local only=$1 skips=()
    for s in $ALL_STAGES; do
        [ "$s" != "$only" ] && skips+=(--skip "$s")
    done
    # shellcheck disable=SC2086  # BUILD_OPTS is a word list on purpose
    python3 build_comparison.py "$CFG" "${CACHES[@]}" \
        --outdir "$OUT/page" --jobs "$(( NCYC < NPROC ? NCYC : NPROC ))" \
        --hours "$HOURS" $BUILD_OPTS "${skips[@]}"
}

echo
echo "== stage 2a: rejoin observations across experiments =="
stage rejoin
echo "rejoin exit code: $?"

echo
echo "== stage 2b: figures + scorecard, concurrently =="
declare -A SPIDS
for s in obsspace obsbins statespace timeseries fronts stability scorecard; do
    stage "$s" > "$OUT/page/stage-$s.log" 2>&1 &
    SPIDS[$s]=$!
    echo "-- launched $s (pid ${SPIDS[$s]}, log: $OUT/page/stage-$s.log)"
done
for s in "${!SPIDS[@]}"; do
    if wait "${SPIDS[$s]}"; then
        echo "-- $s ok"
    else
        echo "!! $s exited non-zero -- see $OUT/page/stage-$s.log (report is still built)"
    fi
done

echo
echo "== stage 2c: report =="
stage report
echo "report exit code: $? (non-zero here just means a figure was flagged"
echo "missing -- see build_report.py's summary above; the report and tarball"
echo "are written regardless)"

echo
echo "done -- report at $OUT/page/letkf_verification.html"
echo "        tarball at $OUT/page/letkf_verification.tar (pages + linked figures)"
echo "        scorecard at $OUT/page/scorecard.md"
