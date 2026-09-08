#!/bin/bash
#SBATCH --job-name=letkf_verif_2exp
#SBATCH --account=da-cpu
#SBATCH --qos=debug
##SBATCH --partition=hera
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=300GB
#SBATCH --time=00:30:00
#SBATCH --output=letkf_verif_2exp.%j.log

# Example sbatch script: precompute + build the LETKF-verif comparison
# report for three experiments, cp06.torchbalance, 3dvar-rt and letkf3.
#
# Copy this (and experiments.example.yaml -> your own experiments.yaml)
# somewhere of your own and edit CFG/OUT/EXPS/GRIDSPEC below for your
# experiments -- nothing here is specific to the suite beyond that. See
# README.md for the two-stage pipeline this drives.
#
# `grid:` in experiments.yaml is expected to already exist next to it
# (lv_grid.nc, resolved relative to the config file) -- this script slims
# GRIDSPEC down to it with make_gridfile.py if it is not there yet, so a
# fresh OUT directory (no lv_grid.nc committed) works without a separate
# manual step. Only ever reads GRIDSPEC; never regenerates an lv_grid.nc
# that already exists, so editing it after the fact is never silently
# undone by a rerun.
#
# Parallelism, two layers, both using existing CLI options -- no code
# changes:
#   - the experiments are precomputed CONCURRENTLY (independent roots,
#     independent cache dirs, nothing shared) instead of one after another
#   - each experiment's own cycles are computed with --jobs 30 (compute_cycle.py
#     already supports this: ProcessPoolExecutor over cycles, which are
#     independent and I/O-bound)
# --jobs is set to the number of cycles in experiments.yaml (30 here, up from
# 7), not an arbitrary smaller number: profiling a real 7-cycle run showed
# precompute at --jobs 4 taking 220s of a 328s total job (67%) purely because
# 7 cycles over 4 workers is 2 sequential waves (ceil(7/4)) -- each cycle
# costs 90-140s on its own, so that second wave was pure waste. --jobs ==
# cycle count runs every cycle in one wave, bounded by the single slowest
# cycle rather than by however many waves ceil(cycles/jobs) works out to.
# (3dvar-rt and letkf3 both typically resolve far fewer of their cycles than
# $JOBS here -- 3dvar-rt fails most of preflight's directory checks, and
# letkf3's window only actually has 4 cycles with ocean/ice DA out of the 34
# in `cycles:` -- so most of their own workers exit almost immediately;
# harmless, just reserved-but-idle capacity for them.)
#
# --cpus-per-task/--mem below were sized for 2 experiments (2 x 30 = 60
# worker processes worst case) and have NOT been re-checked now that EXPS has
# 3 -- nominal worst case is 3 x 30 = 90, which would oversubscribe the 60
# reserved cores if all three actually hit 30-way at once. In practice
# letkf3 (4 real cycles) and 3dvar-rt (a handful) are unlikely to peak
# alongside cp06.torchbalance's real 30-way, so this probably still runs, just
# not with the same "one process per core" headroom the original 60/300GB
# figures assumed -- worth bumping (and re-checking against `sinfo`'s node
# limits, and getting a fresh `sacct` MaxRSS reading) if this turns out to be
# a real bottleneck rather than a theoretical one.
#
# Worst case *for a single experiment* is still 30 worker processes, hence
# --cpus-per-task=60 (2x that, not yet 3x -- see above) -- checked against
# this cluster's default partition (`sinfo`: u1-compute, 192 CPUs / 385 GB
# per node), so 60 fits on one node with headroom. --mem=300GB is NOT the
# naive "8GB/worker" continued from the 14-worker version, though -- that
# would ask for 480GB, over that node's 385 GB ceiling, and the job would
# simply never schedule. `sacct` on the actual 8-worker run showed real peak
# RSS was only ~24GB (~3GB/worker) -- the original 8GB/worker figure was
# never tight, just untested. 300GB (~5GB of 60 workers, ~67% margin over the
# measured figure) is sized from that real number instead of guessing an
# unworkable one forward. OMP/BLAS threads are pinned to 1 below so those
# worker processes don't each also fan out into threads and oversubscribe
# the node. Re-tune --jobs (and cpus/mem, checking them against your own
# partition's per-node limits) to match your own experiments.yaml's cycle
# count if you change EXPS or the cycle list -- more workers than cycles
# just means some exit immediately, so it is safe to round up, not down.
# Untested at this end: whether 30-way concurrency per experiment still
# scales as cleanly as 4-way did, or starts hitting filesystem I/O
# contention that the 4-way profiling run never reached.
#
# build_comparison.py (stage 2) also takes --jobs now, applied to its
# rejoin sub-stage (compute_cycle.py --rejoin over the same independent,
# I/O-bound cycles) -- the only sub-stage expensive enough per cycle to be
# worth it, and it reuses the same worker count for the same reason (one
# wave instead of two). The state-space figures right after it are the more
# expensive sub-stage overall, but build_comparison.py already renders only
# the latest cycle's state maps and profiles unconditionally (--latest; that
# is all the report ever embeds), so O(cycles) work there is gone rather
# than something --jobs still needs to chase. Stage 2 still runs after
# stage 1 fully finishes, reusing the same cores.
#
# NOT `set -e`: build_comparison.py's report-build stage returns a non-zero,
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
EXPS="cp06.torchbalance 3dvar-rt letkf3"
JOBS=34  # match the number of cycles in $CFG; see the parallelism note above
# Workers per experiment, sized to the cycles each actually resolves rather
# than a blanket $JOBS for all three. Unlisted -> $JOBS.
declare -A JOBS_FOR=( [cp06.torchbalance]=34 [3dvar-rt]=16 [letkf3]=4 )
# Full soca_gridspec.nc (~190 MB) to slim down to $OUT/lv_grid.nc. Any
# cycle's works -- lon/lat/area/mask2d are the static model grid, not a
# per-cycle field. Point this at your own experiment's bmatrix output.
GRIDSPEC=/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gfs-dev/cp06.torchbalance/COMROOT/cp06.torchbalance/gdas.20251219/06/bmatrix/ocean/soca_gridspec.nc

cd "$APP"

if [ ! -f "$OUT/lv_grid.nc" ]; then
    echo "== grid: slimming $GRIDSPEC -> $OUT/lv_grid.nc =="
    python3 make_gridfile.py --grid "$GRIDSPEC" --out "$OUT/lv_grid.nc"
fi

echo "== preflight =="
python3 preflight.py "$CFG" || true

echo
echo "== stage 1: precompute, every experiment concurrently, $JOBS-way"
echo "   cycle parallelism within each (--jobs $JOBS) =="
declare -A PIDS
for exp in $EXPS; do
    jobs_exp=${JOBS_FOR[$exp]:-$JOBS}
    python3 precompute_experiment.py "$CFG" \
        --experiment "$exp" \
        --outdir "$OUT/precompute-$exp" \
        --skip-preflight \
        --jobs "$jobs_exp" \
        > "$OUT/precompute-$exp.log" 2>&1 &
    PIDS[$exp]=$!
    echo "-- launched $exp (pid ${PIDS[$exp]}, ${jobs_exp}-way, log: $OUT/precompute-$exp.log)"
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

echo
echo "== stage 2: build comparison (rejoin + figures + scorecard + report) =="
python3 build_comparison.py "$CFG" \
    --cache "$OUT/precompute-cp06.torchbalance/cache" \
    --cache "$OUT/precompute-3dvar-rt/cache" \
    --cache "$OUT/precompute-letkf3/cache" \
    --outdir "$OUT/page" \
    --jobs $JOBS
echo "build_comparison exit code: $? (non-zero here just means a figure was"
echo "flagged missing -- see build_report.py's summary above; the report and"
echo "scorecard are written regardless)"

echo
echo "done -- report at $OUT/page/letkf_verification.html"
echo "        scorecard at $OUT/page/scorecard.md"
