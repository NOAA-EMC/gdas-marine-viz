#!/bin/bash
#SBATCH --job-name=verif
#SBATCH --account=da-cpu
#SBATCH --qos=batch
##SBATCH --partition=hera
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:20:00
#SBATCH --output=slurm-%j.log

# The verification job: precompute + build the LETKF-verif comparison
# report for every experiment in experiments.yaml.
#
# Do NOT copy this: make a directory holding your experiments.yaml (and
# lv_grid.nc, or set GRIDSPEC), symlink this script and run_daily.sh into it,
# and `sbatch run_verif.sh` from there. The config is the one in the submit
# directory and everything is written beside it; the experiment list, the
# cycles, the per-experiment worker counts and the --cache list for stage 2
# are all read from the yaml at run time. Slurm sizing beyond the header
# below goes on the command line: `sbatch --nodes=4 --time=02:00:00
# run_verif.sh`.
#
# `grid:` in experiments.yaml is expected to already exist next to it
# (lv_grid.nc, resolved relative to the config file) -- this script slims
# GRIDSPEC down to it with make_gridfile.py if it is not there yet. Only ever
# reads GRIDSPEC; never regenerates an lv_grid.nc that already exists.
#
# Nodes: every piece of work is its own Slurm job step (srun), placed on the
# node of the allocation with the most room and sized for its worker pool:
# one worker per core it asks for, WORKER_GB of memory per worker plus one
# more share for the step's parent process. Slurm holds each step to that
# memory, so a step that outgrows it is killed alone (and `sacct -j <jobid>`
# shows which, with its MaxRSS) instead of taking the node down, and a step
# that does not fit beside the others yet waits for room. Any number of
# nodes works; with --exclusive --mem=0 each node is used whole. The batch
# script itself (yaml parsing, preflight, the report) runs on the first node,
# which keeps one share back for it. Outside Slurm (no SLURM_JOB_ID) the same
# work runs on this machine, sized to its cores and memory.
#
# Parallelism:
#   stage 1  precompute: every experiment's cycles on disk are split into
#            chunks, at most one node's worth of workers each and small
#            enough that there are at least as many chunks as nodes; every
#            chunk is a step (precompute_experiment.py --cycle ... --jobs
#            <chunk size>), all at once. Cycles are independent and cached
#            one file each, so chunks of one experiment do not collide.
#   stage 2  rejoin first (it rewrites the page cache the rest read), split
#            over cycles across the nodes the same way; then the figure
#            stages -- obs-space, state-space, binned departures, time
#            series, fronts, stability, GREP, background error -- and the
#            scorecard, one step each, CONCURRENTLY (they only share the
#            read-only cache and write different files), then the report
#            once all of them are in. Binned departures, state-space, fronts
#            and stability draw their cycles --jobs-way in parallel, with
#            --jobs set so the four of them fit the allocation together.
#   caching  every stage skips work whose inputs have not changed: the
#            rejoin when the page-cache file is newer than its sources, and
#            each plot stage per cycle (figs/.fresh-<stage>.json records
#            what was drawn from which cache files). A rerun that added one
#            cycle redraws that cycle and the across-date sequences only.
#            BUILD_OPTS="--force" redoes everything.
#
# Sizing: a measured worker peaks around 3 GB RSS (sacct on a real run);
# WORKER_GB=5 (the default) leaves margin. A step killed for memory says
# "oom-kill" in its log: raise WORKER_GB (fewer, roomier workers per node)
# or add nodes. Precompute wall time is one wave bounded by the slowest
# cycle as long as nodes x (node memory / WORKER_GB) covers every cycle on
# disk; past that the chunks queue for room and it takes more waves.
# OMP/BLAS threads are pinned to 1 so worker processes do not also fan out
# into threads.
#
# NOT `set -e`: build_comparison.py's report stage returns a non-zero,
# advisory exit code when it can't find a figure it expected (e.g. ensemble
# spread panels for a var-kind experiment) -- it still writes the report and
# scorecard either way. A hard exit here would throw away a finished report
# over a status flag that just means "go check what's flagged."
set -uo pipefail

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/gdas-marine-viz/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# The application is wherever this script lives; the config is $CFG from the
# environment (run_daily.sh sets it) or the one next to compare-exps below,
# and everything is written beside the config. One copy of this script
# therefore serves any experiments.yaml.
# Where the application is. Slurm runs a COPY of this script from its spool,
# so under sbatch BASH_SOURCE says nothing; the symlink named run_verif.sh in
# the directory it was submitted from does -- it points back at the checkout.
# (run_daily.sh exports APP outright; either way works.)
if [ -z "${APP:-}" ]; then
    self=$(readlink -f "${BASH_SOURCE[0]}")
    case "$self" in
        /var/spool/*|*/slurmd/*) self=$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}/run_verif.sh") ;;
    esac
    APP=$(cd "$(dirname "$self")" && pwd)
fi
if [ ! -f "$APP/build_comparison.py" ]; then
    echo "!! cannot locate the verification application (looked in $APP);" \
         "submit from a directory holding a run_verif.sh symlink to it, or export APP" >&2
    exit 1
fi
# The config is the experiments.yaml in the directory this was submitted
# from (or run from), unless CFG says otherwise; everything is written
# beside it.
CFG=${CFG:-${SLURM_SUBMIT_DIR:-$PWD}/experiments.yaml}
OUT=$(dirname "$CFG")
# UTC hours of the cycles to draw per-date state / background / gridded-
# product / frontal figures for (the report puts a date menu over them); the
# latest cycle is always included. "all" for every cycle.
HOURS=${HOURS:-00}
# Extra build_comparison.py options, e.g. "--force" (redo everything).
BUILD_OPTS=${BUILD_OPTS:-}
# Extra precompute_experiment.py options: "--force" recomputes every cached
# cycle (needed once after a change to what the cache holds -- e.g. a new
# verification product or observation bins; otherwise old cycles keep the
# old content and only new cycles get the new fields).
PRECOMPUTE_OPTS=${PRECOMPUTE_OPTS:-}
# Memory budget per worker process, GB (see Sizing above).
WORKER_GB=${WORKER_GB:-5}
# Full soca_gridspec.nc (~190 MB) to slim down to $OUT/lv_grid.nc. Any
# cycle's works -- lon/lat/area/mask2d are the static model grid, not a
# per-cycle field. Point this at your own experiment's bmatrix output.
GRIDSPEC=${GRIDSPEC:-/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gfs-dev/cp06.torchbalance/COMROOT/cp06.torchbalance/gdas.20251219/06/bmatrix/ocean/soca_gridspec.nc}

cd "$APP"

# ---------------------------------------------------------------------------
# The allocation: its nodes, cores and memory per node, and from those how
# many WORKER_GB shares each node holds (CAP; one share = one process).
# ---------------------------------------------------------------------------
NODES=() CPUS=() MEM_MB=()
if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null; then
    USE_SRUN=1
    mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    # SLURM_JOB_CPUS_PER_NODE reads like "192(x2)" or "192,64"
    IFS=, read -ra parts <<<"$SLURM_JOB_CPUS_PER_NODE"
    for p in "${parts[@]}"; do
        if [[ $p =~ ^([0-9]+)\(x([0-9]+)\)$ ]]; then
            for ((i = 0; i < BASH_REMATCH[2]; i++)); do CPUS+=("${BASH_REMATCH[1]}"); done
        else
            CPUS+=("$p")
        fi
    done
    # Not every site exports SLURM_MEM_PER_NODE for --mem=0 (Ursa does not);
    # then the job's memory is read back from Slurm: the job's allocated
    # memory (AllocTRES mem=, all nodes together, e.g. "mem=1500G"), or
    # failing that each node's RealMemory
    job_mem=
    if [ "${SLURM_MEM_PER_NODE:-}" = 0 ]; then
        unset SLURM_MEM_PER_NODE
    fi
    if [ -z "${SLURM_MEM_PER_NODE:-}${SLURM_MEM_PER_CPU:-}" ]; then
        job_mem=$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' |
            sed -n 's/^AllocTRES=//p' | tr ',' '\n' | awk -F= '$1 == "mem" {
                v = $2; u = substr(v, length(v))
                if (u ~ /[0-9]/) u = "M"; else v = substr(v, 1, length(v) - 1)
                f = (u == "K") ? 1 / 1024 : (u == "G") ? 1024 : (u == "T") ? 1048576 : 1
                printf "%d\n", v * f }')
    fi
    for ((i = 0; i < ${#CPUS[@]}; i++)); do
        if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
            MEM_MB+=("$SLURM_MEM_PER_NODE")
        elif [ -n "${SLURM_MEM_PER_CPU:-}" ]; then
            MEM_MB+=("$(( SLURM_MEM_PER_CPU * CPUS[i] ))")
        elif [ -n "$job_mem" ]; then
            MEM_MB+=("$(( job_mem / ${#CPUS[@]} ))")
        else
            m=$(scontrol show node "${NODES[i]}" | tr ' ' '\n' | sed -n 's/^RealMemory=//p')
            if [ -z "$m" ]; then
                echo "!! cannot tell this job's memory (no SLURM_MEM_PER_NODE," \
                     "SLURM_MEM_PER_CPU, AllocTRES mem or RealMemory); give the job --mem" >&2
                exit 1
            fi
            MEM_MB+=("$m")
        fi
    done
else
    USE_SRUN=0
    NODES=("$(hostname -s)")
    CPUS=("$(getconf _NPROCESSORS_ONLN)")
    if [ -r /proc/meminfo ]; then
        MEM_MB=("$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)")
    else
        MEM_MB=("$(( $(sysctl -n hw.memsize) / 1048576 ))")
    fi
fi
NNODE=${#NODES[@]}
if [ "${#CPUS[@]}" -ne "$NNODE" ]; then
    echo "!! $NNODE node(s) but cores for ${#CPUS[@]} (SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-})" >&2
    exit 1
fi
CAP=() TOTCAP=0 MAXW=
for ((i = 0; i < NNODE; i++)); do
    c=$(( MEM_MB[i] / (WORKER_GB * 1024) ))
    [ "$c" -gt "${CPUS[i]}" ] && c=${CPUS[i]}
    # the first node runs this script too: keep one share back for it
    [ "$i" -eq 0 ] && c=$(( c - 1 ))
    CAP+=("$c")
    TOTCAP=$(( TOTCAP + c ))
    # the most workers one step can have on ANY node, beside its parent
    { [ -z "$MAXW" ] || [ $(( c - 1 )) -lt "$MAXW" ]; } && MAXW=$(( c - 1 ))
done
if [ "$MAXW" -lt 1 ]; then
    echo "!! a node holds too few ${WORKER_GB} GB shares for a worker and its parent" \
         "(cores ${CPUS[*]}, memory ${MEM_MB[*]} MB); lower WORKER_GB or give the job more memory" >&2
    exit 1
fi

# place <shares>: the node with the most room left in this stage, into PICKED
LOAD=()
reset_load() { LOAD=(); for ((i = 0; i < NNODE; i++)); do LOAD+=(0); done; }
place() {
    local i best=0
    for ((i = 1; i < NNODE; i++)); do
        [ $(( CAP[i] - LOAD[i] )) -gt $(( CAP[best] - LOAD[best] )) ] && best=$i
    done
    LOAD[best]=$(( LOAD[best] + $1 ))
    PICKED=$best
}

# launch <log> <workers> <command...>: run the command in the background as
# one job step on the roomiest node, with <workers> + 1 cores and that many
# WORKER_GB shares of memory; its pid in LAUNCHED. (The SLURM_MEM_PER_* the
# batch script inherited are dropped so they cannot clash with --mem.)
launch() {
    local log=$1 w=$2 u
    shift 2
    u=$(( w + 1 ))
    place "$u"
    if [ "$USE_SRUN" -eq 1 ]; then
        env -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_GPU \
            srun --nodes=1 --ntasks=1 --exact --cpus-per-task="$u" \
                 --mem=$(( u * WORKER_GB * 1024 ))M \
                 --nodelist="${NODES[PICKED]}" "$@" > "$log" 2>&1 &
    else
        "$@" > "$log" 2>&1 &
    fi
    LAUNCHED=$!
    echo "   ${NODES[PICKED]}: $w worker(s), $(( u * WORKER_GB )) GB, log: $log"
}

# chunks <max> <word>...: the words on as few lines as keep each line at
# most <max> words long, the lines as even as possible
chunks() {
    local max=$1 n k start=0 size parts
    shift
    n=$#
    [ "$n" -eq 0 ] && return
    local words=("$@")
    parts=$(( (n + max - 1) / max ))
    for ((k = 0; k < parts; k++)); do
        size=$(( (n - start + parts - k - 1) / (parts - k) ))
        echo "${words[*]:start:size}"
        start=$(( start + size ))
    done
}

# words <string>: how many words it holds
words() { local list; read -ra list <<<"$1"; echo "${#list[@]}"; }

# chunk_size <n>: the chunk for splitting n cycles over the nodes -- at most
# one step's worth of workers, small enough to give every node a share
chunk_size() {
    local c=$(( $1 / NNODE ))
    [ "$c" -gt "$MAXW" ] && c=$MAXW
    [ "$c" -lt 1 ] && c=1
    echo "$c"
}

# ---------------------------------------------------------------------------
# Read the yaml: experiment names, the cycles, and which of them each
# experiment has a directory for (the cycles it can actually precompute).
# Uses the same loader the python stages use, so relative paths and stems
# resolve identically.
# ---------------------------------------------------------------------------
eval "$(python3 - "$CFG" <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from lv_common import load_config
cfg = load_config(sys.argv[1], None, None, None)
cycles = [str(c) for c in cfg['cycles']]
print('NCYC=%d' % len(cycles))
print('ALL_CYCLES="%s"' % ' '.join(cycles))
print('EXPS="%s"' % ' '.join(e.name for e in cfg['experiments']))
print('declare -A CYCLES_FOR')
for e in cfg['experiments']:
    print('CYCLES_FOR[%s]="%s"' % (e.name, ' '.join(cycles)))
PY
)" || { echo "!! could not parse $CFG"; exit 1; }

NEXP=$(words "$EXPS")
echo "== config: $CFG"
echo "   $NCYC cycle(s), $NEXP experiment(s): $EXPS"
echo "== allocation: $NNODE node(s), $WORKER_GB GB per worker"
for ((i = 0; i < NNODE; i++)); do
    echo "   ${NODES[i]}: ${CPUS[i]} cores, $(( MEM_MB[i] / 1024 )) GB, room for ${CAP[i]} process(es)"
done
[ "$USE_SRUN" -eq 0 ] && echo "   (not under Slurm: running everything on this machine)"

if [ ! -f "$OUT/lv_grid.nc" ]; then
    echo "== grid: slimming $GRIDSPEC -> $OUT/lv_grid.nc =="
    python3 make_gridfile.py --grid "$GRIDSPEC" --out "$OUT/lv_grid.nc"
fi

echo "== preflight =="
python3 preflight.py "$CFG" || true

# ---------------------------------------------------------------------------
# stage 1: precompute, every experiment's cycles in chunks across the nodes
# ---------------------------------------------------------------------------
echo
echo "== stage 1: precompute, every experiment concurrently =="
declare -A PIDS
CACHES=()
total=0
for exp in $EXPS; do
    total=$(( total + $(words "${CYCLES_FOR[$exp]}") ))
done
chunk=$(chunk_size "$total")
reset_load
for exp in $EXPS; do
    CACHES+=(--cache "$OUT/precompute-$exp/cache")
    n=$(words "${CYCLES_FOR[$exp]}")
    if [ "$n" -eq 0 ]; then
        echo "-- $exp: none of the $NCYC cycles resolve on disk -- skipping precompute"
        continue
    fi
    # shellcheck disable=SC2086
    mapfile -t parts < <(chunks "$chunk" ${CYCLES_FOR[$exp]})
    echo "-- $exp: $n of $NCYC cycles on disk, in ${#parts[@]} step(s)"
    k=0
    for part in "${parts[@]}"; do
        k=$(( k + 1 ))
        log=$OUT/precompute-$exp.log
        [ "${#parts[@]}" -gt 1 ] && log=$OUT/precompute-$exp.part$k.log
        cyc=()
        for c in $part; do cyc+=(--cycle "$c"); done
        w=$(words "$part")
        # shellcheck disable=SC2086  # PRECOMPUTE_OPTS is a word list on purpose
        launch "$log" "$w" python3 precompute_experiment.py "$CFG" \
            --experiment "$exp" \
            --outdir "$OUT/precompute-$exp" \
            --skip-preflight \
            --jobs "$w" "${cyc[@]}" $PRECOMPUTE_OPTS
        PIDS[$exp.part$k]=$LAUNCHED
    done
done

fail=0
for key in "${!PIDS[@]}"; do
    if wait "${PIDS[$key]}"; then
        echo "-- precompute ok: $key"
    else
        echo "!! precompute FAILED: $key -- see $OUT/precompute-${key%.part*}*.log"
        fail=1
    fi
done
if [ $fail -ne 0 ]; then
    echo "!! at least one precompute step failed -- stopping before stage 2"
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
ALL_STAGES="rejoin obsspace obsbins statespace timeseries fronts stability grep bkgerr scorecard report"
stage() {
    # stage <name> <jobs> [option...]: the command (in STAGE) that runs only
    # that sub-stage of build_comparison.py, with --jobs <jobs> and any
    # further options -- a command rather than a function call, so srun can
    # run it
    local only=$1 jobs=$2 skips=() s
    shift 2
    for s in $ALL_STAGES; do
        [ "$s" != "$only" ] && skips+=(--skip "$s")
    done
    # shellcheck disable=SC2206  # BUILD_OPTS is a word list on purpose
    STAGE=(python3 build_comparison.py "$CFG" "${CACHES[@]}"
           --outdir "$OUT/page" --jobs "$jobs"
           --hours "$HOURS" $BUILD_OPTS "${skips[@]}" "$@")
}
mkdir -p "$OUT/page"

echo
echo "== stage 2a: rejoin observations across experiments =="
# shellcheck disable=SC2086
mapfile -t parts < <(chunks "$(chunk_size "$NCYC")" $ALL_CYCLES)
declare -A RPIDS
reset_load
k=0
for part in "${parts[@]}"; do
    k=$(( k + 1 ))
    log=$OUT/page/stage-rejoin.log
    [ "${#parts[@]}" -gt 1 ] && log=$OUT/page/stage-rejoin.part$k.log
    cyc=()
    for c in $part; do cyc+=(--cycle "$c"); done
    w=$(words "$part")
    stage rejoin "$w" "${cyc[@]}"
    launch "$log" "$w" "${STAGE[@]}"
    RPIDS[part$k]=$LAUNCHED
done
for key in "${!RPIDS[@]}"; do
    wait "${RPIDS[$key]}"
    echo "-- rejoin $key exit code: $?"
done

echo
echo "== stage 2b: figures + scorecard, concurrently =="
# The four stages with a worker pool share what the allocation has left
# after the five single-process ones (two shares each: the stage, and room
# for what it loads).
POOLED="obsbins statespace fronts stability"
SINGLE="obsspace timeseries grep bkgerr scorecard"
jobs=$(( TOTCAP - 2 * $(words "$SINGLE") ))
jobs=$(( jobs / $(words "$POOLED") - 1 ))
[ "$jobs" -gt "$NCYC" ] && jobs=$NCYC
[ "$jobs" -gt "$MAXW" ] && jobs=$MAXW
[ "$jobs" -lt 1 ] && jobs=1
declare -A SPIDS
reset_load
for s in $POOLED $SINGLE; do
    w=1
    [[ " $POOLED " == *" $s "* ]] && w=$jobs
    echo "-- launching $s"
    stage "$s" "$w"
    launch "$OUT/page/stage-$s.log" "$w" "${STAGE[@]}"
    SPIDS[$s]=$LAUNCHED
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
stage report 1
"${STAGE[@]}"
echo "report exit code: $? (non-zero here just means a figure was flagged"
echo "missing -- see build_report.py's summary above; the report and tarball"
echo "are written regardless)"

echo
echo "done -- report at $OUT/page/letkf_verification.html"
echo "        tarball at $OUT/page/letkf_verification.tar (pages + linked figures)"
echo "        scorecard at $OUT/page/scorecard.md"
