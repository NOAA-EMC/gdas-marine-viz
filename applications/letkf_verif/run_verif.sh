#!/bin/bash
#SBATCH --job-name=verif
#SBATCH --account=da-cpu
#SBATCH --qos=batch
##SBATCH --partition=hera
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=05:59:00
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
# work runs on this machine, with concurrent steps held to its core and memory
# capacity.
#
# Parallelism:
#   stage 1  precompute: every experiment's cycles not yet in its cache are
#            split into chunks, at most one node's worth of workers each and
#            small enough that there are at least as many chunks as nodes;
#            every chunk is a step (precompute_experiment.py --cycle ...
#            --jobs <chunk size>), all at once. Cached cycles are dropped
#            before the split, so they neither take a node's share nor leave
#            the new cycles bunched on a few nodes. Cycles are independent
#            and cached one file each, so chunks of one experiment do not
#            collide.
#   stage 2  rejoin first (it rewrites the page cache the rest read), its
#            out-of-date cycles split across the nodes the same way; then the figure
#            stages -- obs-space, state-space, binned departures, time
#            series, fronts, stability, GREP, background error -- and the
#            scorecard, one step each, CONCURRENTLY (they only share the
#            read-only cache and write different files), then the report
#            once all of them are in. Binned departures, state-space,
#            fronts, stability and GREP draw (GREP: reduce) their cycles
#            --jobs-way in parallel, with --jobs set so the five of them fit
#            the allocation together.
#   caching  every stage skips work whose inputs have not changed: the
#            precompute for cycles already cached and the rejoin when the
#            page-cache file is newer than its sources (both checked here,
#            before anything is launched), and
#            each plot stage per cycle (figs/.fresh-<stage>.json records
#            what was drawn from which cache files). A rerun that added one
#            cycle redraws that cycle and the across-date sequences only.
#            BUILD_OPTS="--force" redoes everything after the precompute,
#            PRECOMPUTE_OPTS="--force" the precompute.
#
# Sizing: a measured worker peaks around 3 GB RSS (sacct on a real run);
# WORKER_GB=5 (the default) leaves margin. A step killed for memory says
# "oom-kill" in its log: raise WORKER_GB (fewer, roomier workers per node)
# or add nodes. Precompute wall time is one wave bounded by the slowest
# cycle as long as nodes x (node memory / WORKER_GB) covers every cycle still
# to compute (cached ones do not count); past that the chunks queue for room
# and it takes more waves.
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

LOCAL_ACTIVE=() LOCAL_USED=0
declare -A LOCAL_SHARES LOCAL_STATUS

# Reap completed local steps and return their shares to the pool.
reap_local() {
    local running pid rc
    local still=()
    running=$(jobs -pr)
    running=${running//$'\n'/ }
    for pid in "${LOCAL_ACTIVE[@]}"; do
        case " $running " in
            *" $pid "*) still+=("$pid") ;;
            *)
                if wait "$pid"; then rc=0; else rc=$?; fi
                LOCAL_STATUS[$pid]=$rc
                LOCAL_USED=$(( LOCAL_USED - LOCAL_SHARES[$pid] ))
                unset 'LOCAL_SHARES[$pid]'
                ;;
        esac
    done
    LOCAL_ACTIVE=("${still[@]}")
}

# Wait for a step without losing a status already collected by reap_local.
wait_child() {
    local pid=$1 rc
    if [ "$USE_SRUN" -eq 0 ] && [ "${LOCAL_STATUS[$pid]+present}" ]; then
        rc=${LOCAL_STATUS[$pid]}
        unset 'LOCAL_STATUS[$pid]'
    else
        if wait "$pid"; then rc=0; else rc=$?; fi
        if [ "$USE_SRUN" -eq 0 ] && [ "${LOCAL_SHARES[$pid]+present}" ]; then
            LOCAL_USED=$(( LOCAL_USED - LOCAL_SHARES[$pid] ))
            unset 'LOCAL_SHARES[$pid]'
            local still=() active
            for active in "${LOCAL_ACTIVE[@]}"; do
                [ "$active" = "$pid" ] || still+=("$active")
            done
            LOCAL_ACTIVE=("${still[@]}")
        fi
    fi
    return "$rc"
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
    if [ "$USE_SRUN" -eq 0 ]; then
        while :; do
            reap_local
            [ $(( LOCAL_USED + u )) -le "${CAP[0]}" ] && break
            sleep 0.1
        done
    fi
    if [ "$USE_SRUN" -eq 1 ]; then
        env -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_GPU \
            srun --nodes=1 --ntasks=1 --exact --cpus-per-task="$u" \
                 --mem=$(( u * WORKER_GB * 1024 ))M \
                 --nodelist="${NODES[PICKED]}" "$@" > "$log" 2>&1 &
    else
        "$@" > "$log" 2>&1 &
    fi
    LAUNCHED=$!
    if [ "$USE_SRUN" -eq 0 ]; then
        LOCAL_ACTIVE+=("$LAUNCHED")
        LOCAL_SHARES[$LAUNCHED]=$u
        LOCAL_USED=$(( LOCAL_USED + u ))
    fi
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

# has_force <options>: the word list holds --force
has_force() {
    local opt
    # shellcheck disable=SC2086  # options are a word list, as at launch
    for opt in $1; do
        if [ "$opt" = "--force" ]; then echo 1; return; fi
    done
    echo 0
}

# ---------------------------------------------------------------------------
# Read the yaml: experiment names, the cycles, and which of them each
# experiment still has to precompute (TODO_FOR: not in its cache yet, or
# every cycle under PRECOMPUTE_OPTS=--force). Cached cycles are dropped here,
# before the work is split over the nodes, so the split spreads only real
# work. Uses the same loader and cache test the python stages use, so
# relative paths and stems resolve identically.
# ---------------------------------------------------------------------------
eval "$(python3 - "$CFG" "$OUT" "$(has_force "$PRECOMPUTE_OPTS")" <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from lv_common import load_config
from compute_cycle import is_cached
path, out, force = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
cfg = load_config(path, None, None, None)
cycles = [str(c) for c in cfg['cycles']]
print('NCYC=%d' % len(cycles))
print('ALL_CYCLES="%s"' % ' '.join(cycles))
print('EXPS="%s"' % ' '.join(e.name for e in cfg['experiments']))
print('declare -A CYCLES_FOR TODO_FOR')
for e in cfg['experiments']:
    # the cache precompute_experiment.py --outdir writes this experiment to
    ecfg = load_config(path, None, os.path.join(out, 'precompute-' + e.name),
                       None)
    todo = [c for c in cycles if force or not is_cached(ecfg, c)]
    print('CYCLES_FOR[%s]="%s"' % (e.name, ' '.join(cycles)))
    print('TODO_FOR[%s]="%s"' % (e.name, ' '.join(todo)))
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
    total=$(( total + $(words "${TODO_FOR[$exp]}") ))
done
[ "$total" -eq 0 ] && echo "-- every experiment has all $NCYC cycles cached -- nothing to precompute"
chunk=$(chunk_size "$total")
reset_load
for exp in $EXPS; do
    CACHES+=(--cache "$OUT/precompute-$exp/cache")
    n=$(words "${TODO_FOR[$exp]}")
    if [ "$n" -eq 0 ]; then
        echo "-- $exp: all $NCYC cycles cached -- skipping precompute"
        continue
    fi
    # shellcheck disable=SC2086
    mapfile -t parts < <(chunks "$chunk" ${TODO_FOR[$exp]})
    echo "-- $exp: $n of $NCYC cycles to compute ($(( $(words "${CYCLES_FOR[$exp]}") - n )) cached), in ${#parts[@]} step(s)"
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
    if wait_child "${PIDS[$key]}"; then
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
# Only the cycles whose join is out of date (every cycle under
# BUILD_OPTS=--force), found now that stage 1 has rewritten its caches, and
# with the cache list build_comparison.py would give the rejoin: the page's
# own cache first, then the per-experiment ones.
REJOIN_CYCLES=$(python3 - "$CFG" "$OUT/page" "$(has_force "$BUILD_OPTS")" "${CACHES[@]}" <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from lv_common import load_config
from compute_cycle import rejoin_current
path, page, force = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
caches = [os.path.join(page, 'cache')] + sys.argv[5::2]   # skip the --cache words
cfg = load_config(path, None, page, caches)
print(' '.join(str(c) for c in cfg['cycles']
               if force or not rejoin_current(cfg, str(c))))
PY
) || { echo "!! could not check the rejoin cache; rejoining every cycle"; REJOIN_CYCLES=$ALL_CYCLES; }
nrej=$(words "$REJOIN_CYCLES")
if [ "$nrej" -eq 0 ]; then
    echo "-- rejoin: all $NCYC cycles current -- skipping"
    parts=()
else
    echo "-- rejoin: $nrej of $NCYC cycles out of date"
    # shellcheck disable=SC2086
    mapfile -t parts < <(chunks "$(chunk_size "$nrej")" $REJOIN_CYCLES)
fi
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
    if wait_child "${RPIDS[$key]}"; then rc=0; else rc=$?; fi
    echo "-- rejoin $key exit code: $rc"
done

echo
echo "== stage 2b: figures + scorecard, concurrently =="
# The five stages with a worker pool share what the allocation has left
# after the four single-process ones (two shares each: the stage, and room
# for what it loads).
POOLED="obsbins statespace fronts stability grep"
SINGLE="obsspace timeseries bkgerr scorecard"
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
    if wait_child "${SPIDS[$s]}"; then
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
