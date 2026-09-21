#!/bin/bash
# Daily driver for the verification page, meant for cron:
#
#   30 2 * * * /path/to/letkf_verif/run_daily.sh /path/to/experiments.yaml >> /path/to/daily/cron.log 2>&1
#
#   run_daily.sh [experiments.yaml]
#
# The config defaults to the experiments.yaml in the directory this is run
# from: a verification directory holds experiments.yaml, lv_grid.nc and two
# symlinks into the checkout (run_verif.sh, run_daily.sh), nothing else.
#
# 1. newest_cycle.py: find the newest complete cycle on disk for the config's
#    experiments and advance `cycles: stop:` to it (cumulative window; `start`
#    is yours). Nothing newer and a page already built -> exit 0, no job.
# 2. sbatch --wait the application's run_verif.sh, submitted from the config
#    directory with CFG pointing at the config; the caches make this cost the
#    new cycles only. SBATCH_OPTS adds per-comparison Slurm sizing
#    (e.g. SBATCH_OPTS="--time=02:00:00").
# 3. On success stamp daily/LAST_SUCCESS and run $RSYNC_HOOK if set (a
#    command that ships <config dir>/page/ to the server -- to be filled in);
#    on failure stamp daily/LAST_FAILURE with the job id and the log tail, and
#    exit non-zero so cron mails it when MAILTO is set.
#
# A lock under <config dir>/daily/ stops two runs overlapping (a long job
# still running when the next cron fires simply makes that one exit).
# Everything is written beside the config: daily/YYYYMMDD.log (30 days kept),
# daily/slurm-<jobid>.log, the stamps, and the page under page/.
#
# Environment overrides: PRECOMPUTE_OPTS, BUILD_OPTS, HOURS (passed through to
# the run script), SBATCH_OPTS, RSYNC_HOOK, SBATCH_CMD (default "sbatch";
# "echo sbatch" for a dry run), VENV (python environment to activate).
set -uo pipefail

# the application is where this script REALLY lives (it may be a symlink)
APP=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
CFG=${1:-experiments.yaml}
[ -f "$CFG" ] || { echo "usage: run_daily.sh [experiments.yaml] -- no $CFG here" >&2; exit 2; }
CFG=$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")
OUT=$(dirname "$CFG")
RUN=$APP/run_verif.sh
SBATCH_OPTS=${SBATCH_OPTS:-}
DAILY=$OUT/daily
VENV=${VENV:-/scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/gdas-marine-viz}
SBATCH_CMD=${SBATCH_CMD:-sbatch}
RSYNC_HOOK=${RSYNC_HOOK:-}

mkdir -p "$DAILY"
LOG=$DAILY/$(date -u +%Y%m%d).log
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

# one run at a time
exec 9>"$DAILY/lock"
if ! flock -n 9; then
    log "another run holds $DAILY/lock -- exiting"
    exit 0
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$APP" || exit 1
find "$DAILY" -name '????????.log' -mtime +30 -delete 2>/dev/null

log "== daily verification: $CFG"
# -- 1. the cycle window ----------------------------------------------------
out=$(python3 newest_cycle.py "$CFG" --set-stop 2>&1)
rc=$?
echo "$out" | sed 's/^/    /' | tee -a "$LOG" >/dev/null
newest=$(echo "$out" | tail -1)
stop=$(sed -n 's/^ *stop: *\([0-9]\{10\}\).*/\1/p' "$CFG" | head -1)
built=$(sed -n 's/.*stop=\([0-9]\{10\}\).*/\1/p' "$DAILY/LAST_SUCCESS" 2>/dev/null)
if [ $rc -eq 3 ]; then
    newest=$stop
    # "nothing newer" only means "nothing to do" if the page was built for
    # this very stop (LAST_SUCCESS says which); a stop advanced by hand or
    # by a dry run, or a failed build, still needs the job
    if [ -f "$OUT/page/letkf_verification.html" ] && [ "${built:-0}" -ge "${stop:-0}" ]; then
        log "no cycle newer than the config's stop $stop -- page is up to date"
        exit 0
    fi
    log "no new cycle, but the page was last built for stop ${built:-none} -- building for $stop"
elif [ $rc -ne 0 ]; then
    log "!! newest_cycle.py failed (exit $rc)"
    exit $rc
else
    log "stop advanced to $newest"
fi

# -- 2. the job -------------------------------------------------------------
submitted=$(date -u +%s)
export APP CFG PRECOMPUTE_OPTS="${PRECOMPUTE_OPTS:-}" BUILD_OPTS="${BUILD_OPTS:-}" HOURS="${HOURS:-00}"
log "submitting $RUN from $OUT (PRECOMPUTE_OPTS='$PRECOMPUTE_OPTS' BUILD_OPTS='$BUILD_OPTS' HOURS=$HOURS${SBATCH_OPTS:+ SBATCH_OPTS='$SBATCH_OPTS'})"
# shellcheck disable=SC2086  # SBATCH_OPTS is a word list on purpose
jobid=$($SBATCH_CMD --wait --parsable --chdir="$OUT" --output="$DAILY/slurm-%j.log" \
        --job-name="verif_$(basename "$OUT")" $SBATCH_OPTS "$RUN")
rc=$?
jobid=${jobid%%;*}
state=$(sacct -j "$jobid" -X -n -o State,Elapsed 2>/dev/null | head -1 | tr -s ' ')
log "job $jobid finished: exit $rc, sacct: ${state:-n/a}"

# -- 3. post-steps ------------------------------------------------------------
page=$OUT/page/letkf_verification.html
tar=$OUT/page/letkf_verification.tar
fresh=0
if [ -f "$tar" ] && [ "$(stat -c %Y "$tar")" -ge "$submitted" ]; then
    fresh=1
fi
if [ $rc -eq 0 ] && [ $fresh -eq 1 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) stop=$newest job=$jobid" > "$DAILY/LAST_SUCCESS"
    log "page and tarball refreshed ($(du -h "$tar" | cut -f1))"
    if [ -n "$RSYNC_HOOK" ]; then
        log "running RSYNC_HOOK: $RSYNC_HOOK"
        if $RSYNC_HOOK; then
            log "rsync hook ok"
        else
            log "!! rsync hook failed (exit $?)"
            exit 4
        fi
    fi
    exit 0
fi
{
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) stop=$newest job=$jobid exit=$rc tarball_refreshed=$fresh"
    echo "--- tail of $DAILY/slurm-$jobid.log ---"
    tail -40 "$DAILY/slurm-$jobid.log" 2>/dev/null
} > "$DAILY/LAST_FAILURE"
log "!! run failed (exit $rc, tarball refreshed: $fresh) -- see $DAILY/LAST_FAILURE"
exit 1
