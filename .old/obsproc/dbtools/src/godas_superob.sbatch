#!/bin/bash
#SBATCH --job-name=superob
#SBATCH --nodes=1
##SBATCH --qos=debug
#SBATCH --qos=batch
#SBATCH --time=00:30:00
#SBATCH --account=marine-cpu
#SBATCH --output=superob%j.log
#SBATCH --error=superob%j.err
#SBATCH --exclusive
#SBATCH --partition=orion

function ymd2iso () {
    echo $( date -d "$1" +%Y-%m-%dT:%H:%M:%SZ )
}


# User defined stuff
startdate='2015-01-01'
enddate='2015-01-03'
step='PT10M'
r2d2_name='gdas_marine'
r2d2_version='s2s_v1'
obstype='sst_noaa18_l3u'
export R2D2_DB=/work/noaa/ng-godas/r2d2/                                                               # path to the DB
source /work/noaa/marine/Guillaume.Vernieres/runs/s2s/soca-science/configs/machine/machine.orion.intel # Load modules
superob_bin=/work/noaa/marine/Guillaume.Vernieres/runs/s2s/broken-b/build/bin/obs_superob.x            # superob exec
export GRIDSPEC=/work/noaa/marine/Guillaume.Vernieres/runs/s2s/notfinal-0.25/static/soca_gridspec.nc      # 1/4 deg grid
superob_out=$PWD/test                                                                                  # Output path for the superobed files
godas_sandbox=/work/noaa/marine/Guillaume.Vernieres/OBS/sprint1of4/convert2ioda/godas                  # path to godas
superob=${godas_sandbox}/obsproc/dbtools/src/godas_superob.py                                             # Python wrapper to superob

# Superob each files
export OMP_NUM_THREADS=1
ulimit -s unlimited
mkdir -p tmp
cd $PWD/tmp

enddate=$( date -d "$enddate" +%Y%m%d )  # rewrite in YYYYMMDD format
cycle_start=$startdate
while [ "$cycle_end" != "$enddate" ]; do
    cycle_end=$( date -d "$cycle_start + 1 days" +%Y%m%d )

    # Fetch obs from R2D2
    echo "=================================================================="
    echo "Fetching $obstype from $(ymd2iso $cycle_start) to $(ymd2iso $cycle_end) ..."
    ${superob} --start $(ymd2iso $cycle_start) \
               --end $(ymd2iso $cycle_end) \
               --step $step \
               --provider $r2d2_name \
               --experiment $r2d2_version \
               --obstype $obstype \
               --superobout ${superob_out} > r2d2_fetch_$(ymd2iso $cycle_start).log

    # Superob the batch of fetched obs
    echo "Superobing from $(ymd2iso $cycle_start) to $(ymd2iso $cycle_end) ..."

    profiling_start=`date +%s.%N`
    lof=`ls ${obstype}_$( date -d "$cycle_start" +%Y%m%d )*.nc.yaml`
    for f in $lof; do
        echo "Superobing $f"
        $superob_bin $f > ${f}.out
    done
    wait # for all processes to be done
    profiling_end=`date +%s.%N`
    echo "RUNTIME: $( echo "$profiling_end - $profiling_start" | bc -l )"

    # Increment time by one day for next batch of obs
    cycle_start=${cycle_end}

done

exit 0
