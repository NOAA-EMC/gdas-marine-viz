#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=plot.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/gdas-marine-viz/bin/activate

# Link experiments
for src in \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/cp4.04-parallel-hybrid" \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/cp4.04-parallel-3dvar" \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/retrov17_01_realtime"
do
    name=$(basename "$src")
    if [ -e "$name" ] || [ -L "$name" ]; then
        echo "Skipping $name: already exists"
    else
        ln -s "$src" .
    fi
done

# set END_DATE to today (can be overridden by env var END_DATE)
END_DATE=${END_DATE:-$(date +%Y%m%d)}

# number of days before END_DATE for START_DATE (can be overridden by env var N_DAYS)
N_DAYS=${N_DAYS:-100}

# compute START_DATE N_DAYS before END_DATE (can be overridden by env var START_DATE)
START_DATE=${START_DATE:-$(date -d "$END_DATE -${N_DAYS} days" +%Y%m%d)}

current="$START_DATE"
while [ "$current" -le "$END_DATE" ]; do
    echo "Processing date: $current"
    ./generate_config.py -o full_config.yaml --date-wildcard "$current"
    ./plot_timeseries.py full_config.yaml --skip-plots
    current=$(date -d "$current +1 day" +%Y%m%d)
done
./plot_timeseries.py full_config.yaml
./generate_timeseries_index.py ./stats-comparisons --config full_config.yaml
tar cvf stats-comparisons.tar ./stats-comparisons
