#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=plot.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/gdas-marine-viz/bin/activate

# set END_DATE to today (can be overridden by env var END_DATE)
END_DATE=${END_DATE:-$(date +%Y%m%d)}

# number of days before END_DATE for START_DATE (can be overridden by env var N_DAYS)
N_DAYS=${N_DAYS:-2}

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
