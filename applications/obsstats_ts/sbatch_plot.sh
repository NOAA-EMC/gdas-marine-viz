#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=plot.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/gdas-marine-viz/bin/activate

# iterate one day at a time (override START_DATE and END_DATE if desired)
START_DATE=${START_DATE:-20250921}
END_DATE=${END_DATE:-20250922}

current="$START_DATE"
while [ "$current" -le "$END_DATE" ]; do
    echo "Processing date: $current"
    ./generate_config.py -o full_config.yaml --date-wildcard "$current"
    ./plot_timeseries.py full_config.yaml --skip-plots
    current=$(date -d "$current +1 day" +%Y%m%d)
done
./plot_timeseries.py full_config.yaml
./generate_timeseries_index.py ./stats-comparisons --config full_config.yaml
