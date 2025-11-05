#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=plot.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/gdas-marine-viz/bin/activate

# Link experiments
# Link experiments (skip if the links already exist)
for src in \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/cp4.02d-parallel-obsforge" \
    "/scratch3/NCEPDEV/global/Katherine.Lukens/expts/hpss/retrotestgfs16_17_realtime" \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/cp4.03-parallel-hybrid"
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
