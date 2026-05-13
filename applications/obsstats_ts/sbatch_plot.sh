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
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/cp4.02d-parallel-obsforge" \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/rt17_upd03_realtime" \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/cp4.03-parallel-hybrid" \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/cp4.03-parallel-3dvar" \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/cp4.04-parallel-hybrid" \
    "/scratch4/NCEPDEV/global/John.Steffen/hpss_arch/cp4.04-parallel-3dvar"
do
    name=$(basename "$src")
    if [ -e "$name" ] || [ -L "$name" ]; then
        echo "Skipping $name: already exists"
    else
        ln -s "$src" .
    fi
done

# iterate one day at a time (override START_DATE and END_DATE if desired)
START_DATE=20251007
END_DATE=20251117

current="$START_DATE"
while [ "$current" -le "$END_DATE" ]; do
    echo "========================================="
    echo "Processing date: $current"
    echo "========================================="
    ./generate_config.py -o full_config.yaml --date-wildcard "$current" > generate_config_"${current}".log 2>&1
    ./plot_timeseries.py full_config.yaml --skip-plots > plot_"${current}".log 2>&1
    current=$(date -d "$current +1 day" +%Y%m%d)
done
./plot_timeseries.py full_config.yaml
./generate_timeseries_index.py ./stats-comparisons --config full_config.yaml
