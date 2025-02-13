#!/bin/bash
# superob OSTIA L4 SST data horizontally

module load nco

ulimit -s unlimited

set -e
set -u

usage="usage: $0 yyyymmdd input_path output_path"
if [[ $# != 3 ]]; then
    echo $usage
    exit 1
fi

date=$1
input_path=$2
output_path=$3

echo "Job started:  " `date`

export SRC=/work/noaa/ng-godas/spaturi/sandbox/20220317_superob_horiz/build/bin

export OMP_NUM_THREADS=40
year=${date:0:4}

mkdir -p ${output_path}/${year}/${date} #${ostia_superob}/${year}/${dtg}

export input_file=${input_path}/${year}/${date}/sst_ostia_${date}.nc
export output_file=${output_path}/${year}/${date}/sst_ostia_${date}.nc

# Avoid job errors because of filesystem synchronization delays
sync && sleep 1

cp -p config_ORG.yaml config.yaml
sed -i "s;__INPUT_FILE__;${input_file};g" config.yaml
sed -i "s;__OUTPUT_FILE__;${output_file};g" config.yaml

$SRC/obs_superob.x config.yaml

## rename datetime to dateTime
cd ${output_path}/${year}/${date}
ncks sst_ostia_${date}.nc out.nc
ncrename -O -v datetime,dateTime out.nc out.nc
mv out.nc sst_ostia_${date}.nc

echo "Job ended:    " `date







