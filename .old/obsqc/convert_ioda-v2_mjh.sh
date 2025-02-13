#!/bin/bash
# Convert EMC ICEC (SSMI & SSMIS) L2 data to IODA-v2

set -e
set -u

#usage="usage: $0 [ssmi|f28] yyyymmdd input_path output_path"
#if [[ $# != 3 ]]; then
#    echo $usage
#    exit 1
#fi

typ=ssmi #$1
input_path=/work/noaa/ng-godas/spaturi/icel2_emc_2021 #$3
output_path=$PWD/icel2_emc_2021

export SOCA_SCIENCE_RUNTIME=T
#export IODA=/work/noaa/ng-godas/spaturi/nggodas_realtime_ext/forCPC/build/bin
export IODA=/work/noaa/ng-godas/jhossen/s2s_sprint/build/bin
source $IODA/../../soca-science/configs/machine/machine.orion.intel
#unset SOCA_SCIENCE_RUNTIME

pyioda_path=${IODA}/../../build/lib/python3.9/pyioda
export PYTHONPATH=$PYTHONPATH:$pyioda_path

export pySRC=${IODA}

start_date=20210714
end_date=20211231
ymd_date=$(date -d "$start_date")
ymd=$(date -d "$ymd_date" +%Y%m%d )
echo ${ymd}
while [ $ymd -le $end_date ]; do
    year=${ymd:0:4}
    if [ $typ == 'f28' ]; then
        sat=ssmis
    else
        sat=ssmi
    fi

    if [ ! -f $input_path/l2.${typ}*${ymd}.nc ]; then
        echo $typ file not found on $ymd
    else
        mkdir -p $output_path/ioda-v2/${year}/${ymd}
        python $pySRC/emc_ice2ioda.py                                           \
          -i ${input_path}/l2.${typ}*${ymd}.nc                            \
          -o ${output_path}/ioda-v2/${year}/${ymd}/icec_${sat}_${ymd}.nc \
          -d ${ymd}'00'           
        echo $typ $ymd DONE
    fi
    ymd=$(date -d "$ymd + 1 day" +%Y%m%d )
done 
# SSM/I: Special Sensor Microwave - Imager
# f285_51 & f286_52-> SSMI/S: Special Sensor Microwave - Imager/Sounder
           
