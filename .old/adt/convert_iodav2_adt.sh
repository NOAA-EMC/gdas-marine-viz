#!/bin/bash
# Convert ADT from NESDIS to IODA-v2

set -e
set -u

usage="usage: $0 yyyymmdd input_path output_path"
if [[ $# != 3 ]]; then
    echo $usage
    exit 1
fi

date=$1
dy=$(date -d "$date" "+%j")
input_path=$2
output_path=$3

export SOCA_SCIENCE_RUNTIME=T
export IODA=/work/noaa/ng-godas/spaturi/nggodas_realtime_ext/forCPC/build/bin
source $IODA/../../soca-science/configs/machine/machine.orion.intel

pyioda_path=${IODA}/../../build/lib/python3.9/pyioda
export PYTHONPATH=$PYTHONPATH:$pyioda_path

export pySRC=${IODA}

year=${date:0:4}
if [ -d $input_path/$date ]; then
   cd $input_path/${date}
   export lof=`ls rads*.nc`
   mkdir -p $output_path/${year}/${date}
   for obs in $lof; do
       echo $obs
       sat=${obs:9:2}
       python $pySRC/rads_adt2ioda.py                                 \
                -i $input_path/${date}/rads_adt_${sat}_*.nc           \
                -o $output_path/${year}/${date}/adt_${sat}_${date}.nc \
                -d ${date}'12'
       cd $output_path/${year}/${date}
       ncks adt_${sat}_${date}.nc out.nc
       ncrename -O -v datetime,dateTime out.nc out.nc
       mv out.nc adt_${sat}_${date}.nc
   done
fi

