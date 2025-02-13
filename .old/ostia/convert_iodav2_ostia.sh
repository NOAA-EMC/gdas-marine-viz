#!/bin/bash
# Convert OSTIA L4 SST data to IODA-v2

#set -e
#set -u
#
#usage="usage: $0 yyyymmdd input_path output_path"
#if [[ $# != 3 ]]; then
#    echo $usage
#    exit 1
#fi
#
#date=$1
#input_path=$2
#output_path=$3

export SOCA_SCIENCE_RUNTIME=T
export IODA=/work/noaa/ng-godas/spaturi/nggodas_realtime_ext/forCPC/build/bin
source $IODA/../../soca-science/configs/machine/machine.orion.intel

pyioda_path=${IODA}/../../build/lib/python3.9/pyioda
export PYTHONPATH=$PYTHONPATH:$pyioda_path

export pySRC=${IODA}

year=${date:0:4}
if [ -d $input_path/$date ]; then
   mkdir -p $output_path/${year}/${date}
   python $pySRC/ostia_l4sst2ioda.py                  \
                -i $input_path/${date}/*.nc           \
   -o $output_path/${year}/${date}/sst_ostia_${date}.nc
fi
echo " "
echo $date DONE
       
