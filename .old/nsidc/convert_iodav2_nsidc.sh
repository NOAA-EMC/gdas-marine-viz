#!/bin/bash
# Convert NSIDc CDR data (NH & SH) to ioda-v1 format and then to ioda-v2
# TODO: update nsidc_ice2ioda.py to convert to ioda-v2
#

module load nco

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

export SOCA_SCIENCE_RUNTIME=T 
source /work/noaa/ng-godas/spaturi/nggodas_realtime_ext/forCPC/soca-science/configs/machine/machine.orion.intel
export pySRC=/work/noaa/ng-godas/spaturi/nggodas_realtime_ext/forCPC/build/bin

year=${date:0:4}

mkdir -p $output_path/${year}/${date}
# NSIDC sea-ice concentration files have latitude, longitude given in a separate file
# Add latitude, longitude to the sea-ice concentration file

for reg in nh sh; do
   # append latitude, longitude to the input NSIDC nh and sh files
   ncks -A -C -v latitude,longitude $input_path/G02202-cdr-ancillary-${reg}.nc \
                 $input_path/${year}/${date}/seaice_conc_daily_${reg}_${date}_f17_v04r00.nc
   python $pySRC/nsidc_ice2ioda.py -i $input_path/${year}/${date}/seaice*${reg}*nc   \
          -o $output_path/${year}/${date}/icec_nsidc_${reg}_${date}.nc \
          -d ${date}'12'
   $pySRC/ioda-upgrade.x $output_path/${year}/${date}/icec_nsidc_${reg}_${date}.nc \
                         $output_path/${year}/${date}/icec_nsidc_${reg}_${date}_v2.nc
   mv $output_path/${year}/${date}/icec_nsidc_${reg}_${date}_v2.nc \
      $output_path/${year}/${date}/icec_nsidc_${reg}_${date}.nc
done
