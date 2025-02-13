#!/bin/bash
# create individual obsspec files from previously converted FNMOC insitu files
# rename variable: datetime to dateTime

module load nco

set -eu 

usage="usage: $0 yyyymmdd output_path"
if [[ $# != 2 ]]; then
    echo $usage
    exit 1
fi

date=$1
year=${date:0:4}
output_path=$2

cd $output_path/${year}/${date}
export lof=`ls *.nc`
for obs in $lof; do
    if [ "${obs:7:4}" = "ship" ]; then
        echo $obs
        ncks $obs out.nc
        ncrename -O -v datetime,dateTime out.nc out.nc
        mv out.nc sst_ship_fnmoc_${date}.nc

    elif [ "${obs:7:4}" = "trak" ]; then
        echo $obs
        ncks -v sea_surface_salinity -x $obs out.nc
        ncrename -O -v datetime,dateTime out.nc out.nc
        mv out.nc sst_trak_fnmoc_${date}.nc

        ncks -v sea_surface_temperature -x $obs out.nc
        ncrename -O -v datetime,dateTime out.nc out.nc
        mv out.nc sss_trak_fnmoc_${date}.nc
        rm -f out.nc
    elif [ "${obs:7:4}" = "prof" ]; then
        echo $obs
        ncks -v sea_water_salinity -x $obs out.nc
        ncrename -O -v datetime,dateTime out.nc out.nc
        mv out.nc temp_profile_fnmoc_${date}.nc

        ncks -v sea_water_temperature -x $obs out.nc
        ncrename -O -v datetime,dateTime out.nc out.nc
        mv out.nc salt_profile_fnmoc_${date}.nc
        rm -f out.nc
    fi
done
