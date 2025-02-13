#!/bin/bash
# download NSIDC CDR data

set -e
set -u

usage="usage: $0 yyyymmdd output_path"
if [[ $# != 2 ]]; then
    echo $usage
    exit 1
fi

date=$1
output_path=$2

source=ftp://sidads.colorado.edu/pub/DATASETS/NOAA/G02202_V4/
year=${date:0:4}
mkdir -p $output_path/${year}/${date} && cd $output_path/${year}/${date}
wget --ftp-user=anonymous ${source}/north/daily/${year}/seaice_conc_daily_nh_${date}_f17_v04r00.nc
wget --ftp-user=anonymous ${source}/south/daily/${year}/seaice_conc_daily_sh_${date}_f17_v04r00.nc
    
# get the latitude, longitude data files

if [ ! -f $output_path/G02202-cdr-ancillary-nh.nc ]; then 
   wget --ftp-user=anonymous ${source}/ancillary/G02202-cdr-ancillary-nh.nc
elif [ ! -f $output_path/G02202-cdr-ancillary-sh.nc ]; then
   wget --ftp-user=anonymous ${source}/ancillary/G02202-cdr-ancillary-sh.nc
fi

