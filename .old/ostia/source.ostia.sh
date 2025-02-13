#!/bin/bash
# download OSTIA v2 GDS2 L4 UKMO data

set -e
set -u

usage="usage: $0 yyyymmdd output_path"
if [[ $# != 2 ]]; then
    echo $usage
    exit 1
fi

date=$1
output_path=$2

if [ $date -lt 20070101 ]; then
   echo "OSTIA v2 files are available from 20070101; please correct and resubmit"
   exit 1
else
   yr=${date:0:4}
   dy=$(date -d "$date" "+%j")
   datep1=$(date +%Y%m%d -d "$date + 1 day")
   datem1=$(date +%Y%m%d -d "$date - 1 day")
   dym1=$(date -d "$datem1" "+%j")  
   echo $yr $dym1 $dy $datem1 $date $datep1
fi

file_sfx=nc

source_base="https://podaac-tools.jpl.nasa.gov/drive/files/allData/ghrsst/data/GDS2/L4/GLOB/UKMO/OSTIA/v2" 

out_dir="$output_path/OSTIA"
pwd=$(pwd)

d=$out_dir/$date
mkdir -p $d
cd $d

if [[ $date == 20070101 ]]; then
   source_dir="$source_base/2006/${dym1}"
elif [[ $date -ge 20070102 && $date -le 20120214 ]]; then 
   source_dir="$source_base/${yr}/${dym1}"
else
   source_dir="$source_base/${yr}/${dy}"
fi

f=$date"120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc"

echo $source_dir/$f
wget --user=paturishastri --password=FKt3gk6mslILaFUKkTYO $source_dir/$f 

cd $pwd



