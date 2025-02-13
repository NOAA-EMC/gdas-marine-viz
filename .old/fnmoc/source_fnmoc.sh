#!/bin/bash
# Downloads insitu ocean observations from FNMOC (GODAE)

set -eu

usage="usage: $0 [prof|sfc|trak] yyyymmdd output_path"
if [[ $# != 3 ]]; then
    echo $usage
    exit 1
fi

type=$1
date=$2
year=${date:0:4}
out_dir="$3"

source="https://www.usgodae.org/pub/outgoing/fnmoc/data/ocn"

if [[ $type == "prof" ]]; then
    fn1=profile
    fn2=profile
elif [[ $type == "sfc" ]]; then
    fn1=sfcobs
    fn2=ship
elif [[ $type == "trak" ]]; then
    fn1=trak
    fn2=trak
else
    echo $usage
    exit 1
fi

d=$out_dir
pwd=$(pwd)
mkdir -p $d/${year}/${date}
cd $d/${year}/${date}
wget $source/$fn1/$year/${date}00.$fn2.Z --no-check-certificate
gunzip *.Z
cd $pwd
