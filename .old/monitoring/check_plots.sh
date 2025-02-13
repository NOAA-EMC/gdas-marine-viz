#!/bin/bash

while getopts v:s:e:p:l: flag
do
    case "${flag}" in
        v) stv=${OPTARG};;  #state vector
        p) cycle_dir=${OPTARG};;  #path/to/exp
        s) start_date=${OPTARG}Z00;;
        e) end_date=${OPTARG}00;;
        l) fctl=${OPTARG};;     # forecast length, should be consistent with model output
        *) 
            echo please provide correct arguments
            exit 1 ;;
    esac
done
date_YMDH=$(date -ud "$start_date")
YMDH=$(date -ud "$date_YMDH " +%Y%m%d%H )
while [ "$YMDH" -le "$end_date" ]; do
    year=${YMDH:0:4}
    YMD=${YMDH:0:8}
    ddir=${cycle_dir}/monitoring/$year/$YMD/$stv
    files=$(ls ${ddir}/*png 2> /dev/null | wc -l )
    #if  [ ! -f ${ddir}/Temp.global.${YMD}00Z.png ] ; then
    if [ $files == 0 ]; then
            echo missing files on $YMD
    fi
    DH=$(($DH+$fctl))
    YMDH=$(date -ud "$date_YMDH + $DH hours" +%Y%m%d%H )

done  # ymd
           
