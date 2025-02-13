#!/bin/bash
set -eu 

usage="usage: $0 yyyymmdd input_path output_path"
if [[ $# != 3 ]]; then
    echo $usage
    exit 1
fi

date=$1
year=${date:0:4}
input_path=$2
output_path=$3

export MACHINE_ID=gaea

if [[ $MACHINE_ID == gaea ]]; then
   source /lustre/f2/scratch/Shastri.Paturi/sandbox/20220613/soca-science/configs/machine/machine.${MACHINE_ID}.intel
   export pySRC=/lustre/f2/scratch/Shastri.Paturi/sandbox/20220623/build
   export PYTHONPATH=${pySRC}/lib/python3.7/pyioda:$PYTHONPATH
fi

mkdir -p ${output_path}/ioda-v2/${year}/${date}
echo $input_path/${year}/${date}
if [ -d $input_path/${year}/${date} ]; then
   for typ in profile ship trak; do
       if [ -f $input_path/${year}/${date}/${date}'00'.${typ} ]; then
             echo $input_path/${year}/${date}/${date}'00'.${typ}
             python ${pySRC}/bin/godae_${typ}2ioda.py               \
                    -i ${input_path}/${year}/${date}/${date}'00'.${typ}              \
                    -o ${output_path}/ioda-v2/${year}/${date}/insitu_${typ}_fnmoc_${date}.nc \
                    -d ${date}'12'
          echo " "
          echo $typ DONE
          fi
      done 
   fi
   echo " "
echo $date DONE
