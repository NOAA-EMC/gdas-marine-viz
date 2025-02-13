#!/bin/bash
#SBATCH -e err
#SBATCH -o out
#SBATCH --account=gldas
#SBATCH --qos=urgent
#SBATCH --partition=batch
#SBATCH --clusters=c4
#SBATCH --ntasks=2
#SBATCH --time=10
#SBATCH --job-name="adt_ioda"

export DIR=/lustre/f2/scratch/Shastri.Paturi
export SRC=${DIR}/sandbox/20220623/build/bin
export MACHINE=gaea.intel

source $DIR/sandbox/20220613/soca-science/configs/machine/machine.gaea.intel
export PYTHONPATH=${SRC}/../lib/python3.7/pyioda:$PYTHONPATH

export DATE=20190831
year=${DATE:0:4}
dy=$(date -d "$date" "+%j")

echo " "
mkdir -p ${DIR}/DATA_realtime/adt.nesdis/ioda-v2/${year}/${DATE}
cd ${DIR}/DATA_realtime/adt.nesdis/${year}/${DATE}
lof=`ls rads_adt_*.nc`
for obs in $lof; do
   sat=${file:9:2}
   srun -n 1 $SRC/rads_adt2ioda.py \
	  -i ${DIR}/DATA_realtime/adt.nesdis/${year}/${DATE}/${obs} \
	  -o ${DIR}/DATA_realtime/adt.nesdis/ioda-v2/${year}/${DATE}/adt_${sat}_${DATE}.nc \
	  -d ${DATE}'12'
   echo "ADT $sat DONE"
done
