#!/bin/bash
#SBATCH -e err
#SBATCH -o out
#SBATCH --account=gldas
#SBATCH --qos=urgent
#SBATCH --partition=batch
#SBATCH --clusters=c4
#SBATCH --ntasks=2
#SBATCH --time=20
#SBATCH --job-name="osta_ioda"

export DIR=/lustre/f2/scratch/Shastri.Paturi
export SRC=${DIR}/sandbox/20220623/build/bin

source $DIR/sandbox/20220613/soca-science/configs/machine/machine.gaea.intel
export PYTHONPATH=${SRC}/../lib/python3.7/pyioda:$PYTHONPATH

export DATE=20190831
year=${DATE:0:4}

echo " "
mkdir -p ${DIR}/DATA_realtime/OSTIA/ioda-v2/${year}/${DATE}
srun -n 1 $SRC/ostia_l4sst2ioda.py                                                   \
          -i ${DIR}/DATA_realtime/OSTIA/${DATE}/${DATE}*UKMO-L4*OSTIA*.nc            \
          -o ${DIR}/DATA_realtime/OSTIA/ioda-v2/${year}/${DATE}/sst_ostia_${DATE}.nc

echo " "
echo $date DONE

### superob
echo " "
echo "Superoobing to 1deg grid"

export input_file=${DIR}/DATA_realtime/OSTIA/ioda-v2/${year}/${DATE}/sst_ostia_${DATE}.nc
export output_file=${DIR}/DATA_realtime/OSTIA/superob/${year}/${DATE}/sst_ostia_${DATE}.nc

mkdir -p ${DIR}/DATA_realtime/OSTIA/superob/${year}/${DATE}

cd ${DIR}/DATA_realtime/OSTIA
cp -p config_ORG.yaml config.yaml
sed -i "s;__INPUT_FILE__;${input_file};g" config.yaml
sed -i "s;__OUTPUT_FILE__;${output_file};g" config.yaml

srun -n 1 $SRC/obs_superob.x config.yaml

echo "Superobbing DONE"
