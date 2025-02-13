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

# Convert NSIDC CDR data (NH & SH) to ioda-v1 format and then to ioda-v2
# TODO: update nsidc_ice2ioda.py to convert to ioda-v2
#

module load nco

export DIR=/lustre/f2/scratch/Shastri.Paturi
export SRC=${DIR}/sandbox/20220623/build/bin
export MACHINE=gaea.intel

source $DIR/sandbox/20220613/soca-science/configs/machine/machine.gaea.intel
export PYTHONPATH=${SRC}/../lib/python3.7/pyioda:$PYTHONPATH

export DATE=20190831
year=${DATE:0:4}

mkdir -p $DIR/DATA_realtime/NSIDC/ioda-v2/${year}/${DATE}
# NSIDC sea-ice concentration files have latitude, longitude given in a separate file
# Add latitude, longitude to the sea-ice concentration file

for reg in nh sh; do
   # append latitude, longitude to the input NSIDC nh and sh files
   ncks -A -C -v latitude,longitude $DIR/DATA_realtime/NSIDC/G02202-cdr-ancillary-${reg}.nc \
                 $DIR/DATA_realtime/NSIDC/${year}/${DATE}/seaice_conc_daily_${reg}_${date}_f17_v04r00.nc
   srun -n 1 $SRC/nsidc_ice2ioda.py -i $DIR/DATA_realtime/NSIDC/${year}/${DATE}/seaice*${reg}*nc   \
          -o $DIR/DATA_realtime/NSIDC/ioda-v2/${year}/${DATE}/icec_nsidc_${reg}_${DATE}.nc \
          -d ${DATE}'12'
   srun -n 1 $SRC/ioda-upgrade.x $DIR/DATA_realtime/NSIDC/ioda-v2/${year}/${DATE}/icec_nsidc_${reg}_${DATE}.nc \
                         $DIR/DATA_realtime/NSIDC/ioda-v2/${year}/${DATE}/icec_nsidc_${reg}_${DATE}_v2.nc
   mv $DIR/DATA_realtime/NSIDC/ioda-v2/${year}/${DATE}/icec_nsidc_${reg}_${DATE}_v2.nc \
      $DIR/DATA_realtime/NSIDC/ioda-v2/${year}/${DATE}/icec_nsidc_${reg}_${DATE}.nc
done
