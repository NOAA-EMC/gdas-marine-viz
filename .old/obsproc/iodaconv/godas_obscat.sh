#!/bin/bash

set -ex

source godas_obs2ioda.utils

############################################################
# Process the input options. Add options as needed.        #
############################################################
while getopts ":hs:e:i:o:t:d:n:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      s) # Enter start date
         date_start=$OPTARG;;
      e) # Enter end date
         date_end=$OPTARG;;
      i) # Enter input path
         input_path=$OPTARG;;
      o) # Enter output path
         output_path=$OPTARG;;
      t) # Enter obs type
         obs=$OPTARG;;
      d) # Enter obs descriptor
         insts=$OPTARG;;
      n) # Enter num pe's
         N=$OPTARG;;
     
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

echo "s "$date_start
echo "e "$date_end
echo "i "$input_path
echo "o "$output_path
echo "t "$obs
echo "d "$insts
echo "n "$n

socasci_bin=/work/noaa/marine/Guillaume.Vernieres/runs/s2s/build.intel/bin
socasci_src=/work/noaa/marine/Guillaume.Vernieres/runs/s2s/soca-science


date=$date_start
while [[ $date -le $date_end ]]; do
    echo "=============================="
    echo "------ $date"
    echo "=============================="
    lof=`ls $input_path/${date}/*2021010100*.nc`
    rm -r tmp
    mkdir -p tmp
    i=0
echo $lof
    for f in $lof; do
        echo $f
	printf -v j "%04d" $i
	ln -s $f ./tmp/obs_$j.nc
	i=$((i+1)) 
    done
    cat << EOF > ./tmp/test.sh
#!/bin/bash
#SBATCH --job-name=iodaconv-test
#SBATCH --ntasks=6 
#SBATCH --qos=debug
#SBATCH --time=00:05:00
#SBATCH --account=marine-cpu
#SBATCH --output=cat%j.log
#SBATCH --error=cat%j.err
#SBATCH --exclusive
    
echo $PWD

source ${socasci_src}/configs/machine/machine.orion.intel
srun -n 6 ${socasci_bin}/obs_cat.x -i obs -o test.nc

EOF
    
    date=$(date -d "$date + 1 day" +%Y%m%d)

done
wait
