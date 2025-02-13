#!/bin/bash
set -e
while getopts "s:e:t:" option; do
   case $option in
      s) # Enter start date
         date_start=$OPTARG;;
      e) # Enter end date
         date_end=$OPTARG;;
      t) # job time hh:mm:ss
         job_time=$OPTARG;;
   esac
done

jobname=iodaconv-${date_start}_${date_end}

cat << EOF > $jobname.sh
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --nodes=1 
##SBATCH --qos=debug
#SBATCH --qos=batch
#SBATCH --time=$job_time
#SBATCH --account=marine-cpu
#SBATCH --output=${jobname}%j.log
#SBATCH --error=${jobname}%j.err
#SBATCH --exclusive
#SBATCH --partition=orion

echo $PWD

time ./godas_obs2ioda.sh -s ${date_start} -e ${date_end} -i $PWD/data/ -o $PWD/ioda_data/ -t sst -d 'avhrr_l3u' -n 40
EOF

sbatch $jobname.sh
