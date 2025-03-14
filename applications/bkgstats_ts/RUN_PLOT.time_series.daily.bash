#!/bin/bash

#==============================================
# BEGIN USER INPUT
#==============================================

#----------------------------------------------
# Set date range over which to run the program
#	For computational efficiency, it is recommended to loop through one month at a time. 
#	Choose one year/month combination and an array of days.
#
# yyyy 	= year
# mm	= month
# ddarr = day array

# --- Start Date
yyyyS="2021"
mmS="07"
ddS="01"

# --- End Date
yyyyE="2021"
mmE="07"
ddE="05"

#----------------------------------------------
# SET INPUT PARAMETERS
#	'variable'  = variable to be plotted
#	'reference' = name of reference dataset to compare against 'variable'
#	'oceans'    = ocean basin regions over which to plot time series

# --- Variable
variable="SST"
#variable="SEAICE"
#variable="SSH"

# --- Reference dataset
reference="ostia"		# SST, SEAICE
#reference="copernicus"		# SSH

# --- Ocean regions ... Ignored for variable="SEAICE"
#oceans=("Global" "Arctic" "Atlantic" "Indian" "Pacific" "Southern")
#oceans=("Global")
oceans=("Atlantic")

#----------------------------------------------
# SET HPC INFO
#	Account, partition, and runtime limit for jobs

account="da-cpu"		# Account used for SLURM jobs. String format

qos="debug"			# QOS used for SLURM jobs. String format

partition="hera"		# Partition used for SLURM jobs. String format

timelimit="00:30:00"	# Timelimit for SLURM job. String format
#timelimit="01:00:00"

ntasks=1

#==============================================
# END USER INPUT
#==============================================

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!! USERS SHOULD NOT CHANGE ANYTHING BELOW THIS LINE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#``````````````````````````````````````
# Set home path 'dir_home' (i.e., where this script is located)

dir_home="$PWD"
echo 'WORKING DIRECTORY = '${dir_home}

#----------------------------------------------
# Create output directory (where figures go) if it doesn't already exist

dir_out=${dir_home}"/output/"

if [[ ! -d ${dir_out} ]] ; then
  mkdir -p ${dir_out}
fi

#----------------------------------------------
# Find sizes of arrays

nexptpaths=${#exptpaths[@]}
nexptnames=${#exptnames[@]}
noceans=${#oceans[@]}

echo 'num exptpaths = '$nexptpaths
echo 'num exptnames = '$nexptpaths
echo 'num oceans = '$noceans

#----------------------------------------------
# Set working paths
#	Always end path names with a slash "/"

dir_src=${dir_home}"/src/"           #location of collocation source code
echo "SOURCE CODE DIRECTORY = "${dir_src}

#----------------------------------------------
# Set name of run job script (to pass as argument to run job script)
#       This script actually runs the python code

run_python_code="run.time_series.daily.job"

#==============================================
#==============================================
# Set up and run SLURM commands
#==============================================

echo "-- Start Date: $yyyyS $mmS $ddS"
echo "-- End Date:   $yyyyE $mmE $ddE"
dateSTART=$yyyyS$mmS$ddS
dateEND=$yyyyE$mmE$ddE

cd ${dir_home}

#----------------------------------------------
# Input arguments for SLURM (sbatch) commands
#
#       Arguments (arg) to customize sbatch command:
#               1. start date ....................... start of datetime range for plotting
#		2. end date ......................... end of datetime range for plotting
#		3. output path ...................... where output figures are found
#		4. variable ......................... variable to plot
#		5. reference ........................ reference dataset to compare against variable
#		6. oceans ........................... ocean basin over which to plot time series (not used for SeaIce)

	# All possible sbatch arguments
arg1=${dateSTART}
arg2=${dateEND}
arg3=${dir_out}
arg4=${variable}
arg5=${reference}
#arg6 <-- Defined in LOOP below

echo 'ARG 1: '$arg1
echo 'ARG 2: '$arg2
echo 'ARG 3: '$arg3
echo 'ARG 4: '$arg4
echo 'ARG 5: '$arg5

echo 'oceans: '$oceans

#----------------------------------------------
# Set up sbatch command(s) and run 
#
#       Run command format for each job:
#               sbatch $output_logfile_name $job_name $input_directory $colloc_script $arg1 /
#               $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 $partition $timelimit $run_job_script

jname="TimeSeries"
python_code="MAIN_PLOT.time_series.MAE.py"
echo $dir_src" / "$python_code

#-------------------------------------
# RUN time series scripts

if [[ $variable == "SEAICE" ]] ; then
  echo 'Plot SeaIce'

  arg6="NA"

  log=LOG_${jname}_${arg4}_${arg5}_${dateSTART}_${dateEND}            # log name containing output log info
  jlog=${jname}_${arg4}_${arg5}_${dateSTART}_${dateEND} 	      # log name for job in queue

  rm $log                                             # remove old log file

  sbatch --mem=0 --output=${log} --job-name=${jlog} --account=${account} --partition=${partition} --qos=${qos} --time=${timelimit} --ntasks=${ntasks} --export=INDIR=${dir_src},SCRIPT=${python_code},ARG1=${arg1},ARG2=${arg2},ARG3=${arg3},ARG4=${arg4},ARG5=${arg5},ARG6=${arg6} ${run_python_code}

else 
  echo 'Plot SST or SSH: '$variable

  iocean=0
  while [[ $iocean -lt $noceans ]]
  do

    arg6=${oceans[$iocean]}
    echo 'ocean basin: '$arg6

    log=LOG_${jname}_${arg4}_${arg5}_${arg6}_${dateSTART}_${dateEND}            # log name containing output log info
    jlog=${jname}_${arg4}_${arg5}_${arg6}_${dateSTART}_${dateEND}         # log name for job in queue

    rm $log                                             # remove old log file

    sbatch --mem=0 --output=${log} --job-name=${jlog} --account=${account} --partition=${partition} --qos=${qos} --time=${timelimit} --ntasks=${ntasks} --export=INDIR=${dir_src},SCRIPT=${python_code},ARG1=${arg1},ARG2=${arg2},ARG3=${arg3},ARG4=${arg4},ARG5=${arg5},ARG6=${arg6} ${run_python_code}

    let iocean=iocean+1
  done

fi

#----------------------------------------------
#==============================================

##############################################################
# END
##############################################################
