#!/bin/bash

set -e

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
      n) # Enter number of  pe's
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

iodaconvbuild=/work/noaa/marine/Guillaume.Vernieres/OBS/sprint1of4/convert2ioda/build.iodaconv
socasci_src=/work/noaa/marine/Guillaume.Vernieres/runs/s2s/soca-science
iodaconvbin=${iodaconvbuild}/bin
pyioda_path=`ls -d ${iodaconvbuild}/lib/python*/pyioda`

SOCA_SCIENCE_RUNTIME=F
source ${socasci_src}/configs/machine/machine.orion.intel

date=$date_start
while [[ $date -le $date_end ]]; do
    echo "=============================="
    echo "------ $date"
    echo "=============================="
    outdir=$output_path/$date
    mkdir -p $outdir
    for inst in $insts; do
        echo "    Processing $inst:"
        loi=$(infiles_wildcard $inst) # list of instruments
        prov=$(provider $inst)        # a descriptor for the file

        for w in $loi; do
            infiles=$(ls -1 ${input_path}/${obs}.${inst}.${prov}/${date}/*${w} | awk '{ ORS=" "; print; }')
	    obs_descriptor=$(inst2filename $w)
	    for f in $infiles; do
		ymdhm=`basename $f`
		ymdhm=`echo $ymdhm | grep -Eo [0-9]{12}`
		fout=$outdir/${obs}_${obs_descriptor}_${ymdhm}.nc
		convert2ioda="$iodaconvbin/gds2_sst2ioda.py -i $f -o $fout -d ${date}12"
		echo "        starting job $(basename ${fout}).sh"

		cat << EOF > ${fout}.sh
                export PYTHONPATH=$PYTHONPATH:$pyioda_path
                $convert2ioda
                rm ${fout}.sh
                rm ${fout}.log
EOF
		chmod +x ${fout}.sh
		command1="${fout}.sh"
		$command1 > ${fout}.log &
		numjobs=`jobs | wc -l`
		while [ `jobs | wc -l` -ge $N ] ; do
                    sleep 1
		done
	    done
        done
    done
    echo "numjobs: $numjobs"
    date=$(date -d "$date + 1 day" +%Y%m%d)
done
wait
