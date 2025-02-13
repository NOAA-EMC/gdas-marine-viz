#--------------------------------------------------------------------------------
# adt.nesdis_rads.sh
#  download altimetry data from NESDIS.
#--------------------------------------------------------------------------------

set -eu

if [[ $# != 3 ]]; then
    echo "usage: $0 [<sat>|ALL] yyyymmdd output_path"
    exit 1
fi

sat=$1
date=$2
yr=${date:0:4}
dy=$(date -d "$date" "+%j")
out_dir=$3
source="ftp://ftp.star.nesdis.noaa.gov/pub/sod/lsa/rads/adt/${yr}/"

pwd=$(pwd)

for f in $files; do
    # make sure it is the right day
    [[ ! $f =~ ^rads_adt_.._$yr$dy.*$ ]] && continue

    # make sure this is the right satellite
    s=${f:9:2}
    if [[ "$sat" != "ALL" ]]; then
        [[ "$s" != "$sat" ]] && continue
    fi

    d=$out_dir/${yr}/${date}
    mkdir -p $d
    cd $d
    wget $source/$f
    cd $pwd
done
