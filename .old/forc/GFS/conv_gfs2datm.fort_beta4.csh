#! /bin/csh -x
#-- hyun-chul.lee@noaa.gov

module load netcdf/4.5.0
module load cdo/1.9.8
module load wgrib2/3.0.2

set hdir = /work/noaa/da/Hyun-Chul.Lee/GFS
set sdir = ${hdir}/Reanal
set tdir = ${hdir}/GFS2DATM
set wdir = ${hdir}/Work
set grb2 = /apps/intel-2018/wgrib2-3.0.2/wgrib2/wgrib2

set symd = $1
set eymd = $2

echo $symd $eymd

cd $wdir

/bin/cp ${hdir}/conv_gfs2datm_long_beta4 conv_gfs2datm_long_beta4

set ymd = $symd
while ($ymd <= $eymd)
  foreach hh (00 06 12 18)
    set ymdh = ${ymd}${hh}
    echo ${ymdh}
    rm gfs_input*.nc
    #-- convert input grib2 files to netcdf format
    ${grb2} ${sdir}/gfs.${ymd}.t${hh}z.sfcanl.grib2 -netcdf gfs_input1.nc
    ${grb2} ${sdir}/gfs.${ymd}.t${hh}z.puvflx.grib2 -netcdf gfs_input2.nc
    cp ${sdir}/gfs.${ymd}.t${hh}z.atmf000.delz1.nc gfs_input3.nc
    #-- concartnate nc files to an input file of gfs_input.nc
    cdo merge gfs_input1.nc gfs_input2.nc gfs_input3.nc gfs_input.nc

    echo ${ymdh}
     ./conv_gfs2datm_long_beta4
    #-- 
ncdump gfs_output.nc | sed -e "5s#^.time = 1 ;#time = UNLIMITED ; // (1 currently)#" | ncgen -o gfs_output2.nc
    mv gfs_output2.nc gfs_output.nc
    /bin/mv gfs_output.nc ${tdir}/gfs.${ymdh}.nc

#   rm gfs_input.*
  end
  set ymd = `date -d "${ymd} 1 day" +%Y%m%d`
end

