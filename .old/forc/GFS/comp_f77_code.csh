#! /bin/csh

module load impi/2018.4
module load intel/2018.4

set name = conv_gfs2datm_long_beta4

ifort -132 ${name}.f -o ${name} -mcmodel=medium -L/apps/contrib/NCEPLIBS/orion/external/netcdf-4.5.0/lib -lnetcdff -lnetcdf
