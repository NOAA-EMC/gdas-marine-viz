# Marine GDAS Verification Tools: Daily Mean Time Series

This tool generates time series comparing experiment background forecasts and a reference dataset for SST, SSH, or SEAICE fields.

## Before running the tool, add the following files to the /src directory (this only needs to be done once) 
#### On Hera:
/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/data/soca_gridspec.nc
/scratch1/NCEPDEV/da/common/validation/oceanmask_global_0.25deg.20221025.xesmf_nearest_s2d.nc

## How to run
#### In RUN_PLOT.time_series.daily.bash, set the following parameters:
yyyyS, mmS, ddS ....... Start date (year, month, day)
yyyyE, mmE, ddE ....... End date (year, month, day)
variable .............. variable to plot: SST, SSH, SEAICE
reference ............. reference dataset: ostia (SST or SEAICE), copernicus (SSH)
oceans ................ ocean region to plot: Global, Atlantic, Pacific, Indian, Arctic, Southern
account ............... HPC account
qos ................... HPC QOS
partition ............. HPC partition
timelimit ............. HPC max time for job
ntasks ................ HPC number of tasks (should = 1)
#### Run the script:
./RUN_PLOT.time_series.daily.bash

## How to add a new experiment to the time series
#### In src/MAIN_PLOT.time_series.MAE.py, add the experiment path, experiment name, and a unique colorname to the end of the following lists, respectively:
exptpaths ............. add new experiment path to the end of this list
exptnames ............. add new experiment name to the end of this list
colors ................ add line color name for the new experiment to the end of this list
#### NOTE: Make sure to add a comma to the end of the previous item in each of the above lists
