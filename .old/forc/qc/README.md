Sanity Check of DATM forcing by comparison of two data.

-- set the input data
- edit forcing_datm_sanity.yaml

# year and month
ym: 202108
# date and hour
dh: 1512
# input dir 1
indir1: "/work/noaa/ng-godas/marineda/DATM_INPUT/"
d1nm: "CFSR"
d1ad: "cfsr"
# input dir 2
indir2: "/work/noaa/ng-godas/marineda/DATM_INPUT/"
d2nm: "GFS"
d2ad: "gfs"
# output path
plotpath: "/work/noaa/da/Hyun-Chul.Lee/godas/forc/SanityCheck/"

-- run
python ./forcing_datm_sanity.py

-- output
locates at $plotpath as png file. 


