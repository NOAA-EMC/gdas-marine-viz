## Description

* This package generates DATM input data for the forcing fluxes at the sea surface from the NCEP operational GFSv16 outputs.

## Inputs of conversion in conv_gfs2datm_long_beta4.f

* INPUT Sources 

I. `./gdas.${dat}/${hh}/atmos/gdas.t${hh}z.sfluxgrbf000.grib2` 

II. `./gdas.${datb}/${hhb}/atmos/gdas.t${hhb}z.sfluxgrb${hprep}.grib2` :: 6 hour forecast fields at the analysis time

III. `./gdas.${dat}/${hh}/atmos/gdas.t${hh}z.atmf000.nc`  

| Input variable |	Source	| Output variable|
| --- | --- | ---|
| longitude	| I| lon |
| latitude |I |lat |
| time | I| time |
|DLWRF_surface  |I |DLWRF |
|ULWRF_surface  | I|ULWRF |
| DSWRF_surface |I |DSWRF |
|DSWRF_surface |I |vbdsf_ave |
|DSWRF_surface |I | vddsf_ave|
|DSWRF_surface |I |nbdsf_ave |
|DSWRF_surface |I | nddsf_ave|
| UFLX_surface|II | dusfc|
|VFLX_surface |II |dvsfc |
|SHTFL_surface |I | shtfl_ave|
|LHTFL_surface |I |lhtfl_ave |
| PRATE_surface|I |totprcp_ave |
| UGRD_10maboveground|I |u10m |
|VGRD_10maboveground |I |v10m |
|delz |III |hgt_hyblev1 |
|PRES_surface |I | psurf|
|PRATE_surface |II |precp |
|PRATE_surface |II | fprecp|
|ICEC_surface  | I|icecsfc |
|TMP_1hybridlevel |I |tmp_hyblev1 |
| SPFH_1hybridlevel|I |spfh_hyblev1 |
|UGRD_1hybridlevel |I |ugrd_hyblev1 |
|VGRD_1hybridlevel |I | vgrd_hyblev1|
|SPFH_2maboveground |I |q2m |
|TMP_2maboveground |I |t2m |
| LAND_surface|I | slmsksfc|
|ICEC_surface |I | icecsfc|





## Estimation and calculation
* vbdsf_ave[W/m**2], vddsf_ave[W/m**2], nbdsf_ave[W/m**2], nddsf_ave[W/m**2] are estimated from DSWRF_surface[W/m**2], multiplied by 0.285 (vbdsf_ave, addsf_ave) and 0.215 (nbdsf_ave, nddsf_ave).

* hgt_hyblev1 are obtained by delz  : -1* delz

* precp and fprecp are estimated from PRATE_surface with the thresh hold of TMP_2maboveground by -15oC.

* pres_hyblev1 are calculated from `delz`, `TMP_1HYBRIDLEVEL`, `PRES_SURFACE`.

* In the GFS_DATM, there is no bulk formula calculation


## Code and scripts
* comp_f77_code.csh : compilation script for the code
* get_gfs_beta4_gdas.csh : script to import input files from HPSS and merge variables into an input file 
* conv_gfs2datm_long_beta4.f : convert code from input file to DATM output
* conv_gfs2datm.fort_beta4.csh : script for conversion
* run_gfs2datm_inter.csh : wrap script for getting input files and variables and converting to DATM output

## Instruction of conversion
 - Compile the code by comp_f77_code.csh
 - Set configurations in run_gfs2datm_inter.csh
 - Run the script : csh run_gfs2datm_inter.csh
 - Check the output with /work/noaa/ng-godas/marineda/DATM_INPUT/GFS/ in Orion


## Issue(s) addressed

Document the code that generates the forcing :
https://github.com/NOAA-EMC/godas/issues/276



## Testing
The validation of the GFS_DATM from this package has been done by the comparisons of CFSR_DATM/GEFS_DATM in the same periods.


## Dependencies

ng-godas, the mediator in UFS, updates of GFSv16

