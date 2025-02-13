
Purpose: convert RTOFS operational obs binary to IODA V2

(1) Extract RTOFS obs binary data from HPSS, e.g., 
module load hpss
htar -xv -f /NCEPDEV/emc-ocean/5year/emc.ncodapa/emc_parallel/ncoda.20210211/ocnqc.tar 

(2) Shell scripts orgnize data files, call Fortran programs and python scripts to convert RTOFS bin to IODA v2. Fortran codes read RTOFS bin and write ascii. python scripts read ascii and write IODA v2.


