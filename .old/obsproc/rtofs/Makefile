#!/bin/bash

ifort read_ssh.F90 -O2 -fpp -traceback -xHost -fno-alias -fp-model strict -no-fma -ftz -mkl -DLINUX -convert big_endian -assume byterecl -fixed -real_size 32 -i4 -o read_ssh.exe

ifort read_sss.F90 -O2 -fpp -traceback -xHost -fno-alias -fp-model strict -no-fma -ftz -mkl -DLINUX -convert big_endian -assume byterecl -fixed -real_size 32 -i4 -o read_sss.exe

ifort read_sst.F90 -O2 -fpp -traceback -xHost -fno-alias -fp-model strict -no-fma -ftz -mkl -DLINUX -convert big_endian -assume byterecl -fixed -real_size 32 -i4 -o read_sst.exe

ifort read_profile.F90 -O2 -fpp -traceback -xHost -fno-alias -fp-model strict -no-fma -ftz -mkl -DLINUX -convert big_endian -assume byterecl -fixed -real_size 32 -i4 -o read_profile.exe

