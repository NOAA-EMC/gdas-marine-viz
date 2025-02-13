# `dbtools`
Wrapper to JCSDA tools to manage the EMC marine observation database.
You will need to load the jedi modules and use the version of R2D2 that is
currently used in soca-science. The simplest way to get have access to all the dependencies
is to source the appropriate machine files in soca-science.

1 - To test/build
``` console
mkdir build
cd build
ecbuild ../godas/obsproc/dbtools/
make
ctest
```

2 - DO NOT USE THESE TOOLS UNLESS YOU ARE A db GATEKEEPER

**Usage example for storing obs:**

 Populating the database `gdas_marine` database with insitu `sst` from `ships` and `tracks`:
``` console
ln -s <path/to/old/ioda/observations/obs> obs
./godas_store_obs.py --start 2015-01-01T00:00:00Z \
                     --end 2021-12-31T00:00:00Z \
                     --source_dir ./obs/ \
                     --source_file ymd \
                     --source_file_ext nc \
                     --provider gdas_marine \
                     --experiment s2s_v1 \
                     --obstype sst \
                     --platforms ship_fnmoc trak_fnmoc \
                     --storage shared \
                     --shared_db /work/noaa/ng-godas/r2d2/ \
                     --step P1D
```

**Usage example for superobing obs:**


``` console
mkdir scratch
cd scratch
cp ..../godas_superob.sbatch godas_superob.sbatch.metopa # Edit for your needs
sbatch godas_superob.sbatch.metopa
```
