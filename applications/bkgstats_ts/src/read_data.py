import xarray as xr
import os
from os.path import exists
import numpy as np
import glob

###########################################################################
# read_data module
###########################################################################
#
# FUNCTIONS
#


# =========================================================================
# Get ocean model grid (lats, lons)
#
def get_ocngrid():
    # Get current working directory (CWD)
    cwd = os.getcwd()
    # Open the NetCDF file using xarray
    grid_fname = str(cwd) + '/soca_gridspec.nc'
    ds = xr.open_dataset(grid_fname)
    lat = np.squeeze(ds['lat'][:])
    lon = np.squeeze(ds['lon'][:])
    ds.close()

    return lat.values, lon.values


# =========================================================================
# Read ocean mask file
#
def read_ocean_mask(ocean_choice):
    ocean = ocean_choice.lower()
    # Get current working directory (CWD)
    cwd = os.getcwd()
    # Open the NetCDF file using xarray
    fname = "oceanmask_global_0.25deg.20221025.xesmf_nearest_s2d.nc"
    ds = xr.open_dataset(str(cwd) + "/" + fname)
    lat = np.squeeze(ds['lat'][:])      # 1D array
    lon = np.squeeze(ds['lon'][:])      # 1D array
    # Get ocean mask
    ocean_mask = np.squeeze(ds['open_ocean'][:])
    # Set ocean mask value dependent on ocean_choice
    if ocean.find("arctic") != -1:
        iocean = 4
    elif ocean.find("atlantic") != -1:
        iocean = 1
    elif ocean.find("indian") != -1:
        iocean = 3
    elif ocean.find("pacific") != -1:
        iocean = 2
    elif ocean.find("southern") != -1:
        iocean = 5
    else:
        iocean = -1
    # Set ocean mask to missing if not 'global'
    if iocean > 0:
        mask_out = np.where(ocean_mask == iocean, ocean_mask, np.nan)
    else:
        mask_out = ocean_mask
    # Close the NetCDF file
    ds.close()

    return mask_out, lat, lon


# =========================================================================
# Read Copernicus SSH (altimetry) level-4 analysis on ocean model grid
# ... Dataset info: https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/description
# ... NOTE: These files were interpolated by someone else. There is no error information.
#
def read_copernicus(yyyymmdd):
    # Open the NetCDF file using xarray
    fname = '/scratch1/NCEPDEV/da/common/validation/adtl4coper2mom6/' + yyyymmdd + '_L4adt_2_global_0.25deg.nc'
    ds = xr.open_dataset(fname)
    # Get the SSH data variable
    ssh = np.squeeze(ds['adt'][:])          # np.squeeze: removes all axes of length 1 from array
    # Close the NetCDF file
    ds.close()

    return ssh.values


# =========================================================================
# Get ORAS5 (ECMWF) analysis on ocean model grid: SST or SSH
#
def read_ecm(yyyy, mm, dd, var):
    # Open the NetCDF file using xarray
    yyyymmdd = str(yyyy) + '_' + str(mm) + '_' + str(dd)
    ecm_fnames = glob.glob('/scratch1/NCEPDEV/da/common/validation/oras5replay/ocn_' + yyyymmdd + '_*.nc')
    # Loop through ecm_fnames
    cnt = 0
    for iname in ecm_fnames:
        if exists(iname):
            ds = xr.open_dataset(iname)
            # Get appropriate variable
            if cnt == 0:
                if var.lower() == 'sst':
                    sst = np.squeeze(ds['temp'][0, 0, :, :])
                elif var.lower() == 'ssh':
                    sst = np.squeeze(ds['SSH'][0, :, :])
            else:
                if var.lower() == 'sst':
                    sst += np.squeeze(ds['temp'][0, 0, :, :])
                elif var.lower() == 'ssh':
                    sst += np.squeeze(ds['SSH'][0, :, :])
            cnt += 1
            ds.close()
        else:
            sst = np.nan

    return sst.values / cnt


# =========================================================================
# Get OSTIA level-4 analysis on ocean model grid: SST
# ... Dataset info: https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description
#
def read_ostia(yyyymmdd):
    # Open the NetCDF file using xarray
    dpath = '/scratch1/NCEPDEV/da/common/validation/ostia2mom6_xesmf/'
    fname = 'ostia_global_0.25deg.' + yyyymmdd + '_12z.xesmf.nc'
    ostia_fname = dpath + fname
    ds = xr.open_dataset(ostia_fname)
    sst = np.squeeze(ds['analysed_sst'][:])
    mask = np.squeeze(ds['mask'][:])
    # Close the NetCDF file
    ds.close()

    return sst.values, mask.values


# =========================================================================
# Get OSTIA level-4 analysis on ocean model grid: SEAICE
# ... Dataset info: https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description
#
def read_ostia_ice(yyyymmdd):
    # Open the NetCDF file using xarray
    dpath = '/scratch1/NCEPDEV/da/common/validation/ostia2mom6_xesmf/'
    fname = 'ostia_global_0.25deg.' + yyyymmdd + '_12z.xesmf.nc'
    ostia_fname = dpath + fname
    ds = xr.open_dataset(ostia_fname)
    icec = np.squeeze(ds['sea_ice_fraction'][:])           # np.squeeze: removes all axes of length 1 from array
    mask = np.squeeze(ds['mask'][:])
    # Close the NetCDF file
    ds.close()

    return icec.values, mask.values


# =========================================================================
# Get ocean model SEAICE background forecasts
#
def read_icebkg(exptpath, yyyy, mm, dd):
    # Date before yyyymmdd
    # ...year
    byyyy = yyyy
    # ...month
    if int(dd) == 1:
        tbmm = int(mm) - 1
    else:
        tbmm = int(mm)
    if tbmm < 10:
        bmm = "0" + str(tbmm)
    else:
        bmm = str(tbmm)
    # ...day
    if bmm == mm:
        tbdd = int(dd) - 1
        if tbdd < 10:
            bdd = "0" + str(tbdd)
        else:
            bdd = str(tbdd)
    else:
        if bmm == "04" or bmm == "06" or bmm == "09" or bmm == "11":
            bdd = "30"
        elif bmm == "02":
            if int(yyyy) % 4 == 0:
                bdd = "29"
            else:
                bdd = "28"
        else:
            bdd = "31"

    byyyymmdd = byyyy + bmm + bdd
    yyyymmdd = yyyy + mm + dd

    # Get input paths
    if exptpath.find("marine_") != -1 or exptpath.find("cp4") != -1:
        modelstr = "model"
    else:
        modelstr = "model_data"

    # Compute daily average centered around 12z
    # ... Current approach is to only use f006 background files
    if exptpath.find("role") != -1 or exptpath.find("marine_") != -1 or exptpath.find("S2Sda") != -1:
        bprefix_path = exptpath + "/" + byyyymmdd
        prefix_path = exptpath + "/" + yyyymmdd

        bfname18_006 = bprefix_path + "18/gdas." + byyyymmdd + "/18/" + modelstr + "/ice/history/gdas.ice.t18z.inst.f006.nc"
        fname00_006 = prefix_path + "00/gdas." + yyyymmdd + "/00/" + modelstr + "/ice/history/gdas.ice.t00z.inst.f006.nc"
        fname06_006 = prefix_path + "06/gdas." + yyyymmdd + "/06/" + modelstr + "/ice/history/gdas.ice.t06z.inst.f006.nc"
        fname12_006 = prefix_path + "12/gdas." + yyyymmdd + "/12/" + modelstr + "/ice/history/gdas.ice.t12z.inst.f006.nc"
    else:
        bfname18_006 = exptpath + "/gdas." + byyyymmdd + "/18/" + modelstr + "/ice/history/gdas.ice.t18z.inst.f006.nc"
        fname00_006 = exptpath + "/gdas." + yyyymmdd + "/00/" + modelstr + "/ice/history/gdas.ice.t00z.inst.f006.nc"
        fname06_006 = exptpath + "/gdas." + yyyymmdd + "/06/" + modelstr + "/ice/history/gdas.ice.t06z.inst.f006.nc"
        fname12_006 = exptpath + "/gdas." + yyyymmdd + "/12/" + modelstr + "/ice/history/gdas.ice.t12z.inst.f006.nc"

    # Use only f006 bkg files
    gdas_fnames = [bfname18_006]
    gdas_fnames.append(fname00_006)
    gdas_fnames.append(fname06_006)
    gdas_fnames.append(fname12_006)

    # Check if files for all analysis times (00,06,12,18) are available
    gdas_check = []
    for gdas_fname in gdas_fnames:
        if exists(gdas_fname):
            gdas_check.append(gdas_fname)
    gdas_check = str(gdas_check)
    if gdas_check.find("00z") == -1 or gdas_check.find("06z") == -1 or gdas_check.find("12z") == -1 or gdas_check.find("18z") == -1:
        print("read_icebkg: NO FILES for " + exptpath)
        return "NO FILES"
    else:
        print("read_icebkg: FILES EXIST for " + exptpath)

    # Extract data
    cnt = 0.0
    for gdas_fname in gdas_fnames:
        if exists(gdas_fname):
            ds = xr.open_dataset(gdas_fname)
            if cnt == 0:
                sst = np.squeeze(ds['aice_h'][0, :, :])
            else:
                sst += np.squeeze(ds['aice_h'][0, :, :])
            cnt += 1
            ds.close()

    if cnt > 0:
        return sst.values / cnt
    else:
        return "NO FILES"


# =========================================================================
# Get ocean model SSH background forecasts
#
def read_ocnbkg_ssh(exptpath, yyyy, mm, dd):
    # Date before yyyymmdd
    # ...year
    byyyy = yyyy
    # ...month
    if int(dd) == 1:
        tbmm = int(mm) - 1
    else:
        tbmm = int(mm)
    if tbmm < 10:
        bmm = "0" + str(tbmm)
    else:
        bmm = str(tbmm)
    # ...day
    if bmm == mm:
        tbdd = int(dd) - 1
        if tbdd < 10:
            bdd = "0" + str(tbdd)
        else:
            bdd = str(tbdd)
    else:
        if bmm == "04" or bmm == "06" or bmm == "09" or bmm == "11":
            bdd = "30"
        elif bmm == "02":
            if int(yyyy) % 4 == 0:
                bdd = "29"
            else:
                bdd = "28"
        else:
            bdd = "31"

    byyyymmdd = byyyy + bmm + bdd
    yyyymmdd = yyyy + mm + dd

    # Get input path
    if exptpath.find("marine_") != -1 or exptpath.find("cp4") != -1:
        modelstr = "model"
    else:
        modelstr = "model_data"

    # Compute daily average centered around 12z
    # ... Current approach is to only use f006 background files
    if exptpath.find("role") != -1 or exptpath.find("marine_") != -1 or exptpath.find("S2Sda") != -1:
        bprefix_path = exptpath + "/" + byyyymmdd
        prefix_path = exptpath + "/" + yyyymmdd

        bfname18_006 = bprefix_path + "18/gdas." + byyyymmdd + "/18/" + modelstr + "/ocean/history/gdas.ocean.t18z.inst.f006.nc"
        fname00_006 = prefix_path + "00/gdas." + yyyymmdd + "/00/" + modelstr + "/ocean/history/gdas.ocean.t00z.inst.f006.nc"
        fname06_006 = prefix_path + "06/gdas." + yyyymmdd + "/06/" + modelstr + "/ocean/history/gdas.ocean.t06z.inst.f006.nc"
        fname12_006 = prefix_path + "12/gdas." + yyyymmdd + "/12/" + modelstr + "/ocean/history/gdas.ocean.t12z.inst.f006.nc"
    elif exptpath.find("cp4") != -1:
        bfname18_006 = exptpath + "/gdas." + byyyymmdd + "/18/" + modelstr + "/ocean/history/gdas.ocean.t18z.inst.f006.nc"
        fname00_006 = exptpath + "/gdas." + yyyymmdd + "/00/" + modelstr + "/ocean/history/gdas.ocean.t00z.inst.f006.nc"
        fname06_006 = exptpath + "/gdas." + yyyymmdd + "/06/" + modelstr + "/ocean/history/gdas.ocean.t06z.inst.f006.nc"
        fname12_006 = exptpath + "/gdas." + yyyymmdd + "/12/" + modelstr + "/ocean/history/gdas.ocean.t12z.inst.f006.nc"
    else:
        bfname18_006 = exptpath + "/gdas." + byyyymmdd + "/18/" + modelstr + "/ocean/history/gdas.t18z.ocnf006.nc"
        fname00_006 = exptpath + "/gdas." + yyyymmdd + "/00/" + modelstr + "/ocean/history/gdas.t00z.ocnf006.nc"
        fname06_006 = exptpath + "/gdas." + yyyymmdd + "/06/" + modelstr + "/ocean/history/gdas.t06z.ocnf006.nc"
        fname12_006 = exptpath + "/gdas." + yyyymmdd + "/12/" + modelstr + "/ocean/history/gdas.t12z.ocnf006.nc"

    # Use only f006 bkg files
    gdas_fnames = [bfname18_006]
    gdas_fnames.append(fname00_006)
    gdas_fnames.append(fname06_006)
    gdas_fnames.append(fname12_006)

    # Check if files for all analysis times (00,06,12,18) are available
    gdas_check = []
    for gdas_fname in gdas_fnames:
        if exists(gdas_fname):
            gdas_check.append(gdas_fname)
    gdas_check = str(gdas_check)
    if gdas_check.find("00z") == -1 or gdas_check.find("06z") == -1 or gdas_check.find("12z") == -1 or gdas_check.find("18z") == -1:
        print("read_ocnbkg_ssh: NO FILES for " + exptpath)
        return "NO FILES"
    else:
        print("read_ocnbkg_ssh: FILES EXIST for " + exptpath)

    # Extract data
    cnt = 0.0
    for gdas_fname in gdas_fnames:
        if exists(gdas_fname):
            ds = xr.open_dataset(gdas_fname)
            if cnt == 0:
                sst = np.squeeze(ds['ave_ssh'][0, :, :])
            else:
                sst += np.squeeze(ds['ave_ssh'][0, :, :])
            cnt += 1
            ds.close()

    if cnt > 0:
        return sst.values / cnt
    else:
        return "NO FILES"


# =========================================================================
# Get ocean model SST background forecasts
#
def read_ocnbkg_sst(exptpath, yyyy, mm, dd):
    # Date before yyyymmdd
    # ...year
    byyyy = yyyy
    # ...month
    if int(dd) == 1:
        tbmm = int(mm) - 1
    else:
        tbmm = int(mm)
    if tbmm < 10:
        bmm = "0" + str(tbmm)
    else:
        bmm = str(tbmm)
    # ...day
    if bmm == mm:
        tbdd = int(dd) - 1
        if tbdd < 10:
            bdd = "0" + str(tbdd)
        else:
            bdd = str(tbdd)
    else:
        if bmm == "04" or bmm == "06" or bmm == "09" or bmm == "11":
            bdd = "30"
        elif bmm == "02":
            if int(yyyy) % 4 == 0:
                bdd = "29"
            else:
                bdd = "28"
        else:
            bdd = "31"

    byyyymmdd = byyyy + bmm + bdd
    yyyymmdd = yyyy + mm + dd

    # Get input path
    if exptpath.find("marine_") != -1 or exptpath.find("cp4") != -1:
        modelstr = "model"
    else:
        modelstr = "model_data"

    # Compute daily average centered around 12z
    # ... Current approach is to only use f006 background files
    if exptpath.find("role") != -1 or exptpath.find("marine_") != -1 or exptpath.find("S2Sda") != -1:
        bprefix_path = exptpath + "/" + byyyymmdd
        prefix_path = exptpath + "/" + yyyymmdd

        bfname18_006 = bprefix_path + "18/gdas." + byyyymmdd + "/18/" + modelstr + "/ocean/history/gdas.ocean.t18z.inst.f006.nc"
        fname00_006 = prefix_path + "00/gdas." + yyyymmdd + "/00/" + modelstr + "/ocean/history/gdas.ocean.t00z.inst.f006.nc"
        fname06_006 = prefix_path + "06/gdas." + yyyymmdd + "/06/" + modelstr + "/ocean/history/gdas.ocean.t06z.inst.f006.nc"
        fname12_006 = prefix_path + "12/gdas." + yyyymmdd + "/12/" + modelstr + "/ocean/history/gdas.ocean.t12z.inst.f006.nc"
    elif exptpath.find("cp4") != -1:
        bfname18_006 = exptpath + "/gdas." + byyyymmdd + "/18/" + modelstr + "/ocean/history/gdas.ocean.t18z.inst.f006.nc"
        fname00_006 = exptpath + "/gdas." + yyyymmdd + "/00/" + modelstr + "/ocean/history/gdas.ocean.t00z.inst.f006.nc"
        fname06_006 = exptpath + "/gdas." + yyyymmdd + "/06/" + modelstr + "/ocean/history/gdas.ocean.t06z.inst.f006.nc"
        fname12_006 = exptpath + "/gdas." + yyyymmdd + "/12/" + modelstr + "/ocean/history/gdas.ocean.t12z.inst.f006.nc"
    else:
        bfname18_006 = exptpath + "/gdas." + byyyymmdd + "/18/" + modelstr + "/ocean/history/gdas.t18z.ocnf006.nc"
        fname00_006 = exptpath + "/gdas." + yyyymmdd + "/00/" + modelstr + "/ocean/history/gdas.t00z.ocnf006.nc"
        fname06_006 = exptpath + "/gdas." + yyyymmdd + "/06/" + modelstr + "/ocean/history/gdas.t06z.ocnf006.nc"
        fname12_006 = exptpath + "/gdas." + yyyymmdd + "/12/" + modelstr + "/ocean/history/gdas.t12z.ocnf006.nc"

    # Use only f006 bkg files
    gdas_fnames = [bfname18_006]
    gdas_fnames.append(fname00_006)
    gdas_fnames.append(fname06_006)
    gdas_fnames.append(fname12_006)

    # Check if files for all analysis times (00,06,12,18) are available
    gdas_check = []
    for gdas_fname in gdas_fnames:
        if exists(gdas_fname):
            print("read_ocnbkg_sst: FILE EXISTS = " + str(gdas_fname))
            gdas_check.append(gdas_fname)
        else:
            print("read_ocnbkg_sst: FILE DOES NOT EXIST = " + str(gdas_fname))
    gdas_check = str(gdas_check)
    if gdas_check.find("00z") == -1 or gdas_check.find("06z") == -1 or gdas_check.find("12z") == -1 or gdas_check.find("18z") == -1:
        print("read_ocnbkg_sst: NO FILES for " + exptpath)
        return "NO FILES"
    else:
        print("read_ocnbkg_sst: FILES EXIST for " + exptpath)

    # Extract data
    cnt = 0.0
    for gdas_fname in gdas_fnames:
        if exists(gdas_fname):
            ds = xr.open_dataset(gdas_fname)
            if cnt == 0:
                # sst = np.squeeze(ds['Temp'][0, 0, :, :])      # model level 0 (top)
                sst = np.squeeze(ds['Temp'][0, 1, :, :])       # model level 1 (level below top)
            else:
                # sst += np.squeeze(ds['Temp'][0, 0, :, :])     # model level 0 (top)
                sst += np.squeeze(ds['Temp'][0, 1, :, :])      # model level 1 (level below top)
            cnt += 1
            ds.close()

    if cnt > 0:
        return sst.values / cnt
    else:
        return "NO FILES"

# =========================================================================
