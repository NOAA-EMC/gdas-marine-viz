###########################################################################
# read_data module
###########################################################################
#
# Import python modules
#

import xarray as xr
import os
from os.path import exists
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import datetime as dt
import sys
from netCDF4 import Dataset

###########################################################################
#
# FUNCTIONS
#

#------------------------------------------------
# Get ocean model grid (lats, lons)
def get_ocngrid():

    # Get current working directory (CWD)
    cwd = os.getcwd()

    # Open the NetCDF file using xarray
    grid_fname = os.path.join(str(cwd)+'/soca_gridspec.nc')

    ds = xr.open_dataset(grid_fname)

    lat = np.squeeze(ds['lat'][:])
    lon = np.squeeze(ds['lon'][:])

    ds.close()

    return lat.values, lon.values

#===============================================================================================
# Read ocean mask file
#
def read_ocean_mask(ocean_choice):

    ocean = ocean_choice.lower()

    dpath = "/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/data/"
    fname = "oceanmask_global_0.25deg.20221025.xesmf_nearest_s2d.nc"

    ds = xr.open_dataset(dpath+fname)

    lat = np.squeeze(ds['lat'][:])      # 1D array
    lon = np.squeeze(ds['lon'][:])      # 1D array

    # Get ocean mask
    sealand_mask = np.squeeze(ds['seamask'][:])
    ocean_mask = np.squeeze(ds['open_ocean'][:])

    if ocean.find("arctic")!=-1:
        iocean = 4
    elif ocean.find("atlantic")!=-1:
        iocean = 1
    elif ocean.find("indian")!=-1:
        iocean = 3
    elif ocean.find("pacific")!=-1:
        iocean = 2
    elif ocean.find("southern")!=-1:
        iocean = 5
    else:
        iocean = -1

    if iocean>0:
        mask_out = np.where(ocean_mask==iocean, ocean_mask, np.nan)
    else:
        mask_out = ocean_mask

    # Close the NetCDF file
    ds.close()

    return mask_out, lat, lon

#===============================================================================================
# Get OSTIA level-4 analysis on ocean model grid: SST
#
#   Dataset info: https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description
#
def read_ostia(yyyymmdd):

    dpath = '/scratch1/NCEPDEV/da/common/validation/ostia2mom6_xesmf/'
    fname = 'ostia_global_0.25deg.'+yyyymmdd+'_12z.xesmf.nc'
    ostia_fname = dpath+fname
    print("read_ostia: ostia_fname = "+str(ostia_fname))

    ds = xr.open_dataset(ostia_fname)

    # Get the SST data variable
    if ostia_fname.find("Shastri")!=-1:
      sst = np.squeeze(ds['analysed_sst'][:])           # np.squeeze: removes all axes of length 1 from array
    else:
      sst = np.squeeze(ds['analysed_sst'][:])
    mask = np.squeeze(ds['mask'][:])

    print("read_ostia: min/max sst = "+str(np.min(sst.values))+" "+str(np.max(sst.values)))
    print("read_ostia: min/max no nan sst = "+str(np.nanmin(sst.values))+" "+str(np.nanmax(sst.values)))

    # Close the NetCDF file
    ds.close()

    return sst.values, mask.values

#===============================================================================================
# Get OSTIA level-4 analysis on ocean model grid: SEAICE
#
#   Dataset info: https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description
#
def read_ostia_ice(yyyymmdd):

    dpath = '/scratch1/NCEPDEV/da/common/validation/ostia2mom6_xesmf/'
    fname = 'ostia_global_0.25deg.'+yyyymmdd+'_12z.xesmf.nc'
    ostia_fname = dpath+fname

    ds = xr.open_dataset(ostia_fname)

    # Get the SST data variable
    icec = np.squeeze(ds['sea_ice_fraction'][:])           # np.squeeze: removes all axes of length 1 from array
    mask = np.squeeze(ds['mask'][:])

    # Close the NetCDF file
    ds.close()

    return icec.values, mask.values

#===============================================================================================
# Get COPERNICUS level-4 analysis on ocean model grid: SSH (altimetry)
#
#   Dataset info: https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/description
#
def read_copernicus(yyyymmdd):

    fname = '/scratch1/NCEPDEV/da/common/validation/adtl4coper2mom6/'+yyyymmdd+'_L4adt_2_global_0.25deg.nc'
    
    ds = xr.open_dataset(fname)

    # Get the SSH data variable
    ssh = np.squeeze(ds['adt'][:])          # np.squeeze: removes all axes of length 1 from array

    # Close the NetCDF file
    ds.close()

    return ssh.values

#===============================================================================================
# Get ORAS5 (ECMWF) analysis on ocean model grid: SST or SSH
#
def read_ecm(yyyy, mm, dd, var):

    ecm_fnames = glob.glob('/scratch1/NCEPDEV/da/common/validation/oras5replay/ocn_'+str(yyyy)+'_'+str(mm)+'_'+str(dd)+'_*.nc')

    cnt=0
    for iname in ecm_fnames:
     if exists(iname)==True:
      ds = xr.open_dataset(iname)

      if cnt == 0:
        if var.lower()=='sst':
          sst = np.squeeze(ds['temp'][0,0,:,:])
        elif var.lower()=='ssh':
          sst = np.squeeze(ds['SSH'][0,:,:])
      else:
        if var.lower()=='sst':
          sst += np.squeeze(ds['temp'][0,0,:,:])
        elif var.lower()=='ssh':
          sst += np.squeeze(ds['SSH'][0,:,:])

      cnt+=1

      ds.close()
     else:
      sst = np.nan

    return sst.values/cnt

#===============================================================================================
# Get ocean model SEAICE background forecasts
#
def read_icebkg(exptpath, yyyy, mm, dd):

    # Date before yyyymmdd
    # ...year
    yyyyB = yyyy
    # ...month
    if int(dd)==1:
      tmmB = int(mm) - 1
    else:
      tmmB = int(mm)
    if tmmB<10:
      mmB = "0"+str(tmmB)
    else:
      mmB = str(tmmB)
    # ...day
    if mmB==mm:
      tddB = int(dd) - 1
      if tddB<10:
        ddB = "0"+str(tddB)
      else:
        ddB = str(tddB)
    else:
      if mmB=="04" or mmB=="06" or mmB=="09" or mmB=="11":
        ddB = "30"
      elif mmB=="02":
        if int(yyyy)%4==0:
          ddB = "29"
        else:
          ddB = "28"
      else:
        ddB = "31"

    # Get input paths
    if exptpath.find("marine_")!=-1 or exptpath.find("cp4")!=-1:
        modelstr = "model"
    else:
        modelstr = "model_data"

    # Compute daily average centered around 12z
    # ... Current approach is to only use f006 background files
    if exptpath.find("role")!=-1 or exptpath.find("marine_")!=-1 or exptpath.find("S2Sda")!=-1:
      fname18B_006 = exptpath+"/"+yyyyB+mmB+ddB+"18/gdas."+yyyyB+mmB+ddB+"/18/"+modelstr+"/ice/history/gdas.ice.t18z.inst.f006.nc"
      #fname00_003 = exptpath+"/"+yyyy+mm+dd+"00/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ice/history/gdas.ice.t00z.inst.f003.nc"
      fname00_006 = exptpath+"/"+yyyy+mm+dd+"00/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ice/history/gdas.ice.t00z.inst.f006.nc"
      #fname00_009 = exptpath+"/"+yyyy+mm+dd+"00/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ice/history/gdas.ice.t00z.inst.f009.nc"
      #fname06_003 = exptpath+"/"+yyyy+mm+dd+"06/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ice/history/gdas.ice.t06z.inst.f003.nc"
      fname06_006 = exptpath+"/"+yyyy+mm+dd+"06/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ice/history/gdas.ice.t06z.inst.f006.nc"
      #fname06_009 = exptpath+"/"+yyyy+mm+dd+"06/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ice/history/gdas.ice.t06z.inst.f009.nc"
      #fname12_003 = exptpath+"/"+yyyy+mm+dd+"12/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ice/history/gdas.ice.t12z.inst.f003.nc"
      fname12_006 = exptpath+"/"+yyyy+mm+dd+"12/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ice/history/gdas.ice.t12z.inst.f006.nc"
      #fname12_009 = exptpath+"/"+yyyy+mm+dd+"12/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ice/history/gdas.ice.t12z.inst.f009.nc"
      #fname18_003 = exptpath+"/"+yyyy+mm+dd+"18/gdas."+yyyy+mm+dd+"/18/"+modelstr+"/ice/history/gdas.ice.t18z.inst.f003.nc"
    else:
      fname18B_006 = exptpath+"/gdas."+yyyyB+mmB+ddB+"/18/"+modelstr+"/ice/history/gdas.ice.t18z.inst.f006.nc"
      #fname00_003 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ice/history/gdas.ice.t00z.inst.f003.nc"
      fname00_006 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ice/history/gdas.ice.t00z.inst.f006.nc"
      #fname00_009 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ice/history/gdas.ice.t00z.inst.f009.nc"
      #fname06_003 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ice/history/gdas.ice.t06z.inst.f003.nc"
      fname06_006 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ice/history/gdas.ice.t06z.inst.f006.nc"
      #fname06_009 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ice/history/gdas.ice.t06z.inst.f009.nc"
      #fname12_003 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ice/history/gdas.ice.t12z.inst.f003.nc"
      fname12_006 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ice/history/gdas.ice.t12z.inst.f006.nc"
      #fname12_009 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ice/history/gdas.ice.t12z.inst.f009.nc"
      #fname18_003 = exptpath+"/gdas."+yyyy+mm+dd+"/18/"+modelstr+"/ice/history/gdas.ice.t18z.inst.f003.nc"

    # Use all bkg files
    #gdas_fnames = [fname00_003]
    #gdas_fnames.append(fname00_006)
    #gdas_fnames.append(fname00_009)
    #gdas_fnames.append(fname06_003)
    #gdas_fnames.append(fname06_006)
    #gdas_fnames.append(fname06_009)
    #gdas_fnames.append(fname12_003)
    #gdas_fnames.append(fname12_006)
    #gdas_fnames.append(fname12_009)
    #gdas_fnames.append(fname18_003)

    # Use only f006 bkg files
    gdas_fnames = [fname18B_006]
    gdas_fnames.append(fname00_006)
    gdas_fnames.append(fname06_006)
    gdas_fnames.append(fname12_006)

    # Check if files for all analysis times (00,06,12,18) are available
    gdas_check = []
    for gdas_fname in gdas_fnames:
      if exists(gdas_fname)==True: gdas_check.append(gdas_fname)
    gdas_check = str(gdas_check)
    if gdas_check.find("00z")==-1 or gdas_check.find("06z")==-1 or gdas_check.find("12z")==-1 or gdas_check.find("18z")==-1: 
      print("read_icebkg: NO FILES for "+exptpath)
      return "NO FILES"
    else:
      print("read_icebkg: FILES EXIST for "+exptpath)

    # Extract data
    cnt = 0.0
    for gdas_fname in gdas_fnames:
      if exists(gdas_fname)==True:
        ds = xr.open_dataset(gdas_fname)
        if cnt == 0:
            try:
                sst = np.squeeze(ds['aice_h'][0,:,:])
            except:
                sst = ds['TS_FOUND'][:,:] - 273.15
        else:
            try:
                sst += np.squeeze(ds['aice_h'][0,:,:])
            except:
                sst += ds['TS_FOUND'][:,:] - 273.15

        cnt +=1
        ds.close()

    if cnt>0:
      return sst.values/cnt
    else:
      return "NO FILES"

#===============================================================================================
# Get ocean model SSH background forecasts
#
def read_ocnbkg_ssh(exptpath, yyyy, mm, dd):

    # Date before yyyymmdd
    # ...year
    yyyyB = yyyy
    # ...month
    if int(dd)==1:
      tmmB = int(mm) - 1
    else:
      tmmB = int(mm)
    if tmmB<10:
      mmB = "0"+str(tmmB)
    else:
      mmB = str(tmmB)
    # ...day
    if mmB==mm:
      tddB = int(dd) - 1
      if tddB<10:
        ddB = "0"+str(tddB)
      else:
        ddB = str(tddB)
    else:
      if mmB=="04" or mmB=="06" or mmB=="09" or mmB=="11":
        ddB = "30"
      elif mmB=="02":
        if int(yyyy)%4==0:
          ddB = "29"
        else:
          ddB = "28"
      else:
        ddB = "31"

    if exptpath.find("marine_")!=-1 or exptpath.find("cp4")!=-1:
        modelstr = "model"
    else:
        modelstr = "model_data"

    # Compute daily average centered around 12z
    # ... Current approach is to only use f006 background files
    if exptpath.find("role")!=-1 or exptpath.find("marine_")!=-1 or exptpath.find("S2Sda")!=-1:
      fname18B_006 = exptpath+"/"+yyyyB+mmB+ddB+"18/gdas."+yyyyB+mmB+ddB+"/18/"+modelstr+"/ocean/history/gdas.ocean.t18z.inst.f006.nc"
      #fname00_003 = exptpath+"/"+yyyy+mm+dd+"00/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f003.nc"
      fname00_006 = exptpath+"/"+yyyy+mm+dd+"00/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f006.nc"
      #fname00_009 = exptpath+"/"+yyyy+mm+dd+"00/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f009.nc"
      #fname06_003 = exptpath+"/"+yyyy+mm+dd+"06/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f003.nc"
      fname06_006 = exptpath+"/"+yyyy+mm+dd+"06/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f006.nc"
      #fname06_009 = exptpath+"/"+yyyy+mm+dd+"06/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f009.nc"
      #fname12_003 = exptpath+"/"+yyyy+mm+dd+"12/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f003.nc"
      fname12_006 = exptpath+"/"+yyyy+mm+dd+"12/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f006.nc"
      #fname12_009 = exptpath+"/"+yyyy+mm+dd+"12/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f009.nc"
      #fname18_003 = exptpath+"/"+yyyy+mm+dd+"18/gdas."+yyyy+mm+dd+"/18/"+modelstr+"/ocean/history/gdas.ocean.t18z.inst.f003.nc"
    elif exptpath.find("cp4")!=-1:
      fname18B_006 = exptpath+"/gdas."+yyyyB+mmB+ddB+"/18/"+modelstr+"/ocean/history/gdas.ocean.t18z.inst.f006.nc"
      #fname00_003 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f003.nc"
      fname00_006 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f006.nc"
      #fname00_009 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f009.nc"
      #fname06_003 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f003.nc"
      fname06_006 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f006.nc"
      #fname06_009 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f009.nc"
      #fname12_003 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f003.nc"
      fname12_006 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f006.nc"
      #fname12_009 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f009.nc"
      #fname18_003 = exptpath+"/gdas."+yyyy+mm+dd+"/18/"+modelstr+"/ocean/history/gdas.ocean.t18z.inst.f003.nc"
    else:
      fname18B_006 = exptpath+"/gdas."+yyyyB+mmB+ddB+"/18/"+modelstr+"/ocean/history/gdas.t18z.ocnf006.nc"
      #fname00_003 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.t00z.ocnf003.nc"
      fname00_006 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.t00z.ocnf006.nc"
      #fname00_009 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.t00z.ocnf009.nc"
      #fname06_003 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.t06z.ocnf003.nc"
      fname06_006 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.t06z.ocnf006.nc"
      #fname06_009 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.t06z.ocnf009.nc"
      #fname12_003 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.t12z.ocnf003.nc"
      fname12_006 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.t12z.ocnf006.nc"
      #fname12_009 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.t12z.ocnf009.nc"
      #fname18_003 = exptpath+"/gdas."+yyyy+mm+dd+"/18/"+modelstr+"/ocean/history/gdas.t18z.ocnf003.nc"

    # Use all bkg files
    #gdas_fnames = [fname00_003]
    #gdas_fnames.append(fname00_006)
    #gdas_fnames.append(fname00_009)
    #gdas_fnames.append(fname06_003)
    #gdas_fnames.append(fname06_006)
    #gdas_fnames.append(fname06_009)
    #gdas_fnames.append(fname12_003)
    #gdas_fnames.append(fname12_006)
    #gdas_fnames.append(fname12_009)
    #gdas_fnames.append(fname18_003)

    # Use only f006 bkg files
    gdas_fnames = [fname18B_006]
    gdas_fnames.append(fname00_006)
    gdas_fnames.append(fname06_006)
    gdas_fnames.append(fname12_006)

    # Check if files for all analysis times (00,06,12,18) are available
    gdas_check = []
    for gdas_fname in gdas_fnames:
      if exists(gdas_fname)==True: gdas_check.append(gdas_fname)
    gdas_check = str(gdas_check)
    if gdas_check.find("00z")==-1 or gdas_check.find("06z")==-1 or gdas_check.find("12z")==-1 or gdas_check.find("18z")==-1: 
      print("read_ocnbkg_ssh: NO FILES for "+exptpath)
      return "NO FILES"
    else:
      print("read_ocnbkg_ssh: FILES EXIST for "+exptpath)

    # Extract data
    cnt = 0.0
    for gdas_fname in gdas_fnames:
      if exists(gdas_fname)==True:
        ds = xr.open_dataset(gdas_fname)
        if cnt == 0:
            try:
                sst = np.squeeze(ds['ave_ssh'][0,:,:])
            except:
                sst = ds['TS_FOUND'][:,:] - 273.15
        else:
            try:
                sst += np.squeeze(ds['ave_ssh'][0,:,:])
            except:
                sst += ds['TS_FOUND'][:,:] - 273.15

        cnt +=1
        ds.close()

    if cnt>0:
      return sst.values/cnt
    else:
      return "NO FILES"

#===============================================================================================
# Get ocean model SST background forecasts
#
def read_ocnbkg_sst(exptpath, yyyy, mm, dd):

    # Date before yyyymmdd
    # ...year
    yyyyB = yyyy
    # ...month
    if int(dd)==1:
      tmmB = int(mm) - 1
    else:
      tmmB = int(mm)
    if tmmB<10:
      mmB = "0"+str(tmmB)
    else:
      mmB = str(tmmB)
    # ...day
    if mmB==mm:
      tddB = int(dd) - 1
      if tddB<10:
        ddB = "0"+str(tddB)
      else:
        ddB = str(tddB)
    else:
      if mmB=="04" or mmB=="06" or mmB=="09" or mmB=="11":
        ddB = "30"
      elif mmB=="02":
        if int(yyyy)%4==0:
          ddB = "29"
        else:
          ddB = "28"
      else:
        ddB = "31"

    if exptpath.find("marine_")!=-1 or exptpath.find("cp4")!=-1:
        modelstr = "model"
    else:
        modelstr = "model_data"

    # Compute daily average centered around 12z
    # ... Current approach is to only use f006 background files
    if exptpath.find("role")!=-1 or exptpath.find("marine_")!=-1 or exptpath.find("S2Sda")!=-1:
      fname18B_006 = exptpath+"/"+yyyyB+mmB+ddB+"18/gdas."+yyyyB+mmB+ddB+"/18/"+modelstr+"/ocean/history/gdas.ocean.t18z.inst.f006.nc"
      #fname00_003 = exptpath+"/"+yyyy+mm+dd+"00/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f003.nc"
      fname00_006 = exptpath+"/"+yyyy+mm+dd+"00/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f006.nc"
      #fname00_009 = exptpath+"/"+yyyy+mm+dd+"00/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f009.nc"
      #fname06_003 = exptpath+"/"+yyyy+mm+dd+"06/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f003.nc"
      fname06_006 = exptpath+"/"+yyyy+mm+dd+"06/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f006.nc"
      #fname06_009 = exptpath+"/"+yyyy+mm+dd+"06/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f009.nc"
      #fname12_003 = exptpath+"/"+yyyy+mm+dd+"12/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f003.nc"
      fname12_006 = exptpath+"/"+yyyy+mm+dd+"12/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f006.nc"
      #fname12_009 = exptpath+"/"+yyyy+mm+dd+"12/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f009.nc"
      #fname18_003 = exptpath+"/"+yyyy+mm+dd+"18/gdas."+yyyy+mm+dd+"/18/"+modelstr+"/ocean/history/gdas.ocean.t18z.inst.f003.nc"
    elif exptpath.find("cp4")!=-1:
      fname18B_006 = exptpath+"/gdas."+yyyyB+mmB+ddB+"/18/"+modelstr+"/ocean/history/gdas.ocean.t18z.inst.f006.nc"
      #fname00_003 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f003.nc"
      fname00_006 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f006.nc"
      #fname00_009 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.ocean.t00z.inst.f009.nc"
      #fname06_003 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f003.nc"
      fname06_006 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f006.nc"
      #fname06_009 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.ocean.t06z.inst.f009.nc"
      #fname12_003 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f003.nc"
      fname12_006 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f006.nc"
      #fname12_009 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.ocean.t12z.inst.f009.nc"
      #fname18_003 = exptpath+"/gdas."+yyyy+mm+dd+"/18/"+modelstr+"/ocean/history/gdas.ocean.t18z.inst.f003.nc"
    else:
      fname18B_006 = exptpath+"/gdas."+yyyyB+mmB+ddB+"/18/"+modelstr+"/ocean/history/gdas.t18z.ocnf006.nc"
      #fname00_003 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.t00z.ocnf003.nc"
      fname00_006 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.t00z.ocnf006.nc"
      #fname00_009 = exptpath+"/gdas."+yyyy+mm+dd+"/00/"+modelstr+"/ocean/history/gdas.t00z.ocnf009.nc"
      #fname06_003 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.t06z.ocnf003.nc"
      fname06_006 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.t06z.ocnf006.nc"
      #fname06_009 = exptpath+"/gdas."+yyyy+mm+dd+"/06/"+modelstr+"/ocean/history/gdas.t06z.ocnf009.nc"
      #fname12_003 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.t12z.ocnf003.nc"
      fname12_006 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.t12z.ocnf006.nc"
      #fname12_009 = exptpath+"/gdas."+yyyy+mm+dd+"/12/"+modelstr+"/ocean/history/gdas.t12z.ocnf009.nc"
      #fname18_003 = exptpath+"/gdas."+yyyy+mm+dd+"/18/"+modelstr+"/ocean/history/gdas.t18z.ocnf003.nc"

    # Use all bkg files
    #gdas_fnames = [fname00_003]
    #gdas_fnames.append(fname00_006)
    #gdas_fnames.append(fname00_009)
    #gdas_fnames.append(fname06_003)
    #gdas_fnames.append(fname06_006)
    #gdas_fnames.append(fname06_009)
    #gdas_fnames.append(fname12_003)
    #gdas_fnames.append(fname12_006)
    #gdas_fnames.append(fname12_009)
    #gdas_fnames.append(fname18_003)

    # Use only f006 bkg files
    gdas_fnames = [fname18B_006]
    gdas_fnames.append(fname00_006)
    gdas_fnames.append(fname06_006)
    gdas_fnames.append(fname12_006)

    # Check if files for all analysis times (00,06,12,18) are available
    gdas_check = []
    for gdas_fname in gdas_fnames:
      if exists(gdas_fname)==True: gdas_check.append(gdas_fname)
    gdas_check = str(gdas_check)
    if gdas_check.find("00z")==-1 or gdas_check.find("06z")==-1 or gdas_check.find("12z")==-1 or gdas_check.find("18z")==-1: 
      print("read_ocnbkg_sst: NO FILES for "+exptpath)
      return "NO FILES"
    else:
      print("read_ocnbkg_sst: FILES EXIST for "+exptpath)

    # Extract data
    cnt = 0.0
    for gdas_fname in gdas_fnames:
      if exists(gdas_fname)==True:
        ds = xr.open_dataset(gdas_fname)
        if cnt == 0:
            try:
                #sst = np.squeeze(ds['Temp'][0,0,:,:])      # model level 0 (top)
                sst = np.squeeze(ds['Temp'][0,1,:,:])       # model level 1 (level below top)
            except:
                sst = ds['TS_FOUND'][:,:] - 273.15
        else:
            try:
                #sst += np.squeeze(ds['Temp'][0,0,:,:])     # model level 0 (top)
                sst += np.squeeze(ds['Temp'][0,1,:,:])      # model level 1 (level below top)
            except:
                sst += ds['TS_FOUND'][:,:] - 273.15

        cnt +=1
        ds.close()

    if cnt>0:
      return sst.values/cnt
    else:
      return "NO FILES"

#===============================================================================================
