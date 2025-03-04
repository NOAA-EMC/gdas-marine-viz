###################################################################################################################
###################################################################################################################
# PLOT Time Series of Multiple Runs/Observation Datasets
###################################################################################################################
###################################################################################################################
#
# Output: Figure 
#
# How to run: python NameOfScript.py yyyymmdd yyyymmdd
#	First yyyymmdd is START DATE
#	Second yyyymmdd is END DATE
#
###################################################################################################################
# Import python modules
###################################################################################################################
print("***** Import Python Modules *****")

import xarray as xr
import os
import numpy as np
import glob
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pickle
import datetime as dt
import sys
from scipy.interpolate import interpn
from scipy.interpolate import RectBivariateSpline

from read_data import read_ostia_ice
from read_data import read_ecm
from read_data import read_icebkg
from read_data import get_ocngrid

###################################################################################################################
print("========== BEGIN MAIN PROGRAM ==========")

#=============================================
# Set global parameters

fill = -999.0

inpath  = "/scratch2/NCEPDEV/ocean/Guillaume.Vernieres/runs/low-res/"
outpath = "/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/tools/stats/indiv_scripts/output/"

#=============================================
# Read raw user input from command line

yyyymmddS = sys.argv[1]		# Start date: year month day
yyyymmddE = sys.argv[2]		# End date: year month day
outpath   = sys.argv[3]
var       = sys.argv[4]
reference = sys.argv[5]

#=============================================

# start date
yyyyS = yyyymmddS[0:4]
mmS   = yyyymmddS[4:6]
ddS   = yyyymmddS[6:8]
# end date
yyyyE = yyyymmddE[0:4]
mmE   = yyyymmddE[4:6]
ddE   = yyyymmddE[6:8]

start_date = dt.date(int(yyyyS), int(mmS), int(ddS))
end_date = dt.date(int(yyyyE), int(mmE), int(ddE))
#print("start_date = "+str(start_date))

#---------------------------------------------------------------
# Get array of dates for time series plot

delta = dt.timedelta(days=1)
dates = []
while start_date <= end_date:
    dates.append(start_date)
    start_date += delta

#******************************************************************************************************************
#******************************************************************************************************************
# PLOTS
#******************************************************************************************************************
#******************************************************************************************************************

#-------------------------------------------
# Get ocean lat/lon arrays (2D each)
lat, lon = get_ocngrid()

# Read soca_gridspec file for proper land/sea masking and apply to OSTIA
cwd = os.getcwd()       # get current working directory (CWD)
gridfile = str(cwd)+"/soca_gridspec.nc"
ds       = xr.open_dataset(gridfile)
ufs_mask = np.squeeze(ds['mask2d'][:])          # land/sea mask
ufs_area = np.squeeze(ds['area'][:])            # grid cell area
del gridfile, ds

# Sea ice area extent minimum
area_min = 0.15

#-------------------------------------------
# Total month, day, hour arrays for LOOP
mmARR = [ '01','02','03','04','05','06','07','08','09','10','11','12' ]
ddARR = [ "01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31" ]
hours = [ '00','06','12','18' ]

#-------------------------------------------------------------------------
# LOOP
#-------------------------------------------------------------------------

maeNP_atmatmB = []
maeNP_s2satmB = []
maeNP_C03B = []
maeNP_MCB = []
maeNP_v1B = []
maeNP_v2B = []
maeNP_cp4B = []

maeSP_atmatmB = []
maeSP_s2satmB = []
maeSP_C03B = []
maeSP_MCB = []
maeSP_v1B = []
maeSP_v2B = []
maeSP_cp4B = []

rmsdNP_atmatmB = []
rmsdNP_s2satmB = []
rmsdNP_C03B = []
rmsdNP_MCB = []
rmsdNP_v1B = []
rmsdNP_v2B = []
rmsdNP_cp4B = []

rmsdSP_atmatmB = []
rmsdSP_s2satmB = []
rmsdSP_C03B = []
rmsdSP_MCB = []
rmsdSP_v1B = []
rmsdSP_v2B = []
rmsdSP_cp4B = []

mae_extentNP_obs = []
mae_extentNP_atmatmB = []
mae_extentNP_s2satmB = []
mae_extentNP_C03B = []
mae_extentNP_MCB = []
mae_extentNP_v1B = []
mae_extentNP_v2B = []
mae_extentNP_cp4B = []

mae_extentSP_obs = []
mae_extentSP_atmatmB = []
mae_extentSP_s2satmB = []
mae_extentSP_C03B = []
mae_extentSP_MCB = []
mae_extentSP_v1B = []
mae_extentSP_v2B = []
mae_extentSP_cp4B = []

extentNP_obs = []
extentNP_atmatmB = []
extentNP_s2satmB = []
extentNP_C03B = []
extentNP_MCB = []
extentNP_v1B = []
extentNP_v2B = []
extentNP_cp4B = []

extentSP_obs = []
extentSP_atmatmB = []
extentSP_s2satmB = []
extentSP_C03B = []
extentSP_MCB = []
extentSP_v1B = []
extentSP_v2B = []
extentSP_cp4B = []

yyyy = yyyyS
iyy  = int(yyyy)
iyyE = int(yyyyE)
while iyy <= int(yyyyE):
  if yyyyS==yyyyE:
    Simm = mmARR.index(mmS)
    Eimm = mmARR.index(mmE)
  elif yyyyS!=yyyyE:
    if iyy==int(yyyyS):
      Simm = mmARR.index(mmS)
      Eimm = 12-1
    elif iyy>int(yyyyS) and iyy<iyyE:
      Simm = 0
      Eimm = 12-1
    elif iyy==iyyE:
      Simm = 0
      Eimm = mmARR.index(mmE)

  imm=Simm
  while imm <= Eimm:
    mm = mmARR[imm]
    if mm=="02":
      if int(yyyy)%4==0:
        # leap year
        nDD = 29
      else:
        nDD = 28
    elif mm=="04" or mm=="06" or mm=="09" or mm=="11":
      nDD = 30
    else:
      nDD = 31

    if yyyyS==yyyyE:
      if mmS==mmE:
        Sidd = ddARR.index(ddS)
        Eidd = ddARR.index(ddE)
      else:
        if mm==mmS:
          Sidd = ddARR.index(ddS)
          Eidd = nDD-1
        elif mm==mmE:
          Sidd = 0
          Eidd = ddARR.index(ddE)
        else:
          Sidd = 0
          Eidd = nDD-1
    elif yyyyS!=yyyyE:
      if yyyy==yyyyS and mm==mmS:
        Sidd = ddARR.index(ddS)
        Eidd = nDD-1
      elif iyy==int(yyyyE) and mm==mmE:
        Sidd = 0
        Eidd = ddARR.index(ddE)
      else:
        Sidd = 0
        Eidd = nDD-1

    idd=Sidd
    while idd <= Eidd:
        dd = ddARR[idd]
        print("===== DATE: "+str(yyyy)+"-"+str(mm)+"-"+str(dd)+" =====")

        dd_str = str(dd).zfill(2)		# add zeroes to beginning of string

        #if int(mm)>=7 or int(mm)<=12:
        #------------------------------------------
        # OBSERVATION DATASET
        print("... OBS ...")
        if reference.upper()=="OSTIA":
          # Get OSTIA sea ice level-4 analysis product
          obs, obs_mask = read_ostia_ice(yyyy+mm+dd_str)      # extracts sea ice area fraction

        # Combine UFS mask and OBS mask
        #   mask==1 = where sea ice exists in both ufs_mask and obs_mask, and land does not exist in either
        mask = np.where((obs_mask==9) & (ufs_mask==1), 1, 0)      # obs_mask==9 = sea ice | ufs_mask==0 = land ; ufs_mask==1 = not land
        del obs_mask

        # Calc sea ice area extent for obs
        t0 = obs
        #   Arctic
        print("Arctic idx: "+str(dt.datetime.now()))
        s0 = np.where( (lat>0.0) & (mask==1) & (t0>=area_min), t0, 0 )
        obsNP = np.sum(s0)
        del s0
        #   Antarctic
        print("Antarctic idx: "+str(dt.datetime.now()))
        s0 = np.where( (lat<0.0) & (mask==1) & (t0>=area_min), t0, 0 )
        obsSP = np.sum(s0)
        del s0
        del t0

        # Append extent to total arrays
        extentNP_obs.append(obsNP)
        extentSP_obs.append(obsSP)

        #------------------------------------------

        print("----- EXPERIMENTS: ")#+str(dt.datetime.now()))
  
        #------------------------------------------
        # EXPT: S2Smodel_ATMda (GFSv17 prototype)

        print("... S2Smodel_ATMda ...")
        exptpath = "/scratch1/NCEPDEV/climate/role.ufscpara/ModelOutput/CycledTests/HR_3_5_s2s_model_atm_da/"

                #``````````````````````````````````
                # backgrounds
        bkg = read_icebkg(exptpath, yyyy, mm, dd_str)
        if bkg=="NO FILES":
          maeNP_s2satmB.append(np.nan)
          rmsdNP_s2satmB.append(np.nan)
          maeSP_s2satmB.append(np.nan)
          rmsdSP_s2satmB.append(np.nan)
          mae_extentNP_s2satmB.append(np.nan)
          mae_extentSP_s2satmB.append(np.nan)
          extentNP_s2satmB.append(np.nan)
          extentSP_s2satmB.append(np.nan)
        else:

          # Calculate sea ice extent
          t0 = bkg
          #     Arctic
          s0 = np.where( (lat>0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgNP = np.sum(s0)
          del s0
          #   Antarctic
          s0 = np.where( (lat<0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgSP = np.sum(s0)
          del s0
          del t0

          diffNP = obsNP - bkgNP
          diffSP = obsSP - bkgSP

          mae_extentNP_s2satmB.append(np.mean(np.abs(diffNP)))
          mae_extentSP_s2satmB.append(np.mean(np.abs(diffSP)))

          extentNP_s2satmB.append(bkgNP)
          extentSP_s2satmB.append(bkgSP)
          del diffNP, diffSP
          del bkgNP, bkgSP

          # Daily difference stats (per grid cell)
          diff = obs - bkg                # same shape: (1080, 1440)
          #     Arctic
          I = np.where( (lat>0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeNP_s2satmB.append(np.mean(np.abs(diff[I])))
          rmsdNP_s2satmB.append(np.mean((diff[I])**2))
          del I
          #     Antarctic
          I = np.where( (lat<0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeSP_s2satmB.append(np.mean(np.abs(diff[I])))
          rmsdSP_s2satmB.append(np.mean((diff[I])**2))
          del I
          del bkg, diff
        del exptpath

        #------------------------------------------
        # EXPT: S2Smodel_S2Sda_C03 (GFSv17 prototype)

        print("... S2Smodel_S2Sda ...")
        exptpath = "/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/data/expts/S2Smodel_S2Sda_C03/"

                #``````````````````````````````````
                # backgrounds
        bkg = read_icebkg(exptpath, yyyy, mm, dd_str)
        if bkg=="NO FILES":
          maeNP_C03B.append(np.nan)
          rmsdNP_C03B.append(np.nan)
          maeSP_C03B.append(np.nan)
          rmsdSP_C03B.append(np.nan)
          mae_extentNP_C03B.append(np.nan)
          mae_extentSP_C03B.append(np.nan)
          extentNP_C03B.append(np.nan)
          extentSP_C03B.append(np.nan)
        else:

          # Calculate sea ice extent
          t0 = bkg
          #     Arctic
          s0 = np.where( (lat>0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgNP = np.sum(s0)
          del s0
          #   Antarctic
          s0 = np.where( (lat<0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgSP = np.sum(s0)
          del s0
          del t0

          diffNP = obsNP - bkgNP
          diffSP = obsSP - bkgSP

          mae_extentNP_C03B.append(np.mean(np.abs(diffNP)))
          mae_extentSP_C03B.append(np.mean(np.abs(diffSP)))

          extentNP_C03B.append(bkgNP)
          extentSP_C03B.append(bkgSP)
          del diffNP, diffSP
          del bkgNP, bkgSP

          # Daily difference stats (per grid cell)
          diff = obs - bkg                # same shape: (1080, 1440)
                # Arctic
          I = np.where( (lat>0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeNP_C03B.append(np.mean(np.abs(diff[I])))
          rmsdNP_C03B.append(np.mean((diff[I])**2))
          del I
                # Antarctic
          I = np.where( (lat<0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeSP_C03B.append(np.mean(np.abs(diff[I])))
          rmsdSP_C03B.append(np.mean((diff[I])**2))
          del I
          del bkg, diff
        del exptpath

        #------------------------------------------
        # EXPT: marine_test_v1 (GFSv17 prototype)

        print("... marine_test_v1 ...")
        exptpath = "/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/data/expts/marine_test_112024_v1/"

                #``````````````````````````````````
                # backgrounds
        bkg = read_icebkg(exptpath, yyyy, mm, dd_str)
        if bkg=="NO FILES":
          maeNP_v1B.append(np.nan)
          rmsdNP_v1B.append(np.nan)
          maeSP_v1B.append(np.nan)
          rmsdSP_v1B.append(np.nan)
          mae_extentNP_v1B.append(np.nan)
          mae_extentSP_v1B.append(np.nan)
          extentNP_v1B.append(np.nan)
          extentSP_v1B.append(np.nan)
        else:

          # Calculate sea ice extent
          t0 = bkg
          #     Arctic
          s0 = np.where( (lat>0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgNP = np.sum(s0)
          del s0
          #   Antarctic
          s0 = np.where( (lat<0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgSP = np.sum(s0)
          del s0
          del t0

          diffNP = obsNP - bkgNP
          diffSP = obsSP - bkgSP

          mae_extentNP_v1B.append(np.mean(np.abs(diffNP)))
          mae_extentSP_v1B.append(np.mean(np.abs(diffSP)))

          extentNP_v1B.append(bkgNP)
          extentSP_v1B.append(bkgSP)
          del diffNP, diffSP
          del bkgNP, bkgSP

          # Daily difference stats (per grid cell)
          diff = obs - bkg                # same shape: (1080, 1440)
                # Arctic
          I = np.where( (lat>0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeNP_v1B.append(np.mean(np.abs(diff[I])))
          rmsdNP_v1B.append(np.mean((diff[I])**2))
          del I
                # Antarctic
          I = np.where( (lat<0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeSP_v1B.append(np.mean(np.abs(diff[I])))
          rmsdSP_v1B.append(np.mean((diff[I])**2))
          del I
          del bkg, diff
        del exptpath

        #------------------------------------------
        # EXPT: marine_test_v2 (GFSv17 prototype)

        print("... marine_test_v2 ...")
        exptpath = "/scratch1/NCEPDEV/global/John.Steffen/hpss_arch/marine_test_112024_v2/"

                #``````````````````````````````````
                # backgrounds
        bkg = read_icebkg(exptpath, yyyy, mm, dd_str)
        if bkg=="NO FILES":
          maeNP_v2B.append(np.nan)
          rmsdNP_v2B.append(np.nan)
          maeSP_v2B.append(np.nan)
          rmsdSP_v2B.append(np.nan)
          mae_extentNP_v2B.append(np.nan)
          mae_extentSP_v2B.append(np.nan)
          extentNP_v2B.append(np.nan)
          extentSP_v2B.append(np.nan)
        else:

          # Calculate sea ice extent
          t0 = bkg
          #     Arctic
          s0 = np.where( (lat>0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgNP = np.sum(s0)
          del s0
          #   Antarctic
          s0 = np.where( (lat<0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgSP = np.sum(s0)
          del s0
          del t0

          diffNP = obsNP - bkgNP
          diffSP = obsSP - bkgSP

          mae_extentNP_v2B.append(np.mean(np.abs(diffNP)))
          mae_extentSP_v2B.append(np.mean(np.abs(diffSP)))

          extentNP_v2B.append(bkgNP)
          extentSP_v2B.append(bkgSP)
          del diffNP, diffSP
          del bkgNP, bkgSP

          # Daily difference stats (per grid cell)
          diff = obs - bkg                # same shape: (1080, 1440)
                # Arctic
          I = np.where( (lat>0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeNP_v2B.append(np.mean(np.abs(diff[I])))
          rmsdNP_v2B.append(np.mean((diff[I])**2))
          del I
                # Antarctic
          I = np.where( (lat<0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeSP_v2B.append(np.mean(np.abs(diff[I])))
          rmsdSP_v2B.append(np.mean((diff[I])**2))
          del I
          del bkg, diff
        del exptpath

	#------------------------------------------
        # EXPT: marine_candidate (4th GFSv17 prototype)

        print("... marince_candidate ...")
        exptpath = "/scratch1/NCEPDEV/global/John.Steffen/hpss_arch/marine_candidate_092024/"

                #``````````````````````````````````
                # backgrounds
        bkg = read_icebkg(exptpath, yyyy, mm, dd_str)
        if bkg=="NO FILES":
          maeNP_MCB.append(np.nan)
          rmsdNP_MCB.append(np.nan)
          maeSP_MCB.append(np.nan)
          rmsdSP_MCB.append(np.nan)
          mae_extentNP_MCB.append(np.nan)
          mae_extentSP_MCB.append(np.nan)
          extentNP_MCB.append(np.nan)
          extentSP_MCB.append(np.nan)
        else:

          # Calculate sea ice extent
          t0 = bkg
          #     Arctic
          s0 = np.where( (lat>0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgNP = np.sum(s0)
          del s0
          #   Antarctic
          s0 = np.where( (lat<0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgSP = np.sum(s0)
          del s0
          del t0

          diffNP = obsNP - bkgNP
          diffSP = obsSP - bkgSP

          mae_extentNP_MCB.append(np.mean(np.abs(diffNP)))
          mae_extentSP_MCB.append(np.mean(np.abs(diffSP)))

          extentNP_MCB.append(bkgNP)
          extentSP_MCB.append(bkgSP)
          del diffNP, diffSP
          del bkgNP, bkgSP

          # Daily difference stats (per grid cell)
          diff = obs - bkg                # same shape: (1080, 1440)
                # Arctic
          I = np.where( (lat>0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeNP_MCB.append(np.mean(np.abs(diff[I])))
          rmsdNP_MCB.append(np.mean((diff[I])**2))
          del I
                # Antarctic
          I = np.where( (lat<0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeSP_MCB.append(np.mean(np.abs(diff[I])))
          rmsdSP_MCB.append(np.mean((diff[I])**2))
          del I
          del bkg, diff
        del exptpath

        #------------------------------------------
        # EXPT: cp4.01 run (GFSv17 prototype)

        print("... cp4.01 ...")
        exptpath = "/scratch1/NCEPDEV/global/John.Steffen/hpss_arch/cp4.01-JS/COMROOT/cp4.01-JS/"

                #``````````````````````````````````
                # backgrounds
        bkg = read_icebkg(exptpath, yyyy, mm, dd_str)
        if bkg=="NO FILES":
          maeNP_cp4B.append(np.nan)
          rmsdNP_cp4B.append(np.nan)
          maeSP_cp4B.append(np.nan)
          rmsdSP_cp4B.append(np.nan)
          mae_extentNP_cp4B.append(np.nan)
          mae_extentSP_cp4B.append(np.nan)
          extentNP_cp4B.append(np.nan)
          extentSP_cp4B.append(np.nan)
        else:

          # Calculate sea ice extent
          t0 = bkg
          #     Arctic
          s0 = np.where( (lat>0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgNP = np.sum(s0)
          del s0
          #   Antarctic
          s0 = np.where( (lat<0.0) & (mask==1) & (t0>=area_min), t0, 0 )
          bkgSP = np.sum(s0)
          del s0
          del t0

          diffNP = obsNP - bkgNP
          diffSP = obsSP - bkgSP

          mae_extentNP_cp4B.append(np.mean(np.abs(diffNP)))
          mae_extentSP_cp4B.append(np.mean(np.abs(diffSP)))

          extentNP_cp4B.append(bkgNP)
          extentSP_cp4B.append(bkgSP)
          del diffNP, diffSP
          del bkgNP, bkgSP

          # Daily difference stats (per grid cell)
          diff = obs - bkg                # same shape: (1080, 1440)
                # Arctic
          I = np.where( (lat>0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeNP_cp4B.append(np.mean(np.abs(diff[I])))
          rmsdNP_cp4B.append(np.mean((diff[I])**2))
          del I
                # Antarctic
          I = np.where( (lat<0.0) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          maeSP_cp4B.append(np.mean(np.abs(diff[I])))
          rmsdSP_cp4B.append(np.mean((diff[I])**2))
          del I
          del bkg, diff
        del exptpath

        del mask

        #sys.exit()

        idd += 1
    imm += 1
  iyy += 1
#-------------------------------------------------------------------------
# END LOOP
#-------------------------------------------------------------------------

#*************************************************************************
# Plot Statistics
#*************************************************************************

#=======================================================
#=======================================================
print("===== MAE Sea Ice Concentration PLOTS: ARCTIC =====")
fig, ax = plt.subplots()

ax.plot_date(dates[0:len(maeNP_s2satmB)], maeNP_s2satmB[0:len(maeNP_s2satmB)], '.-', label='GFSv17 prototype: S2Smodel_ATMda', color='orange')
ax.plot_date(dates[0:len(maeNP_C03B)], maeNP_C03B[0:len(maeNP_C03B)], '.-', label='GFSv17 prototype: S2Smodel_S2Sda', color='magenta')
ax.plot_date(dates[0:len(maeNP_MCB)], maeNP_MCB[0:len(maeNP_MCB)], '.-', label='GFSv17 prototype: marine_candidate', color='red')
ax.plot_date(dates[0:len(maeNP_v1B)], maeNP_v1B[0:len(maeNP_v1B)], '.-', label='GFSv17 prototype: marine_test_v1', color='purple')
ax.plot_date(dates[0:len(maeNP_v2B)], maeNP_v2B[0:len(maeNP_v2B)], '.-', label='GFSv17 prototype: marine_test_v2', color='blue')
ax.plot_date(dates[0:len(maeNP_cp4B)], maeNP_cp4B[0:len(maeNP_cp4B)], '.-', label='GFSv17 prototype: cp4.01', color='cyan')

	#```````````````````````````````````````
	# Plotting specs
	#```````````````````````````````````````

legendloc = 'best' #'upper right'
        # plot all legend labels together
        #       ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
bbox = 0.5 #0.75
ax.legend(lines,labels,loc=legendloc, handlelength=10, bbox_to_anchor=[0.5, 0.9], prop={'size': 6})

plt.title('Mean Absolute Error (MAE) of Sea Ice Concentration \nBkg Forecasts vs OSTIA: Arctic', fontsize=12)
plt.ylabel('|bkg - OSTIA| (area fraction)', fontsize=10)
plt.grid(True)

axissize = 8
ax.xaxis.set_tick_params(rotation=30, labelsize=axissize)
ax.yaxis.set_tick_params(labelsize=axissize)
ax.autoscale_view()

plt.savefig(outpath+'time_series.seaice_area_fraction.MAE.'+str(yyyymmddS)+'-'+str(yyyymmddE)+'.arctic.png', dpi=600)

#=======================================================
#=======================================================
print("===== MAE Sea Ice Concentration PLOTS: ANTARCTIC =====")
fig, axSP = plt.subplots()

axSP.plot_date(dates[0:len(maeSP_s2satmB)], maeSP_s2satmB[0:len(maeSP_s2satmB)], '.-', label='GFSv17 prototype: S2Smodel_ATMda', color='orange')
axSP.plot_date(dates[0:len(maeSP_C03B)], maeSP_C03B[0:len(maeSP_C03B)], '.-', label='GFSv17 prototype: S2Smodel_S2Sda', color='magenta')
axSP.plot_date(dates[0:len(maeSP_MCB)], maeSP_MCB[0:len(maeSP_MCB)], '.-', label='GFSv17 prototype: marine_candidate', color='red')
axSP.plot_date(dates[0:len(maeSP_v1B)], maeSP_v1B[0:len(maeSP_v1B)], '.-', label='GFSv17 prototype: marine_test_v1', color='purple')
axSP.plot_date(dates[0:len(maeSP_v2B)], maeSP_v2B[0:len(maeSP_v2B)], '.-', label='GFSv17 prototype: marine_test_v2', color='blue')
axSP.plot_date(dates[0:len(maeSP_cp4B)], maeSP_cp4B[0:len(maeSP_cp4B)], '.-', label='GFSv17 prototype: cp4.01', color='cyan')

        #```````````````````````````````````````
        # Plotting specs
        #```````````````````````````````````````

legendloc = 'best' #'upper right'
        # plot all legend labels together
        #       ask matplotlib for the plotted objects and their labels
lines, labels = axSP.get_legend_handles_labels()
bbox = 0.5 #0.75
axSP.legend(lines,labels,loc=legendloc, handlelength=10, bbox_to_anchor=[bbox, 0.3], prop={'size': 6})

plt.title('Mean Absolute Error (MAE) of Sea Ice Concentration \nBkg Forecasts vs OSTIA: Antarctic', fontsize=12)
plt.ylabel('|bkg - OSTIA| (area fraction)', fontsize=10)
plt.grid(True)

axissize = 8
axSP.xaxis.set_tick_params(rotation=30, labelsize=axissize)
axSP.yaxis.set_tick_params(labelsize=axissize)
axSP.autoscale_view()

plt.savefig(outpath+'time_series.seaice_area_fraction.MAE.'+str(yyyymmddS)+'-'+str(yyyymmddE)+'.antarctic.png', dpi=600)

#=======================================================
#=======================================================
print("===== MAE Sea Ice Extent PLOTS: ARCTIC =====")
fig, axENPmae = plt.subplots()

axENPmae.plot_date(dates[0:len(mae_extentNP_s2satmB)], mae_extentNP_s2satmB[0:len(mae_extentNP_s2satmB)], '.-', label='GFSv17 prototype: S2Smodel_ATMda', color='orange')
axENPmae.plot_date(dates[0:len(mae_extentNP_C03B)], mae_extentNP_C03B[0:len(mae_extentNP_C03B)], '.-', label='GFSv17 prototype: S2Smodel_S2Sda', color='magenta')
axENPmae.plot_date(dates[0:len(mae_extentNP_MCB)], mae_extentNP_MCB[0:len(mae_extentNP_MCB)], '.-', label='GFSv17 prototype: marine_candidate', color='red')
axENPmae.plot_date(dates[0:len(mae_extentNP_v1B)], mae_extentNP_v1B[0:len(mae_extentNP_v1B)], '.-', label='GFSv17 prototype: marine_test_v1', color='purple')
axENPmae.plot_date(dates[0:len(mae_extentNP_v2B)], mae_extentNP_v2B[0:len(mae_extentNP_v2B)], '.-', label='GFSv17 prototype: marine_test_v2', color='blue')
axENPmae.plot_date(dates[0:len(mae_extentNP_cp4B)], mae_extentNP_cp4B[0:len(mae_extentNP_cp4B)], '.-', label='GFSv17 prototype: cp4.01', color='cyan')

        #```````````````````````````````````````
        # Plotting specs
        #```````````````````````````````````````

legendloc = 'best' #'upper right'
        # plot all legend labels together
        #       ask matplotlib for the plotted objects and their labels
lines, labels = axENPmae.get_legend_handles_labels()
bbox = 0.5 #0.75
axENPmae.legend(lines,labels,loc=legendloc, handlelength=10, bbox_to_anchor=[0.5, 0.65], prop={'size': 6})

plt.title('Mean Absolute Error (MAE) of Sea Ice Extent \nBkg Forecasts vs OSTIA: Arctic', fontsize=11)
plt.ylabel('|bkg - OSTIA| in km\u00B2', fontsize=10)      # units written here: km^2
plt.grid(True)

axissize = 8
axENPmae.xaxis.set_tick_params(rotation=30, labelsize=axissize)
axENPmae.yaxis.set_tick_params(labelsize=axissize)
axENPmae.autoscale_view()

plt.savefig(outpath+'time_series.seaice_extent.MAE.'+str(yyyymmddS)+'-'+str(yyyymmddE)+'.arctic.png', dpi=600)

#=======================================================
#=======================================================
print("===== MAE Sea Ice Extent PLOTS: ANTARCTIC =====")
fig, axESPmae = plt.subplots()

axESPmae.plot_date(dates[0:len(mae_extentSP_s2satmB)], mae_extentSP_s2satmB[0:len(mae_extentSP_s2satmB)], '.-', label='GFSv17 prototype: S2Smodel_ATMda', color='orange')
axESPmae.plot_date(dates[0:len(mae_extentSP_C03B)], mae_extentSP_C03B[0:len(mae_extentSP_C03B)], '.-', label='GFSv17 prototype: S2Smodel_S2Sda', color='magenta')
axESPmae.plot_date(dates[0:len(mae_extentSP_MCB)], mae_extentSP_MCB[0:len(mae_extentSP_MCB)], '.-', label='GFSv17 prototype: marine_candidate', color='red')
axESPmae.plot_date(dates[0:len(mae_extentSP_v1B)], mae_extentSP_v1B[0:len(mae_extentSP_v1B)], '.-', label='GFSv17 prototype: marine_test_v1', color='purple')
axESPmae.plot_date(dates[0:len(mae_extentSP_v2B)], mae_extentSP_v2B[0:len(mae_extentSP_v2B)], '.-', label='GFSv17 prototype: marine_test_v2', color='blue')
axESPmae.plot_date(dates[0:len(mae_extentSP_cp4B)], mae_extentSP_cp4B[0:len(mae_extentSP_cp4B)], '.-', label='GFSv17 prototype: cp4.01', color='cyan')

        #```````````````````````````````````````
        # Plotting specs
        #```````````````````````````````````````

legendloc = 'best' #'upper right'
        # plot all legend labels together
        #       ask matplotlib for the plotted objects and their labels
lines, labels = axESPmae.get_legend_handles_labels()
bbox = 0.5 #0.75
axESPmae.legend(lines,labels,loc=legendloc, handlelength=10, bbox_to_anchor=[bbox, 0.75], prop={'size': 6})

plt.title('Mean Absolute Error (MAE) of Sea Ice Extent \nBkg Forecasts vs OSTIA: Antarctic', fontsize=11)
plt.ylabel('|bkg - OSTIA| in km\u00B2', fontsize=10)      # units written here: km^2
plt.grid(True)

axissize = 8
axESPmae.xaxis.set_tick_params(rotation=30, labelsize=axissize)
axESPmae.yaxis.set_tick_params(labelsize=axissize)
axESPmae.autoscale_view()

plt.savefig(outpath+'time_series.seaice_extent.MAE.'+str(yyyymmddS)+'-'+str(yyyymmddE)+'.antarctic.png', dpi=600)

#=======================================================
#=======================================================
print("===== Sea Ice Extent PLOTS: ARCTIC =====")
fig, axENP = plt.subplots()

axENP.plot_date(dates[0:len(extentNP_obs)], extentNP_obs[0:len(extentNP_obs)], '.-', label='OSTIA (obs analysis)', color='grey')
axENP.plot_date(dates[0:len(extentNP_s2satmB)], extentNP_s2satmB[0:len(extentNP_s2satmB)], '.-', label='GFSv17 prototype: S2Smodel_ATMda (bkg)', color='orange')
axENP.plot_date(dates[0:len(extentNP_C03B)], extentNP_C03B[0:len(extentNP_C03B)], '.-', label='GFSv17 prototype: S2Smodel_S2Sda (bkg)', color='magenta')
axENP.plot_date(dates[0:len(extentNP_MCB)], extentNP_MCB[0:len(extentNP_MCB)], '.-', label='GFSv17 prototype: marine_candidate (bkg)', color='red')
axENP.plot_date(dates[0:len(extentNP_v1B)], extentNP_v1B[0:len(extentNP_v1B)], '.-', label='GFSv17 prototype: marine_test_v1 (bkg)', color='purple')
axENP.plot_date(dates[0:len(extentNP_v2B)], extentNP_v2B[0:len(extentNP_v2B)], '.-', label='GFSv17 prototype: marine_test_v2 (bkg)', color='blue')
axENP.plot_date(dates[0:len(extentNP_cp4B)], extentNP_cp4B[0:len(extentNP_cp4B)], '.-', label='GFSv17 prototype: cp4.01 (bkg)', color='cyan')

        #```````````````````````````````````````
        # Plotting specs
        #```````````````````````````````````````

legendloc = 'best' #'upper right'
        # plot all legend labels together
        #       ask matplotlib for the plotted objects and their labels
lines, labels = axENP.get_legend_handles_labels()
bbox = 0.5 #0.75
axENP.legend(lines,labels,loc=legendloc, handlelength=10, bbox_to_anchor=[0.5, 0.7], prop={'size': 6})

plt.title('Sea Ice Extent: Arctic', fontsize=12)
plt.ylabel('Area in km\u00B2', fontsize=10)      # units written here: km^2
plt.grid(True)

axissize = 8
axENP.xaxis.set_tick_params(rotation=30, labelsize=axissize)
axENP.yaxis.set_tick_params(labelsize=axissize)
axENP.autoscale_view()

plt.savefig(outpath+'time_series.seaice_extent.'+str(yyyymmddS)+'-'+str(yyyymmddE)+'.arctic.png', dpi=600)

#=======================================================
#=======================================================
print("===== Sea Ice Extent PLOTS: ANTARCTIC =====")
fig, axESP = plt.subplots()

axESP.plot_date(dates[0:len(extentSP_obs)], extentSP_obs[0:len(extentSP_obs)], '.-', label='OSTIA (obs analysis)', color='grey')
axESP.plot_date(dates[0:len(extentSP_s2satmB)], extentSP_s2satmB[0:len(extentSP_s2satmB)], '.-', label='GFSv17 prototype: S2Smodel_ATMda (bkg)', color='orange')
axESP.plot_date(dates[0:len(extentSP_C03B)], extentSP_C03B[0:len(extentSP_C03B)], '.-', label='GFSv17 prototype: S2Smodel_S2Sda (bkg)', color='magenta')
axESP.plot_date(dates[0:len(extentSP_MCB)], extentSP_MCB[0:len(extentSP_MCB)], '.-', label='GFSv17 prototype: marine_candidate (bkg)', color='red')
axESP.plot_date(dates[0:len(extentSP_v1B)], extentSP_v1B[0:len(extentSP_v1B)], '.-', label='GFSv17 prototype: marine_test_v1 (bkg)', color='purple')
axESP.plot_date(dates[0:len(extentSP_v2B)], extentSP_v2B[0:len(extentSP_v2B)], '.-', label='GFSv17 prototype: marine_test_v2 (bkg)', color='blue')
axESP.plot_date(dates[0:len(extentSP_cp4B)], extentSP_cp4B[0:len(extentSP_cp4B)], '.-', label='GFSv17 prototype: cp4.01 (bkg)', color='cyan')

        #```````````````````````````````````````
        # Plotting specs
        #```````````````````````````````````````

legendloc = 'best' #'upper right'
        # plot all legend labels together
        #       ask matplotlib for the plotted objects and their labels
lines, labels = axESP.get_legend_handles_labels()
bbox = 0.5 #0.75
axESP.legend(lines,labels,loc=legendloc, handlelength=10, bbox_to_anchor=[0.5, 0.25], prop={'size': 6})

plt.title('Sea Ice Extent: Antarctic', fontsize=12)
plt.ylabel('Area in km\u00B2', fontsize=10)      # units written here: km^2
plt.grid(True)

axissize = 8
axESP.xaxis.set_tick_params(rotation=30, labelsize=axissize)
axESP.yaxis.set_tick_params(labelsize=axissize)
axESP.autoscale_view()

plt.savefig(outpath+'time_series.seaice_extent.'+str(yyyymmddS)+'-'+str(yyyymmddE)+'.antarctic.png', dpi=600)

###################################################################################################################
print("========== END MAIN PROGRAM ==========")
###################################################################################################################
###################################################################################################################
