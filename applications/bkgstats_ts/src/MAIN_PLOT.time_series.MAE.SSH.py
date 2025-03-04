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

from read_data import read_copernicus
from read_data import read_ecm
from read_data import read_ocnbkg_ssh
from read_data import read_ocean_mask
from read_data import get_ocngrid

###################################################################################################################
print("========== BEGIN MAIN PROGRAM ==========")

#=============================================
# Set global parameters

fill = -999.0

inpath  = "/scratch2/NCEPDEV/ocean/Guillaume.Vernieres/runs/low-res/"

#=============================================
# Read raw user input from command line

yyyymmddS    = sys.argv[1]		# Start date: year month day
yyyymmddE    = sys.argv[2]		# End date: year month day
outpath      = sys.argv[3]
var          = sys.argv[4]
reference    = sys.argv[5]
ocean_choice = sys.argv[6]              # ocean basin choice: Global, Arctic, Atlantic, Indian, Pacific, Southern

# Choose to plot "Global" time series using only gridpoints at abs(lat)<60 (yes), or using all gridpoints (no)
latlimit = "yes"
#latlimit = "no"

if latlimit=="yes":
  latlimitstr = "LAT_lt_60."
elif latlimit=="no":
  latlimitstr = ""

#=============================================
# Ocean basins and masking file
#
#   ocean_choice = ocean basin over which to compute time series stats
#       Can be one of the following:
#           1. Arctic
#           2. Atlantic
#           3. Indian
#           4. Pacific
#           5. Southern
#

if ocean_choice!="Global":

    # open ocean masking file
    #   omask_lat, omask_lon should be 2D
    omask, omask_lat, omask_lon = read_ocean_mask(ocean_choice)
    #omask_lon2d, omask_lat2d = np.meshgrid(omask_lon, omask_lat)

    print("shape omask, omask_lat, omask_lon = "+str(np.shape(omask))+" "+str(np.shape(omask_lat))+" "+str(np.shape(omask_lon)))
    print("omask num not NaN = "+str(np.size(~np.isnan(omask))))

#-------------------------------------------
# Get ocean lat/lon arrays (2D each)
lat, lon = get_ocngrid()

# Read soca_gridspec file for proper land/sea masking and apply to OSTIA
gridfile = "/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/data/soca_gridspec.nc"
ds       = xr.open_dataset(gridfile)
ufs_mask = np.squeeze(ds['mask2d'][:])          # land/sea mask
#ufs_area = np.squeeze(ds['area'][:])            # grid cell area
del gridfile, ds

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

#---------------------------------------------------------------
# Read pickled data files
#	Various runs are pickled
#	Does not include GSI analysis, background/analysis from marine DA (JEDI/SOCA)

# !!! There is not SSH in operations. Do not use this file.

#with open('/scratch2/NCEPDEV/ocean/Guillaume.Vernieres/runs/low-res/nadt-ops-mae.pickle', 'rb') as f:
#    nsst_ops = pickle.load(f)

#******************************************************************************************************************
#******************************************************************************************************************
# PLOTS
#******************************************************************************************************************
#******************************************************************************************************************

#---------------------------------------------------------------
# Time Series
#---------------------------------------------------------------

fig, ax = plt.subplots()

lat, lon = get_ocngrid()

#days = list(range(int(ddS), int(ddE)))

mmARR = [ '01','02','03','04','05','06','07','08','09','10','11','12' ]
ddARR = [ "01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31" ]
hours = [ '00','06','12','18' ]

#```````````````````````````````````````
# LOOP
#```````````````````````````````````````

mae_ecm     = []
mae_atmatmB = []
mae_s2satmB = []
mae_C03B = []
mae_C03A = []
mae_MCB = []
mae_MCA = []
mae_v1B = []
mae_v2B = []
mae_cp4B = []
mae_cp4drifterB = []

rmsd_ecm    = []
rmsd_atmatmB = []
rmsd_s2satmB = []
rmsd_C03B = []
rmsd_C03A = []
rmsd_MCB = []
rmsd_MCA = []
rmsd_v1B = []
rmsd_v2B = []
rmsd_cp4B = []
rmsd_cp4drifterB = []

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
        print("... OBS ...")
        #------------------------------------------
        # get COPERNICUS ssh
        if reference.upper()=="COPERNICUS":
          obs_sst = read_copernicus(yyyy+mm+dd_str)
      
                # get indices where obs_sst != NaN for SSH mask
        arr_obs = obs_sst.flatten()
        ind_obs = np.argwhere(~np.isnan(arr_obs))
                # remove global mean from each grid point
        meanvar = np.nanmean(arr_obs[ind_obs])
        obs_sst = obs_sst - meanvar
        del meanvar, arr_obs

        # Combine UFS mask and OBS mask
        #   mask==1 = where open ocean exists in both ufs_mask and obs_mask, and land does not exist in either
        if ocean_choice=="Global":
          mask = np.where((~np.isnan(obs_sst)) & (ufs_mask==1), 1, 0)    # obs_sst!=NaN = open ocean | ufs_mask==0 = land ; ufs_mask==1 = not land
        else:
          mask = np.where((~np.isnan(obs_sst)) & (ufs_mask==1) & (~np.isnan(omask)), 1, 0)     # only ocean grid cells in specified ocean basin

        #------------------------------------------

        print("----- EXPERIMENTS: ")#+str(dt.datetime.now()))

        #------------------------------------------
        # get ORAS5 ssh
        print("... ORAS5 ...")
        if mm!='07':
          mae_ecm.append(np.nan)
          rmsd_ecm.append(np.nan)
        else:
          ecm_sst = read_ecm(yyyy, mm, dd_str, var.lower())
                
                # remove global mean from each grid point
          arr = ecm_sst.flatten()
          meanvar = np.nanmean(arr[ind_obs])
          ecm_sst = ecm_sst - meanvar
          del meanvar, arr

          diff_ecm = ecm_sst - obs_sst

          if ocean_choice=="Global":
            if latlimit=="yes":
              I = np.where( (abs(lat<60.0)) & (mask==1) & (~np.isnan(diff_ecm)) & (abs(diff_ecm)<999.9) )
            elif latlimit=="no":
              I = np.where( (mask==1) & (~np.isnan(diff_ecm)) & (abs(diff_ecm)<999.9) )
          else:
            I = np.where( (mask==1) & (~np.isnan(diff_ecm)) & (abs(diff_ecm)<999.9) )

          mae_ecm.append(np.mean(np.abs(diff_ecm[I])))
          rmsd_ecm.append(np.mean((diff_ecm[I])**2))
          del ecm_sst,diff_ecm,I

        #------------------------------------------
        # EXPT: S2Smodel_ATMda (GFSv17 prototype)

        print("... S2Smodel_ATMda ...")
        exptpath = "/scratch1/NCEPDEV/climate/role.ufscpara/ModelOutput/CycledTests/HR_3_5_s2s_model_atm_da/"

                #``````````````````````````````````
                # backgrounds
        bkg_sst = read_ocnbkg_ssh(exptpath, yyyy, mm, dd_str)
        
        if bkg_sst=="NO FILES":
          mae_s2satmB.append(np.nan)
          rmsd_s2satmB.append(np.nan)
        else:

                # remove global mean from each grid point
          arr = bkg_sst.flatten()
          meanvar = np.nanmean(arr[ind_obs])
          bkg_sst = bkg_sst - meanvar
          del meanvar, arr

          diff = obs_sst - bkg_sst                # same shape: (1080, 1440)

          if ocean_choice=="Global":
            if latlimit=="yes":
              I = np.where( (abs(lat<60.0)) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
            elif latlimit=="no":
              I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          else:
            I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )

          mae_s2satmB.append(np.mean(np.abs(diff[I])))
          rmsd_s2satmB.append(np.mean((diff[I])**2))
          del bkg_sst,diff,I

        del exptpath

        #------------------------------------------
        # EXPT: S2Smodel_S2Sda_C03 (GFSv17 prototype)

        print("... S2Smodel_S2Sda ...")
        exptpath = "/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/data/expts/S2Smodel_S2Sda_C03/"

                #``````````````````````````````````
                # backgrounds
        bkg_sst = read_ocnbkg_ssh(exptpath, yyyy, mm, dd_str)
        
        if bkg_sst=="NO FILES":
          mae_C03B.append(np.nan)
          rmsd_C03B.append(np.nan)
        else:

                # remove global mean from each grid point
          arr = bkg_sst.flatten()
          meanvar = np.nanmean(arr[ind_obs])
          bkg_sst = bkg_sst - meanvar
          del meanvar, arr

          diff = obs_sst - bkg_sst                # same shape: (1080, 1440)

          if ocean_choice=="Global":
            if latlimit=="yes":
              I = np.where( (abs(lat<60.0)) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
            elif latlimit=="no":
              I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          else:
            I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )

          mae_C03B.append(np.mean(np.abs(diff[I])))
          rmsd_C03B.append(np.mean((diff[I])**2))
          del bkg_sst,diff,I

        del exptpath

        #------------------------------------------
        # EXPT: marine_test_v1 (GFSv17 prototype)

        print("... marine_test_v1 ...")
        exptpath = "/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/data/expts/marine_test_112024_v1/"

                #``````````````````````````````````
                # backgrounds
        bkg_sst = read_ocnbkg_ssh(exptpath, yyyy, mm, dd_str)
        
        if bkg_sst=="NO FILES":
          mae_v1B.append(np.nan)
          rmsd_v1B.append(np.nan)
        else:

                # remove global mean from each grid point
          arr = bkg_sst.flatten()
          meanvar = np.nanmean(arr[ind_obs])
          bkg_sst = bkg_sst - meanvar
          del meanvar, arr

          diff = obs_sst - bkg_sst                # same shape: (1080, 1440)

          if ocean_choice=="Global":
            if latlimit=="yes":
              I = np.where( (abs(lat<60.0)) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
            elif latlimit=="no":
              I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          else:
            I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )

          mae_v1B.append(np.mean(np.abs(diff[I])))
          rmsd_v1B.append(np.mean((diff[I])**2))
          del bkg_sst,diff,I

        del exptpath

        #------------------------------------------
        # EXPT: marine_test_v2 (GFSv17 prototype)

        print("... marine_test_v2 ...")
        exptpath = "/scratch1/NCEPDEV/global/John.Steffen/hpss_arch/marine_test_112024_v2/"

                #``````````````````````````````````
                # backgrounds
        bkg_sst = read_ocnbkg_ssh(exptpath, yyyy, mm, dd_str)
        
        if bkg_sst=="NO FILES":
          mae_v2B.append(np.nan)
          rmsd_v2B.append(np.nan)
        else:

                # remove global mean from each grid point
          arr = bkg_sst.flatten()
          meanvar = np.nanmean(arr[ind_obs])
          bkg_sst = bkg_sst - meanvar
          del meanvar, arr

          diff = obs_sst - bkg_sst                # same shape: (1080, 1440)

          if ocean_choice=="Global":
            if latlimit=="yes":
              I = np.where( (abs(lat<60.0)) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
            elif latlimit=="no":
              I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          else:
            I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )

          mae_v2B.append(np.mean(np.abs(diff[I])))
          rmsd_v2B.append(np.mean((diff[I])**2))
          del bkg_sst,diff,I

        del exptpath

	#------------------------------------------
        # EXPT: marine_candidate (4th GFSv17 prototype)

        print("... marine_candidate ...")
        exptpath = "/scratch1/NCEPDEV/global/John.Steffen/hpss_arch/marine_candidate_092024/"

                #``````````````````````````````````
                # backgrounds
        bkg_sst = read_ocnbkg_ssh(exptpath, yyyy, mm, dd_str)
        
        if bkg_sst=="NO FILES":
          mae_MCB.append(np.nan)
          rmsd_MCB.append(np.nan)
        else:

                # remove global mean from each grid point
          arr = bkg_sst.flatten()
          meanvar = np.nanmean(arr[ind_obs])
          bkg_sst = bkg_sst - meanvar
          del meanvar, arr

          diff = obs_sst - bkg_sst                # same shape: (1080, 1440)

          if ocean_choice=="Global":
            if latlimit=="yes":
              I = np.where( (abs(lat<60.0)) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
            elif latlimit=="no":
              I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          else:
            I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )

          mae_MCB.append(np.mean(np.abs(diff[I])))
          rmsd_MCB.append(np.mean((diff[I])**2))
          del bkg_sst,diff,I

        del exptpath

        #------------------------------------------
        # EXPT: cp4.01 run (GFSv17 prototype)

        print("... cp4.01 ...")
        exptpath = "/scratch1/NCEPDEV/global/John.Steffen/hpss_arch/cp4.01-JS/COMROOT/cp4.01-JS/"

                #``````````````````````````````````
                # backgrounds
        bkg_sst = read_ocnbkg_ssh(exptpath, yyyy, mm, dd_str)
        
        if bkg_sst=="NO FILES":
          mae_cp4B.append(np.nan)
          rmsd_cp4B.append(np.nan)
        else:

                # remove global mean from each grid point
          arr = bkg_sst.flatten()
          meanvar = np.nanmean(arr[ind_obs])
          bkg_sst = bkg_sst - meanvar
          del meanvar, arr

          diff = obs_sst - bkg_sst                # same shape: (1080, 1440)

          if ocean_choice=="Global":
            if latlimit=="yes":
              I = np.where( (abs(lat<60.0)) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
            elif latlimit=="no":
              I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          else:
            I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )

          mae_cp4B.append(np.mean(np.abs(diff[I])))
          rmsd_cp4B.append(np.mean((diff[I])**2))
          del bkg_sst,diff,I

        del exptpath

        #------------------------------------------
        # EXPT: cp4.01 run by John (GFSv17 prototype)

        print("... cp4.01 drifterdepth ...")
        exptpath = "/scratch1/NCEPDEV/global/Katherine.Lukens/expts/sst_alternate_hofx/cp4.01-KL-drifterdepth/cp4.01-sst-hofx-drifterdepth-KL.v2/COMROOT/cp4.01-sst-hofx-drifterdepth-KL.v2/"

                #``````````````````````````````````
                # backgrounds
        bkg_sst = read_ocnbkg_ssh(exptpath, yyyy, mm, dd_str)

        if bkg_sst=="NO FILES":
          mae_cp4drifterB.append(np.nan)
          rmsd_cp4drifterB.append(np.nan)
        else:
            # remove global mean from each grid point
          arr = bkg_sst.flatten()
          meanvar = np.nanmean(arr[ind_obs])
          bkg_sst = bkg_sst - meanvar
          del meanvar, arr

          diff = obs_sst - bkg_sst                # same shape: (1080, 1440)

          if ocean_choice=="Global":
            if latlimit=="yes":
              I = np.where( (abs(lat<60.0)) & (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
            elif latlimit=="no":
              I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )
          else:
            I = np.where( (mask==1) & (~np.isnan(diff)) & (abs(diff)<999.9) )

          mae_cp4drifterB.append(np.mean(np.abs(diff[I])))
          rmsd_cp4drifterB.append(np.mean((diff[I])**2))
          del bkg_sst,diff,I

        del exptpath


        #------------------------------------------
        
        del ind_obs

        del mask

        idd += 1
    imm += 1
  iyy += 1
#---------------------------------------
# END LOOP
#---------------------------------------

#```````````````````````````````````````
# Plot stats
#```````````````````````````````````````

#print("MAE: NOAA Operations = "+str(nsst_ops))
print("MAE: ORAS5 (ECMWF) = "+str(mae_ecm))
print("MAE: S2Smodel_ATMda = "+str(mae_s2satmB))
print("MAE: S2Smodel_S2Sda = "+str(mae_C03B))
print("MAE: marine_candidate = "+str(mae_MCB))

	# MAE
ax.plot_date(dates[0:len(mae_ecm)], mae_ecm[0:len(mae_ecm)], '.-', label='ORAS5 Replay (ECMWF)', color='black')
#ax.plot_date(dates[1:len(mae_atmatmB)], mae_atmatmB[1:len(mae_atmatmB)], '.-', label='GFSv17 prototype: ATMmodel_ATMda', color='magenta')

ax.plot_date(dates[0:len(mae_s2satmB)], mae_s2satmB[0:len(mae_s2satmB)], '.-', label='GFSv17 prototype: S2Smodel_ATMda', color='orange')
ax.plot_date(dates[0:len(mae_C03B)], mae_C03B[0:len(mae_C03B)], '.-', label='GFSv17 prototype: S2Smodel_S2Sda', color='magenta')
ax.plot_date(dates[0:len(mae_MCB)], mae_MCB[0:len(mae_MCB)], '.-', label='GFSv17 prototype: marine_candidate', color='red')
ax.plot_date(dates[0:len(mae_v1B)], mae_v1B[0:len(mae_v1B)], '.-', label='GFSv17 prototype: marine_test_v1', color='purple')
ax.plot_date(dates[0:len(mae_v2B)], mae_v2B[0:len(mae_v2B)], '.-', label='GFSv17 prototype: marine_test_v2', color='blue')
ax.plot_date(dates[0:len(mae_cp4B)], mae_cp4B[0:len(mae_cp4B)], '.-', label='GFSv17 prototype: cp4', color='cyan')
ax.plot_date(dates[0:len(mae_cp4drifterB)], mae_cp4drifterB[0:len(mae_cp4drifterB)], '.-', label='GFSv17 prototype: cp4-drifterdepth', color='lime')

	# RMSD
##ax.plot_date(dates[0:len(rmsd_gdas)], rmsd_gdas[:], '.-', label='cp0.b Ocean BKG: RMSD')
##ax.plot_date(dates[0:len(rmsd_gsi)], rmsd_gsi[:], '.-', label='cp0.b Ocean ANL: RMSD')


	#```````````````````````````````````````
	# Plotting specs
	#```````````````````````````````````````

legendloc = 'best' #'upper right'
        # plot all legend labels together
        #       ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
bbox = 0.5 #0.75
#ax.legend(lines,labels,loc=legendloc, handlelength=10, bbox_to_anchor=[bbox, bbox], prop={'size': 6})
ax.legend(lines,labels,loc=legendloc, handlelength=10, bbox_to_anchor=[0.5, 0.2], prop={'size': 6})

plt.title('Mean Absolute Error (MAE) of SSH Bkg Forecasts vs COPERNICUS: '+str(ocean_choice), fontsize=12)
plt.ylabel('|bkg - COPERNICUS| (m)', fontsize=10)
plt.grid(True)

axissize = 8
ax.xaxis.set_tick_params(rotation=30, labelsize=axissize)
ax.yaxis.set_tick_params(labelsize=axissize)
ax.autoscale_view()

plt.savefig(outpath+'time_series.ssh.MAE.'+latlimitstr+str(yyyymmddS)+'-'+str(yyyymmddE)+'.'+str(ocean_choice.lower())+'.png', dpi=600)


###################################################################################################################
print("========== END MAIN PROGRAM ==========")
###################################################################################################################
###################################################################################################################
