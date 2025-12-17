import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import datetime as dt

from read_data import read_ostia
from read_data import read_ostia_ice
from read_data import read_ecm
from read_data import read_copernicus
from read_data import read_ocnbkg_sst
from read_data import read_ocnbkg_ssh
from read_data import read_icebkg
from read_data import read_ocean_mask
from read_data import get_ocngrid

# ##################################################################################################################
# PLOT Time Series of Multiple Runs/Observation Datasets
# ##################################################################################################################
#
# Output: Figure
#
# How to run: python NameOfScript.py yyyymmdd yyyymmdd outpath var reference ocean_choice [HPSS_root]
#       First yyyymmdd is START DATE
#       Second yyyymmdd is END DATE
#       outpath is the output directory path
#       var is the variable to plot (SST, SSH, SEAICE)
#       reference is the reference dataset (OSTIA, COPERNICUS)
#       ocean_choice is the ocean basin (Global, Arctic, Atlantic, Indian, Pacific, Southern)
#       HPSS_root is optional HPSS archive root path (default: /scratch1/NCEPDEV/global/John.Steffen/hpss_arch/)
#
# ##################################################################################################################


def plot_time_series_mae(yyyymmdd_s, yyyymmdd_e, outpath, var, reference, ocean_choice,
                         HPSS_root="/scratch1/NCEPDEV/global/John.Steffen/hpss_arch/"):
    """
    Plot time series of Mean Absolute Error (MAE) for multiple experiments.

    Parameters
    ----------
    yyyymmdd_s : str
        Start date in YYYYMMDD format
    yyyymmdd_e : str
        End date in YYYYMMDD format
    outpath : str
        Output directory path for saving plots
    var : str
        Variable to plot (SST, SSH, or SEAICE)
    reference : str
        Reference dataset name (OSTIA or COPERNICUS)
    ocean_choice : str
        Ocean basin choice (Global, Arctic, Atlantic, Indian, Pacific, Southern)
    HPSS_root : str, optional
        Root path for HPSS archive, by default "/scratch1/NCEPDEV/global/John.Steffen/hpss_arch/"
    """
    print("========== BEGIN MAIN PROGRAM ==========")

    # =============================================
    # USER INPUT

    # --- Paths to experiments you wish to plot
    # Dynamically construct paths using HPSS_root parameter
    exptpaths = [
        "/scratch1/NCEPDEV/climate/role.ufscpara/ModelOutput/CycledTests/HR_3_5_s2s_model_atm_da/",
        "/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/data/expts/S2Smodel_S2Sda_C03/",
        "/scratch1/NCEPDEV/da/Katherine.Lukens/NSST/data/expts/marine_test_112024_v1/",
        HPSS_root + "marine_test_112024_v2/",
        HPSS_root + "marine_candidate_092024/",
        HPSS_root + "cp4.01-JS/COMROOT/cp4.01-JS/",
        "/scratch1/NCEPDEV/global/Katherine.Lukens/expts/sst_alternate_hofx/cp4.01-KL-drifterdepth/"
        + "cp4.01-sst-hofx-drifterdepth-KL.v2/COMROOT/cp4.01-sst-hofx-drifterdepth-KL.v2/"
    ]

    # ... ... Experiment names (make sure they line up with their corresponding paths in "exptpaths"
    exptnames = [
        "GFSv17_prototype__S2Smodel_ATMda",
        "GFSv17_prototype__S2Smodel_S2Sda",
        "GFSv17_prototype__marine_candidate",
        "GFSv17_prototype__marine_test_v1",
        "GFSv17_prototype__marine_test_v2",
        "GFSv17_prototype__cp4.01",
        "GFSv17_prototype__cp4.01-drifterdepth"
    ]

    # ... ... Line colors corresponding to experiments
    colors = [
        "orange",
        "magenta",
        "red",
        "purple",
        "blue",
        "cyan",
        "lime"
    ]

    # ... ... Choose latitude expanse over which to compute daily means:
    # ...        yes = plot time series using only gridpoints at |lat|<60
    # ...        no  = use all gridpoints
    latlimit = "yes"
    # latlimit = "no"

    # ... ... Choose the model level to compare against reference
    # sst_model_level = 0
    sst_model_level = 1     # 1 = model level below top

    # =============================================
    # Set global parameters

    fill = -999.0

    inpath = "/scratch2/NCEPDEV/ocean/Guillaume.Vernieres/runs/low-res/"

    # =============================================
    # Validate input parameters

    # Set latitude expanse over which to compute daily means
    if latlimit == "yes":
        latlimitstr = "latLT60."
    elif latlimit == "no":
        latlimitstr = ""

    # Set units
    if var.upper() == "SST":
        units = "deg-C"
    elif var.upper() == "SSH":
        units = "m"
    elif var.upper() == "SEAICE":
        units_fraction = "area fraction"
        units_extent = "km\u00B2"

    # =============================================
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

    if ocean_choice != "Global":
        # open ocean masking file
        #   omask_lat, omask_lon should be 2D
        omask, omask_lat, omask_lon = read_ocean_mask(ocean_choice)
        # print("shape omask, omask_lat, omask_lon = "+str(np.shape(omask))+" "+str(np.shape(omask_lat))+" "+str(np.shape(omask_lon)))
        # print("omask num not NaN = "+str(np.size(~np.isnan(omask))))

    # -------------------------------------------
    # Get ocean lat/lon arrays (2D each)
    lat, lon = get_ocngrid()

    # Read soca_gridspec file for proper land/sea masking and apply to OSTIA
    cwd = os.getcwd()       # get current working directory (CWD)
    gridfile = str(cwd) + "/soca_gridspec.nc"
    ds = xr.open_dataset(gridfile)
    ufs_mask = np.squeeze(ds['mask2d'][:])          # land/sea mask
    ufs_area = np.squeeze(ds['area'][:])            # grid cell area
    del gridfile, ds

    # Sea ice area extent minimum
    area_min = 0.15

    # -------------------------------------------
    # Day/time information

    # Start date
    yyyy_s = yyyymmdd_s[0:4]
    mm_s = yyyymmdd_s[4:6]
    dd_s = yyyymmdd_s[6:8]
    # End date
    yyyy_e = yyyymmdd_e[0:4]
    mm_e = yyyymmdd_e[4:6]
    dd_e = yyyymmdd_e[6:8]

    # Create array of dates for time series plots
    start_date = dt.date(int(yyyy_s), int(mm_s), int(dd_s))
    end_date = dt.date(int(yyyy_e), int(mm_e), int(dd_e))
    delta = dt.timedelta(days=1)
    dates = []
    while start_date <= end_date:
        dates.append(start_date)
        start_date += delta
    del start_date, end_date, delta

    # List containing all possible months (string)
    mm_arr = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    # List containing all possible days (string)
    dd_arr = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
              "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
              "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    # List containing all possible analysis hours (string)
    hours = ['00', '06', '12', '18']

    # -------------------------------------------
    # Read pickled data files
    # ... Various runs are pickled
    # ... Does not include GSI analysis, background/analysis from marine DA (JEDI/SOCA)
    if var.upper() == "SST":
        with open(inpath + 'mae.hc.pkl', 'rb') as f:
            hc_sst = pickle.load(f)
        with open(inpath + 'wcda-mae.pickle', 'rb') as f:
            wcda_sst = pickle.load(f)
        with open('/scratch2/NCEPDEV/ocean/Guillaume.Vernieres/runs/low-res/nsst-ops-mae.pickle', 'rb') as f:
            nsst_ops = pickle.load(f)
        with open(inpath + 'cp0.b-mae.pickle', 'rb') as f:
            cp0b_sst = pickle.load(f)

    # ******************************************************************************************************************
    # ******************************************************************************************************************
    # LOOP through each day in selected time range
    # ******************************************************************************************************************
    # ******************************************************************************************************************

    # -------------------------------------------
    # Initialize daily mean time series output lists

    # ... SST or SSH
    # ... ... ECMWF analysis
    sd_ecm = []
    mae_ecm = []
    rmsd_ecm = []
    # ... ... background forecasts
    print("len exptpaths = " + str(len(exptpaths)))
    sd_b = [[] for i in range(len(exptpaths))]
    mae_b = [[] for i in range(len(exptpaths))]
    rmsd_b = [[] for i in range(len(exptpaths))]

    print("sd_b shape = " + str(np.shape(sd_b)) + " type = " + str(type(sd_b)))

    # ... SEAICE
    # ... ... obs
    extent_np_obs = []
    extent_sp_obs = []
    mae_extent_np_obs = []
    mae_extent_sp_obs = []

    # ... ... background forecasts
    mae_np_b = [[] for i in range(len(exptpaths))]
    mae_sp_b = [[] for i in range(len(exptpaths))]
    rmsd_np_b = [[] for i in range(len(exptpaths))]
    rmsd_sp_b = [[] for i in range(len(exptpaths))]
    extent_np_b = [[] for i in range(len(exptpaths))]
    extent_sp_b = [[] for i in range(len(exptpaths))]
    mae_extent_np_b = [[] for i in range(len(exptpaths))]
    mae_extent_sp_b = [[] for i in range(len(exptpaths))]

    # -------------------------------------------
    # START LOOP
    # -------------------------------------------
    yyyy = yyyy_s
    iyy = int(yyyy)
    iyy_e = int(yyyy_e)
    while iyy <= int(yyyy_e):
        if yyyy_s == yyyy_e:
            simm = mm_arr.index(mm_s)
            eimm = mm_arr.index(mm_e)
        elif yyyy_s != yyyy_e:
            if iyy == int(yyyy_s):
                simm = mm_arr.index(mm_s)
                eimm = 12 - 1
            elif iyy > int(yyyy_s) and iyy < iyy_e:
                simm = 0
                eimm = 12 - 1
            elif iyy == iyy_e:
                simm = 0
                eimm = mm_arr.index(mm_e)

        imm = simm
        while imm <= eimm:
            mm = mm_arr[imm]
            if mm == "02":
                if int(yyyy) % 4 == 0:
                    # leap year
                    n_dd = 29
                else:
                    n_dd = 28
            elif mm == "04" or mm == "06" or mm == "09" or mm == "11":
                n_dd = 30
            else:
                n_dd = 31
            if yyyy_s == yyyy_e:
                if mm_s == mm_e:
                    sidd = dd_arr.index(dd_s)
                    eidd = dd_arr.index(dd_e)
                else:
                    if mm == mm_s:
                        sidd = dd_arr.index(dd_s)
                        eidd = n_dd - 1
                    elif mm == mm_e:
                        sidd = 0
                        eidd = dd_arr.index(dd_e)
                    else:
                        sidd = 0
                        eidd = n_dd - 1
            elif yyyy_s != yyyy_e:
                if yyyy == yyyy_s and mm == mm_s:
                    sidd = dd_arr.index(dd_s)
                    eidd = n_dd - 1
                elif iyy == int(yyyy_e) and mm == mm_e:
                    sidd = 0
                    eidd = dd_arr.index(dd_e)
                else:
                    sidd = 0
                    eidd = n_dd - 1

            idd = sidd
            while idd <= eidd:
                dd = dd_arr[idd]
                print("===== DATE: " + str(yyyy) + "-" + str(mm) + "-" + str(dd) + " =====")

                dd_str = str(dd).zfill(2)		# add zeroes to beginning of string

                # ------------------------------------------
                print("----- OBSERVATIONS: ")
                print("... " + str(reference.upper()) + " " + str(var.upper()))
                if reference.upper() == 'OSTIA':
                    if var.upper() == 'SST':
                        obs, obs_mask = read_ostia(yyyy + mm + dd_str)
                    else:
                        obs, obs_mask = read_ostia_ice(yyyy + mm + dd_str)      # extracts sea ice area fraction
                elif reference.upper() == "COPERNICUS":
                    obs = read_copernicus(yyyy + mm + dd_str)
                    # Get indices where obs_sst != NaN for SSH mask
                    arr_obs = obs.flatten()
                    ind_obs = np.argwhere(~np.isnan(arr_obs))
                    # Remove global mean from each grid point
                    meanvar = np.nanmean(arr_obs[ind_obs])
                    obs = obs - meanvar
                    del meanvar, arr_obs

                # Combine UFS mask and OBS mask
                if var.upper() == "SST":
                    # ... mask==1 = where open ocean exists in both ufs_mask and obs_mask, and land does not exist in either
                    if ocean_choice == "Global":
                        # obs_mask==1 = ocean | ufs_mask==0 = land ; ufs_mask==1 = not land
                        mask = np.where((obs_mask == 1) & (ufs_mask == 1), 1, 0)
                    else:
                        # only ocean grid cells in specified ocean basin
                        mask = np.where((obs_mask == 1) & (ufs_mask == 1) & (~np.isnan(omask)), 1, 0)
                    del obs_mask
                elif var.upper() == "SSH":
                    # ... mask==1 = where open ocean exists in both ufs_mask and obs_mask, and land does not exist in either
                    if ocean_choice == "Global":
                        # obs_sst!=NaN = open ocean | ufs_mask==0 = land ; ufs_mask==1 = not land
                        mask = np.where((~np.isnan(obs)) & (ufs_mask == 1), 1, 0)
                    else:
                        # only ocean grid cells in specified ocean basin
                        mask = np.where((~np.isnan(obs)) & (ufs_mask == 1) & (~np.isnan(omask)), 1, 0)
                elif var.upper() == "SEAICE":
                    # ... mask==1 = where sea ice exists in both ufs_mask and obs_mask, and land does not exist in either
                    # obs_mask==9 = sea ice | ufs_mask==0 = land ; ufs_mask==1 = not land
                    mask = np.where((obs_mask == 9) & (ufs_mask == 1), 1, 0)
                    del obs_mask
                    # Calc sea ice area extent for obs
                    t0 = obs
                    # ... Arctic
                    s0 = np.where((lat > 0.0) & (mask == 1) & (t0 >= area_min), t0, 0)
                    obs_np = np.sum(s0)
                    del s0
                    # ... Antarctic
                    s0 = np.where((lat < 0.0) & (mask == 1) & (t0 >= area_min), t0, 0)
                    obs_sp = np.sum(s0)
                    del s0
                    del t0
                    # Append extent to total arrays
                    extent_np_obs.append(obs_np)
                    extent_sp_obs.append(obs_sp)

                # ------------------------------------------
                # Get ORAS5 temperature
                if var.upper() == "SST" or var.upper() == "SSH":
                    print("... ORAS5 from ECMWF ...")
                    if mm != '07':
                        sd_ecm.append(np.nan)
                        mae_ecm.append(np.nan)
                        rmsd_ecm.append(np.nan)
                    else:
                        ecm = read_ecm(yyyy, mm, dd_str, var)
                        if var.upper() == "SSH":
                            # Remove global mean from each grid point
                            arr = ecm.flatten()
                            meanvar = np.nanmean(arr[ind_obs])
                            ecm = ecm - meanvar
                            del meanvar, arr
                        diff_ecm = ecm - obs
                        # Apply masking
                        if ocean_choice == "Global":
                            if latlimit == "yes":
                                II = np.where((abs(lat < 60.0)) & (mask == 1) & (~np.isnan(diff_ecm)) & (abs(diff_ecm) < 999.9))
                            elif latlimit == "no":
                                II = np.where((mask == 1) & (~np.isnan(diff_ecm)) & (abs(diff_ecm) < 999.9))
                        else:
                            II = np.where((mask == 1) & (~np.isnan(diff_ecm)) & (abs(diff_ecm) < 999.9))
                        # Append stats to arrays for plotting
                        sd_ecm.append(np.std(diff_ecm[II]))
                        mae_ecm.append(np.mean(np.abs(diff_ecm[II])))
                        rmsd_ecm.append(np.mean((diff_ecm[II])**2))
                        del ecm, diff_ecm, II

                # ------------------------------------------
                # ------------------------------------------
                print("----- LOOP THROUGH EXPERIMENTS:")
                # ------------------------------------------
                for iexpt in range(len(exptpaths)):
                    print("exptname = " + str(exptnames[iexpt]))
                    print("iexpt type = " + str(type(iexpt)) + " " + str(iexpt) + "/" + str(len(exptpaths) - 1))

                    # Get path to experiment
                    exptpath = exptpaths[iexpt]
                    # Get background forecasts
                    if var.upper() == "SST":
                        bkg = read_ocnbkg_sst(exptpath, yyyy, mm, dd_str)
                    elif var.upper() == "SSH":
                        bkg = read_ocnbkg_ssh(exptpath, yyyy, mm, dd_str)
                    elif var.upper() == "SEAICE":
                        bkg = read_icebkg(exptpath, yyyy, mm, dd_str)

                    if bkg == "NO FILES":
                        if var.upper() == "SEAICE":
                            mae_np_b[iexpt].append(np.nan)
                            mae_sp_b[iexpt].append(np.nan)
                            rmsd_np_b[iexpt].append(np.nan)
                            rmsd_sp_b[iexpt].append(np.nan)
                            mae_extent_np_b[iexpt].append(np.nan)
                            mae_extent_sp_b[iexpt].append(np.nan)
                            extent_np_b[iexpt].append(np.nan)
                            extent_sp_b[iexpt].append(np.nan)
                        else:
                            sd_b[iexpt].append(np.nan)
                            mae_b[iexpt].append(np.nan)
                            rmsd_b[iexpt].append(np.nan)
                    else:
                        if var.upper() == "SEAICE":
                            # Calculate sea ice extent
                            t0 = bkg
                            # ... Arctic
                            s0 = np.where((lat > 0.0) & (mask == 1) & (t0 >= area_min), t0, 0)
                            bkg_np = np.sum(s0)
                            del s0
                            # ... Antarctic
                            s0 = np.where((lat < 0.0) & (mask == 1) & (t0 >= area_min), t0, 0)
                            bkg_sp = np.sum(s0)
                            del s0
                            del t0
                            # Compute differences between obs and bkg
                            diff_np = obs_np - bkg_np
                            diff_sp = obs_sp - bkg_sp
                            # Append sea ice extent stats to arrays
                            mae_extent_np_b[iexpt].append(np.mean(np.abs(diff_np)))
                            mae_extent_sp_b[iexpt].append(np.mean(np.abs(diff_sp)))
                            extent_np_b[iexpt].append(bkg_np)
                            extent_sp_b[iexpt].append(bkg_sp)
                            del diff_np, diff_sp
                            del bkg_np, bkg_sp

                            # Daily difference stats (per grid cell)
                            diff = obs - bkg                # same shape: (1080, 1440)
                            # ... Arctic
                            II = np.where((lat > 0.0) & (mask == 1) & (~np.isnan(diff)) & (abs(diff) < 999.9))
                            mae_np_b[iexpt].append(np.mean(np.abs(diff[II])))
                            rmsd_np_b[iexpt].append(np.mean((diff[II])**2))
                            del II
                            # ... Antarctic
                            II = np.where((lat < 0.0) & (mask == 1) & (~np.isnan(diff)) & (abs(diff) < 999.9))
                            mae_sp_b[iexpt].append(np.mean(np.abs(diff[II])))
                            rmsd_sp_b[iexpt].append(np.mean((diff[II])**2))
                            del II
                            del bkg, diff
                        else:
                            if var.upper() == "SSH":
                                # Remove global mean from each grid point
                                arr = bkg.flatten()
                                meanvar = np.nanmean(arr[ind_obs])
                                bkg = bkg - meanvar
                                del meanvar, arr
                            # Compute difference between obs and bkg
                            diff = obs - bkg                # same shape: (1080, 1440)
                            # Apply masking
                            if ocean_choice == "Global":
                                if latlimit == "yes":
                                    II = np.where((abs(lat < 60.0)) & (mask == 1) & (~np.isnan(diff)) & (abs(diff) < 999.9))
                                elif latlimit == "no":
                                    II = np.where((mask == 1) & (~np.isnan(diff)) & (abs(diff) < 999.9))
                            else:
                                II = np.where((mask == 1) & (~np.isnan(diff)) & (abs(diff) < 999.9))
                            # Append stats to arrays
                            sd_b[iexpt].append(np.std(diff[II]))
                            mae_b[iexpt].append(np.mean(np.abs(diff[II])))
                            rmsd_b[iexpt].append(np.mean((diff[II])**2))
                            del bkg, diff, II
                        del exptpath

                del mask

                idd += 1
            imm += 1
        iyy += 1
    # ******************************************************************************************************************
    # END LOOP of days in time range
    # ******************************************************************************************************************

    # ******************************************************************************************************************
    # PLOT TIME SERIES
    # ******************************************************************************************************************

    # ==============================================
    # SST or SSH
    # ==============================================
    if var.upper() == "SST" or var.upper() == "SSH":

        # ------------------------------------------
        # MAE
        # ------------------------------------------

        fig, ax = plt.subplots()

        # ... NCEP Operations (Global SST only)
        if ocean_choice == "Global" and var.upper() == "SST":
            if len(mae_ecm) >= len(nsst_ops):
                ax.plot_date(dates[0:len(nsst_ops)], nsst_ops[0:len(nsst_ops)], '.-',
                             label='NOAA Operations (No ocean model)', color='green')
        # ... ECMWF ORAS5 analysis (SST, SSH only)
        if var.upper() == "SST" or var.upper() == "SSH":
            ax.plot_date(dates[0:len(mae_ecm)], mae_ecm[0:len(mae_ecm)], '.-',
                         label='ORAS5 Replay (ECMWF)', color='black')
        # ... Background forecasts of experiments
        for iexpt in range(len(exptpaths)):
            ax.plot_date(dates[0:len(mae_b[iexpt])], mae_b[iexpt], '.-',
                         label=exptnames[iexpt], color=colors[iexpt])

        legendloc = 'best'
        lines, labels = ax.get_legend_handles_labels()
        if var.upper() == "SST":
            bbox = [0.5, 0.5]
        elif var.upper() == "SSH":
            bbox = [0.5, 0.3]
        ax.legend(lines, labels, loc=legendloc, handlelength=10, bbox_to_anchor=bbox, prop={'size': 6})

        plt.title('Mean Absolute Error (MAE) of ' + str(var.upper()) + ' \nBkg Forecasts vs '
                  + str(reference.upper()) + ': ' + str(ocean_choice.upper()), fontsize=12)
        plt.ylabel('|bkg - ' + str(reference.upper()) + '| (' + str(units) + ')', fontsize=12)
        plt.grid(True)

        axissize = 8
        ax.xaxis.set_tick_params(rotation=30, labelsize=axissize)
        ax.yaxis.set_tick_params(labelsize=axissize)
        ax.autoscale_view()

        plt.savefig(outpath + 'time_series.' + str(var.upper()) + '_MAE.' + str(yyyymmdd_s) + '-' + str(yyyymmdd_e)
                    + '.' + str(ocean_choice.upper()) + '.model_level_' + str(sst_model_level) + '.' + latlimitstr + '.png', dpi=600)

    # ==============================================
    # SEAICE
    # ==============================================
    elif var.upper() == "SEAICE":

        # Set fields to plot
        varstr = [
            var.upper() + "_AREA_FRACTION_MAE",
            var.upper() + "_AREA_EXTENT_MAE",
            var.upper() + "_AREA_EXTENT"
        ]

        # Set polar regions to plot
        pole_choice = ["Arctic", "Antarctic"]

        # Loop through varstr (fields to plot)
        for ivar in range(len(varstr)):
            # Loop through polar regions
            for ipole in range(len(pole_choice)):

                fig, ax = plt.subplots()

                # Determine which pole is being plotted, and which field to plot
                if pole_choice[ipole] == "Arctic":
                    if str(varstr[ivar]).find("FRACTION_MAE") != -1:
                        stat = mae_np_b
                    elif str(varstr[ivar]).find("EXTENT_MAE") != -1:
                        stat = mae_extent_np_b
                    elif str(varstr[ivar]).find("EXTENT") != -1:
                        stat = extent_np_b
                elif pole_choice[ipole] == "Antarctic":
                    if str(varstr[ivar]).find("FRACTION_MAE") != -1:
                        stat = mae_sp_b
                    elif str(varstr[ivar]).find("EXTENT_MAE") != -1:
                        stat = mae_extent_sp_b
                    elif str(varstr[ivar]).find("EXTENT") != -1:
                        stat = extent_sp_b

                # Plot time series of background forecasts of experiments
                for iexpt in range(len(exptpaths)):
                    ax.plot_date(dates[0:len(stat[iexpt])], stat[iexpt], '.-', label=exptnames[iexpt], color=colors[iexpt])

                # Set location of legend box
                if pole_choice[ipole] == "Arctic":
                    bbox = [0.5, 0.8]
                elif pole_choice[ipole] == "Antarctic":
                    bbox = [0.5, 0.3]
                legendloc = 'best'
                lines, labels = ax.get_legend_handles_labels()
                ax.legend(lines, labels, loc=legendloc, handlelength=10, bbox_to_anchor=bbox, prop={'size': 6})

                # Set plot title
                if str(varstr[ivar]).find("_MAE") != -1:
                    title = ('Mean Absolute Error (MAE) of ' + str(varstr[ivar].upper())
                             + ' \nBkg Forecasts vs ' + str(reference.upper()))
                else:
                    title = str(varstr[ivar].upper()) + ': ' + str(pole_choice[ipole].upper())
                plt.title(title, fontsize=12)

                # Set plot axis labels
                if str(varstr[ivar]).find("FRACTION_MAE") != -1:
                    unitstr = units_fraction
                elif str(varstr[ivar]).find("EXTENT") != -1:
                    unitstr = units_extent

                if str(varstr[ivar]).find("_MAE") != -1:
                    label = '|bkg - ' + str(reference.upper()) + '| (' + str(unitstr) + ')'
                else:
                    label = 'Area in ' + str(unitstr)
                plt.ylabel(label, fontsize=12)

                plt.grid(True)

                axissize = 8
                ax.xaxis.set_tick_params(rotation=30, labelsize=axissize)
                ax.yaxis.set_tick_params(labelsize=axissize)
                ax.autoscale_view()

                plt.savefig(outpath + 'time_series.' + str(varstr[ivar].upper()) + '.' + str(yyyymmdd_s) + '-' + str(yyyymmdd_e)
                            + '.' + str(pole_choice[ipole].upper()) + '.png', dpi=600)

    # ******************************************************************************************************************
    # END PLOTS
    # ******************************************************************************************************************

    # ##################################################################################################################
    print("========== END MAIN PROGRAM ==========")


if __name__ == "__main__":
    # Read command line arguments
    if len(sys.argv) < 7:
        print("Usage: python MAIN_PLOT.time_series.MAE.py yyyymmdd_start yyyymmdd_end outpath var reference ocean_choice [HPSS_root]")
        print("  yyyymmdd_start: Start date (YYYYMMDD)")
        print("  yyyymmdd_end: End date (YYYYMMDD)")
        print("  outpath: Output directory path")
        print("  var: Variable (SST, SSH, SEAICE)")
        print("  reference: Reference dataset (OSTIA, COPERNICUS)")
        print("  ocean_choice: Ocean basin (Global, Arctic, Atlantic, Indian, Pacific, Southern)")
        print("  HPSS_root: Optional HPSS archive root path (default: /scratch1/NCEPDEV/global/John.Steffen/hpss_arch/)")
        sys.exit(1)

    yyyymmdd_s = sys.argv[1]
    yyyymmdd_e = sys.argv[2]
    outpath = sys.argv[3]
    var = sys.argv[4]
    reference = sys.argv[5]
    ocean_choice = sys.argv[6]

    # Optional HPSS_root parameter - use provided value or let function use its default
    HPSS_root = sys.argv[7] if len(sys.argv) >= 8 else None
    if HPSS_root is not None:
        plot_time_series_mae(yyyymmdd_s, yyyymmdd_e, outpath, var, reference, ocean_choice, HPSS_root)
    else:
        plot_time_series_mae(yyyymmdd_s, yyyymmdd_e, outpath, var, reference, ocean_choice)
# ##################################################################################################################
# ##################################################################################################################
