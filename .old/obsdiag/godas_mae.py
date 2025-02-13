#!/home/gvernier/anaconda3/bin/python
from scipy.ndimage import gaussian_filter1d
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import matplotlib
import pickle
import os.path
import datetime
import sys, getopt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# user defined moduels
from ioda_common_tools import *
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

def get_from_pickle(fname):
    if ( os.path.isfile(fname) ):
        print('Loading '+fname, end='\n', flush=True)
        pfile = open( fname, "rb" )
        ts_omb = pickle.load( pfile )
        ts_oma = pickle.load( pfile )
        ts_cntall = pickle.load( pfile )
        ts_cntqc = pickle.load( pfile )
        time = pickle.load( pfile )
        pfile.close()
    else:
        ts_omb = []
        ts_oma = []
        ts_cntall = []
        ts_cntqc = []
        time = []
    return ts_omb, ts_oma, ts_cntall, ts_cntqc, time

def get_obs_stats_from_ioda(path2ioda, obsfname, varname, d0):
  lof=glob.glob(path2ioda+'/ctrl/*'+obsfname+'*.nc')
  if not lof:
    lof=glob.glob(path2ioda+'/'+obsfname+'*.nc')
  if not lof:
    lof=glob.glob(path2ioda+'/ens/'+obsfname+'*.nc')

  if not lof:
    #d=datetime.datetime(2015,1,1,12)
    return [], [], []
#    return omb[I], oma[I], qc[I] #, d

  increment_date=True
  # Loop through multiple ioda files (old experiments have 1 file/cpu)

  for fname in lof:
    lat, lon = get_ioda_latlon(fname)
    try:
        omb= get_var_grp_data(fname, varname,"ombg")
        oma= get_var_grp_data(fname, varname,"oman")
        qc= get_var_grp_data(fname, varname,"EffectiveQC0")
    except:
        o = get_var_grp_data(fname, varname,"ObsValue")
        b = get_var_grp_data(fname, varname,"hofx")
        omb=o-b
        oma=o-b
        qc= get_var_grp_data(fname, varname,"EffectiveQC")

    I=np.where(abs(lat)<90)
    #I=np.where(lon<-80.0)
    return omb[I], oma[I], qc[I] #, d

def main():

  desc = 'Time series of mean absolute error of ioda output. Example: godas_mae.py -e ./geos3dvar -n geos3dvar -y 201*'
  parser = ArgumentParser(
           description=desc,
           formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument(
        '-e', '--experiments', help='Path to experiment(s)',
        type=str, nargs="*", required=True)
  parser.add_argument(
        '-n', '--names', help='Name of experiment(s)',
        type=str, nargs="*", required=True)
  parser.add_argument(
        '-y', '--year', help='Year, can include wild cards',
        type=str, nargs="*", required=False, default='*')
  parser.add_argument(
        '-ms', '--startmonth', help='Start Month, can include wild cards',
        type=str, nargs="*", required=False, default='*')
  parser.add_argument(
        '-me', '--endmonth', help='End Month, can include wild cards',
        type=str, nargs="*", required=False, default='*')
  parser.add_argument(
        '-f', '--filter', help='Gaussian filtering',
        type=int, required=False, default=0)
  helpufovar='UFO variable. One of: \n' + \
                 'absolute_dynamic_topography \n' + \
                 'sea_water_temperature \n' + \
                 'sea_water_salinity \n' + \
                 'sea_surface_temperature \n' + \
                 'sea_surface_salinity \n' + \
                 'sea_ice_area_fraction \n' + \
                 'sea_ice_category_thickness \n' + \
                 'sea_ice_freeboard \n'
  parser.add_argument('-v', '--ufo_var', help=helpufovar, type=str, required=True)
  parser.add_argument(
        '-i', '--instrument', help='Instrument (adt_c2, adt_coperl4, ...)',
        type=str, required=True)
  args = parser.parse_args()

  yyyy = args.year[0]
  ms = args.startmonth[0]
  me = args.endmonth[0]
  loe = args.experiments
  listofnames = args.names

  obsname = args.ufo_var
  inst=args.instrument
  color=['g','r','c','y','k','g','r','k','m','y','c']
  t=0
  fig, ax = plt.subplots(figsize=(16, 10))
  axr = ax.twinx()
  # Loops through experiments
  print('========================================== '+obsname, end='\n', flush=True)
  expindex=0
  for exp in loe:
      expname=listofnames[expindex]
      print('--- '+expname, end='\n', flush=True)
      expindex+=1
      im=int(ms)
      files=[]
 
      while im <= int(me):
         #lod=glob.glob(exp+'/obs_out/'+yyyy+'/'+yyyy+mm+'*')
          if im == int(ms):
              if im < 10:
                  mm="0"+str(im)
              else:
                  mm=str(im)
              print(mm)
              lod=glob.glob(exp+'/obs_out/'+yyyy+'/'+yyyy+mm+'*')
              im=im+1
          else:
              if im < 10:
                  mm="0"+str(im)
              else:
                  mm=str(im)
              lod.extend(glob.glob(exp+'/obs_out/'+yyyy+'/'+yyyy+mm+'*'))
              im=im+1
         #print(exp+'/obs_out/'+yyyy+'/'+yyyy+mm+'*')
         #if not lod:
         #    lod=glob.glob(exp+'/'+yyyy+mm+'*')
      lod.sort()

      # Try to read previously computed stats from pickle file
      picklefname=expname+'.'+obsname+'.'+inst+'.pkl'
      ts_omb, ts_oma, ts_cntall, ts_cntqc, time = get_from_pickle(picklefname)

      # Get starting index
      index=0
      for ymdh in lod:
          if not time:
              break
          pickleymdh=int(str(time[-1].year).zfill(4)+str(time[-1].month).zfill(2)+str(time[-1].day).zfill(2)+str(time[-1].hour).zfill(2))
          if (int(ymdh[-10:])==pickleymdh):
              break
          index+=1

      # Loop through time
      for path2ioda in lod[index+1:]:

        # Get stats for the cycle
        if (len(time)>0):
            d0 = time[-1]
        else:
            d0 = datetime.datetime(1900,1,1,0)
        omb, oma, qc = get_obs_stats_from_ioda(path2ioda, inst, obsname, d0)
        ymd=path2ioda[-10:]

        d = datetime.datetime(year=int(ymd[0:4]),month=int(ymd[4:6]),day=int(ymd[6:8]),hour=int(ymd[8:10]))

        # Filter out stuff and append
        if (len(omb)>0):

            I=np.where( abs(omb)<=20.0 )
              #I=np.where( qc==0 )
              #I=np.where( abs(lat<30) )
            #I=np.where( np.logical_and( (qc<=1), (abs(omb)<=50.0)) )
            ts_omb.append(np.mean(np.abs(omb[I])))
            ts_oma.append(np.mean(np.abs(oma[I])))
            Iqc=np.where( qc==0 )
            ts_cntall.append(len(oma))
            ts_cntqc.append(len(oma[Iqc]))
            time.append(d)
      print(ts_omb)

      # Save omb/oma's for exp/obs type at every cycle (why?)
      pfile = open( picklefname, "wb" )
      pickle.dump( ts_omb,  pfile )
      pickle.dump( ts_oma,  pfile )
      pickle.dump( ts_cntall,  pfile )
      pickle.dump( ts_cntqc,  pfile )
      pickle.dump( time,  pfile )
      pfile.close()

      # Plot stats for current experiment
      ts_omb = np.array(ts_omb)
      ts_oma = np.array(ts_oma)
      ts_cntall = np.array(ts_cntall)
      ts_cntqc = np.array(ts_cntqc)

      #I=np.where(ts_omb>2.0)
      #ts_omb[I]=np.nan
      #ts_oma[I]=np.nan
      if (args.filter>0):
        plt.plot_date(time,gaussian_filter1d(ts_omb,args.filter),'-',color=color[t],linewidth=3,label=expname)
        #plt.plot_date(time,gaussian_filter1d(ts_oma,args.filter),'--',color=color[t])
      else:
        ax.plot_date(time,ts_omb,'-*',color=color[t],linewidth=1,label=expname)
        #ax.plot_date(time,ts_oma,'--',color=color[t],linewidth=0.5,label=expname)
        #ax.plot_date(time,ts_oma,'--*',color=color[t])
        axr.set_ylabel('% total obs')
        ax.set_ylabel('<|Obs-bkg|>')
        #axr.plot_date(time,ts_cntall,'--',color=color[t],alpha=0.1)
        axr.plot_date(time,ts_cntqc/ts_cntall,'-',color=color[t],alpha=0.2,)
      if (t<10):
          t+=1
      else:
          t=0
      ax.xaxis.set_tick_params(rotation=30, labelsize=16)
      #ax.set_xlim([datetime.date(2015, 1, 1), datetime.date(2015, 3, 31)])
      ax.autoscale_view()

  ax.legend()
  plt.title(obsname+' '+inst)
  plt.grid(True)
  plt.savefig(obsname+'_'+inst+'.png')
  plt.clf()

if __name__ == '__main__':
    main()
