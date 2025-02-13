from netCDF4 import Dataset, num2date, date2num
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime, timedelta
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import sys
from glob import glob

class ObsSanityCheck:
    def __init__(self,args=None):
        self.folder = args.path 
        self.start_date = args.start
        self.end_date = args.end
        self.year=self.start_date[0:4]
        self.args=args

    def extract_info_minutes(self, folder=None):
        if folder is None:
            folder=self.folder
        file_descriptor=self.args.file_descriptor
        start_date=datetime.strptime(self.start_date, '%Y%m%d%H%M')
        end_date=datetime.strptime(self.end_date, '%Y%m%d%H%M')
        current_date=start_date
        missing_hours=[]
        nlocs=[]
        avail_dates=[]
        minval=[]
        maxval=[]
        while current_date < end_date:
            tmdhm=datetime.strftime(current_date, '%Y%m%d%H%M')
            ddir=folder+'/'+tmdhm[0:8]
            list_of_files=glob(ddir+'/*%s*%s*.nc*'%(file_descriptor, tmdhm))
            if not list_of_files:
                missing_hours.append(tmdhm[8:])
            else:
                nloc, minn, maxx=self.find_param(list_of_files[0])
                nlocs.append(nloc)
                minval.append(minn)
                maxval.append(maxx)
                if nloc==0: 
                    missing_hours.append(tmdhm[8:])
                avail_dates.append(current_date)
            current_date=current_date+timedelta(minutes=10)
        self.plot_info(nlocs, minval, maxval, avail_dates, missing_hours, freq='hours')

    def extract_info_daily(self, folder=None):
        file_descriptor=self.args.file_descriptor
        if folder== None:
            folder=self.folder
        start_date=datetime.strptime(self.start_date, '%Y%m%d').date()
        end_date=datetime.strptime(self.end_date, '%Y%m%d').date()
        current_date=start_date
        missing_dates=[]
        nlocs=[]
        avail_dates=[]
        minval=[]
        maxval=[]
        while current_date < end_date:
            ddir=folder+'/'+str(current_date)
            list_of_files=glob(ddir+'/*%s*.nc*'%file_descriptor)
            if not list_of_files:
                missing_dates.append(current_date)
            else:
                nloc, minn, maxx=self.find_param(list_of_files[0])
                nlocs.append(nloc)
                minval.append(minn)
                maxval.append(maxx)
                avail_dates.append(current_date)
            current_date=current_date+timedelta(days=1)
        self.plot_info(nlocs, minval, maxval, avail_dates, missing_dates, freq='dates')

    def plot_info(self, nlocs, minval, maxval, avail_dates, missing_dates, freq=None):        
        
        file_descriptor=self.args.file_descriptor
        formatter = mdates.DateFormatter('%Y/%m/%d')
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(2, 2)
        
        # --- plot number of observations ---
        ax = fig.add_subplot(gs[0, 0])
        nlocs=[xx/1000 for xx in nlocs]
        ax.plot_date(avail_dates, nlocs,'-og')
        ax.title.set_text('No of observations')
        ax.set_ylabel('No of obs')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.set_xticklabels([])
        if len(nlocs) != 0:
            ytkl= np.linspace(min(nlocs), max(nlocs), 5)
            yticks=['%sk'%int(xx) for xx in ytkl]
            ax.set_yticks(ytkl)
            ax.set_yticklabels(yticks)

        # --- Plot min/max value of a variable
        ax = fig.add_subplot(gs[1, 0])
        plt.title('Min and Max value of a variable')
        plt.plot_date(avail_dates, minval,'-og', label='Min')
        plt.plot_date(avail_dates, maxval,'-*r', label='Max')
        ax.set_ylabel('Min/Max')
        ax.set_xlabel('Dates')
        ax.legend(loc=3)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.set_xticklabels([])

        # --- Write missing files ----
        ax=fig.add_subplot(gs[:,1])
        md_len=len(missing_dates)
        plt.title('Missing files (%s): %s'%(md_len, freq))
        if md_len == 0:
            plt.text(.12, .75, 'No missing files!', fontsize=13)
        else:
            xstep=0.01
            ystep=0.95
            for tx in missing_dates:
                plt.text(xstep, ystep, '%s'%tx, fontsize=8)
                ystep=ystep-0.05
                if ystep < .005: 
                    xstep=xstep+0.33
                    ystep=0.95
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.suptitle("%s: %s-%s"%(file_descriptor, self.start_date, self.end_date), y=.96, fontsize=14);
        plt.savefig('Fig_%s_%s.png'%(file_descriptor, self.year), bbox_inches='tight', pad_inches=0.1)

    # --- extract no of observations and min/max value of a varibale -----

    def find_param(self, fname):
        args=self.args
        ncfile = Dataset(fname,'r')
        metadata_grp=ncfile.groups['MetaData']
        var_grp=ncfile.groups['ObsValue']
        var_name=list(var_grp.variables)
        var_tmp=np.squeeze(var_grp.variables[var_name[0]][:])
        nlocs=len(var_tmp)
        minval=0
        maxval=0
        if nlocs != 0:
            minval=min(var_tmp)
            maxval=max(var_tmp)
        ncfile.close()
        return nlocs,  minval, maxval
        
if __name__ == '__main__':
    description = """ Ex: godas_obssanitycheck.py -f adt_c2 -s start -e end -q frequency
                      Suggestions: limit the range between 3-6 months for daily data
                                for hourly or minutes data limit a day to few days       
                  """
    # Command line argument
    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file_descriptor',
        type=str, required=True)
    parser.add_argument('-s','--start',
        help="start date (yearmmdd, yearmmddhhmn, ...)",
        type=str, required=True)
    parser.add_argument( '-e','--end',
        help="end date in the format yearmmdd",type=str, required=True)
    parser.add_argument('-p','--path',
        help="path of the data dir", type=str, required=False)
    parser.add_argument('-q', '--frequency', default='daily',
        help="frequency for daily or minutes data", type=str, required=False)

    args = parser.parse_args()
    obj_sanity=ObsSanityCheck(args=args)

    if args.frequency=='daily':
        if len(args.start) != 8:
            sys.exit('default frequency is daily. Please provide date as %Y%m%d or provide frequency for hourly/minutes')
        if args.path == None:        
            args.path='/work/noaa/marine/marineda/r2d2/obs/jcsda_soca/ob/benchmark_v2/P1D'
        obj_sanity.extract_info_daily(folder=args.path)

    if args.frequency=='minutes' or args.frequency=='hourly':
        if len(args.start) <= 8:
            sys.exit('Please provide date as %Y%m%d%hh%mm')
        if args.path == None:
            args.path='/work/noaa/marine/Guillaume.Vernieres/OBS/sprint1of4/convert2ioda/scratch/ioda_data'
        obj_sanity.extract_info_minutes(folder=args.path)


