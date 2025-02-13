'''
Description:
    This python script can be used to produce an overview of the database inventory, 
    such as the time period when data is available and when data is missing.  

input:
    start date (for P1D: yyyymmdd, for PT10M: yyyymmddhhmn)
    end date (for P1D: yyyymmdd, for PT10M: yyyymmddhhmn)
    yaml file (contains step, r2d2 dir and list of observation)
output:
    will produce a figure that shows when observation is available and a gap when data are missing
'''

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
import plotly.express as px
import pandas as pd
import plotly.io as pio
import yaml

class ObsInventory:
    def __init__(self,args=None):
        self.folder = args.path 
        self.start_date = args.start
        self.end_date = args.end
        self.year=self.start_date[0:4]
        self.args=args
    # ---------- P1D database ----------------- 
    def inventory_p1d(self, folder=None):
        '''
        For missing file: ind=0
        and for existing: ind=1
        This index is used for plotting
        '''
        if folder== None:
            folder=self.folder
        start_date=datetime.strptime(self.start_date, '%Y%m%d').date()
        end_date=datetime.strptime(self.end_date, '%Y%m%d').date()
        plf_avail_date={}
        for file_descriptor in self.args.file_descriptor:
            #print('des',file_descriptor)
            current_date=start_date
            while current_date < end_date:
                ddir=folder+'/'+str(current_date)
                list_of_files=glob(ddir+'/*%s*.nc*'%(file_descriptor))
                if not list_of_files:
                    ind=0
                else:
                    ind=1
                plf_avail_date.setdefault(file_descriptor, []).append([ind, current_date])                    
                current_date=current_date+timedelta(days=1)
            plf_avail_date.setdefault(file_descriptor, []).append([0, end_date])                    
        self.plf_avail_date=plf_avail_date
        self.plot_p1d(plf_avail_date)
    
    def plot_p1d(self, plf_data):
        '''
        is_key is used to find the total length of time when data is available, 
	like start date and end date. Initially, is_key=1 is setup when start date 
	(first instance of ind[i]=1) is picked and put is_key=0. 
        Do not update is_key until ind[i]=0 is found, at which the end_date is 
        picked and put is_key=1
        '''
        listd=[]
        for plf in plf_data.keys():
            data=plf_data[plf]
            data=np.array(data)
            ind=data[:,0]
            avail_dates=data[:,1]
            is_key=1
            current=avail_dates[0]
            end_date=avail_dates[-1]
            start=current
            end=current
            i=0
            while current<end_date:
                current=avail_dates[i]
                if ind[i]==1:
                    if is_key==1:
                        start=current
                        is_key=0
                if ind[i]==0:
                    if is_key==0:
                        end=current
                        is_key=1
                        if (end-start).days >2:
                            listd.append(dict(File_descriptor="%s"%plf, Start='%s'%start, Finish='%s'%end))
                        start=avail_dates[0]
                        end=avail_dates[0]
                i=i+1 
            if (end-start).days >2:
                listd.append(dict(Obs_type="%s"%plf, Start='%s'%start, Finish='%s'%end))

        df= pd.DataFrame(listd)
        #print('data frame', listd)
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="File_descriptor", 
            facet_row_spacing=.1, template="simple_white",
            title="gdas_marine s2s_v1 P1D files")
        fig.update_traces(width=0.7, marker=dict(color='lightgrey', 
            line=dict(width=1,  color='green')))
        fig.update_yaxes(showgrid=True, linewidth=2, mirror=True, title_standoff = 5)
        fig.update_xaxes(showgrid=True, linewidth=2, mirror=True)
        fig.update_layout(title_x=0.55, yaxis_title=None, margin=dict(l=20, r=20, t=25, b=20))
        pio.write_image(fig, 'Fig_inventory_P1D_%s-%s.png'%(self.start_date[:4], self.end_date[:4]), format='png')

    # ------------ SST 10 min database ---------------
    # ------------------------------------------------
    def inventory_pt10m(self, folder=None):
        '''
        For missing file: ind=0
        and for existing: ind=1
        This index is used for plotting
        '''
        if folder== None:
            folder=self.folder
        start_date=datetime.strptime(self.start_date, '%Y%m%d%H%M')
        end_date=datetime.strptime(self.end_date, '%Y%m%d%H%M')
        
        plf_avail_date={}
        for file_descriptor in self.args.file_descriptor:
            current_date=start_date
            while current_date < end_date:
                ddir=folder+'/%s'%current_date.date()
                list_of_files=glob(ddir+'/*%s*%s*.nc*'%(file_descriptor, current_date.time()))
                if not list_of_files:
                    ind=0
                else:
                    ind=1
                plf_avail_date.setdefault(file_descriptor, []).append([ind, current_date])                    
                current_date=current_date+timedelta(minutes=10)
            plf_avail_date.setdefault(file_descriptor, []).append([0, end_date])                    
        self.plf_avail_date=plf_avail_date
        self.plot_pt10m(plf_avail_date)
        
    def plot_pt10m(self, plf_data):
        '''
        is_key is used to find the total length of time when data is available, 
	like start date and end date. Initially, is_key=1 is setup when start date 
	(first instance of ind[i]=1) is picked and put is_key=0. 
        Do not update is_key until ind[i]=0 is found, at which the end_date is 
        picked and put is_key=1
        '''
        listd=[]
        for plf in plf_data.keys():
            data=plf_data[plf]
            data=np.array(data)
            ind=data[:,0]
            avail_dates=data[:,1]
            is_key=1
            current=avail_dates[0]
            end_date=avail_dates[-1]
            start=current
            end=current
            i=0
            while current<end_date:
                current=avail_dates[i]
                if ind[i]==1:
                    if is_key==1:
                        start=current
                        is_key=0
                if ind[i]==0:
                    if is_key==0:
                        end=current
                        is_key=1
                        if (end-start)/timedelta(minutes=1) >10:
                            listd.append(dict(File_descriptor="%s"%plf, Start='%s'%start, Finish='%s'%end))
                        start= avail_dates[i]
                        end  = avail_dates[i]
                i=i+1 
            if (end-start)/timedelta(minutes=1) >10:
                listd.append(dict(Obs_type="%s"%plf, Start='%s'%start, Finish='%s'%end))

        df= pd.DataFrame(listd)
        print('data frame', listd)
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="File_descriptor", 
            facet_row_spacing=.1, template="simple_white",
            title="gdas_marine s2s_v1 PT10M files")
        fig.update_traces(width=0.5, marker=dict(color='lightgrey', 
            line=dict(width=1,  color='green')))
        fig.update_yaxes(showgrid=True, linewidth=2, mirror=True, title_standoff = 5)
        fig.update_xaxes(showgrid=True, linewidth=2, mirror=True)
        fig.update_layout(title_x=0.55, yaxis_title=None, margin=dict(l=20, r=20, t=25, b=20))
        #fig.update_layout(yaxis=dict(ticktext=self.args.file_descriptor))
        pio.write_image(fig, 'Fig_inventory_PT10M_%s-%s.png'%(self.start_date[:6], self.end_date[:6]), format='png')

  
if __name__ == '__main__':
    description = """ Ex: python godas_inventory_plot.py -s start -e end -y yaml_file
                      Suggestions: for PT10M, limit the range between 1-2 years
                  """
    # Command line argument
    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s','--start',
        help="start date (yyyymmdd, yyyymmddhhmn)", type=str, required=True)
    parser.add_argument( '-e','--end',
        help="end date (yyyymmdd, yyyymmddhhmn)",type=str, required=True)
    parser.add_argument('-y','--yaml',
        help="Provide yaml file that contains step, r2d2 dir, obs_type", type=str, required=True)

    args = parser.parse_args()
    if args.yaml is not None:
        with open(args.yaml, 'r') as yaml_file:
            config=yaml.safe_load(yaml_file)
        args.file_descriptor=config['obs_types']
        args.path=config['obs_dir']
        args.step=config['step']
    #print(args)
    obj_inventory=ObsInventory(args=args)

    #obj_inventory.inventory_barplot(folder=args.path)
    if len(args.start)==8 and args.step=='P1D': 
        obj_inventory.inventory_p1d(folder=args.path)

    elif len(args.start)>8 and args.step == 'PT10M':
        obj_inventory.inventory_pt10m(folder=args.path)

    else:
        sys.exit('inconsistent information. Please check your date and yaml file')


