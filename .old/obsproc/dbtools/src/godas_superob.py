#!/usr/bin/env python3
import sys
from r2d2 import fetch
from solo.configuration import Configuration
from solo.date import date_sequence
import yaml
import os
import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, RawTextHelpFormatter
import subprocess
import re

def str2ymdhm(strdate):
        year = strdate[0:4]
        month = strdate[4:6]
        day = strdate[6:8]
        hour = strdate[8:10]
        minute = strdate[10:12]
        return year, month, day, hour, minute

def fetch_obs(args):
    date_start = args.start
    date_end = args.end
    db_step = args.step
    dates = date_sequence(date_start, date_end, db_step)
    provider = args.provider
    experiment = args.experiment
    obs_type = args.obstype

    for date in dates:
        strdate = str(date)
        year, month, day, hour, minute = str2ymdhm(strdate)
        #fout = './superob_tmp/orig/'+obs_type+'_'+year+month+day+hour+minute+'.nc'
        fout = obs_type+'_'+year+month+day+hour+minute+'.nc'
        fetch(
            provider=provider,
            type='ob',
            experiment=experiment,
            database='local',
            date=date,
            obs_type=obs_type,
            time_window=db_step,
            ignore_missing=True,
            target_file=fout,
        )

def dumpconfig(fname, config):
    f = open(fname, 'w')
    yaml.dump(config, f, sort_keys=False, default_flow_style=False)


def config_r2d2(localdb_path):
    # Configure R2D2
    r2d2_config = {'databases': {'archive': {'bucket': 'archive.jcsda',
                                             'cache_fetch': True,
                                             'class': 'S3DB'},
                                 'local': {'cache_fetch': False,
                                           'class': 'LocalDB',
                                           'root': localdb_path},
                                 'shared': {'cache_fetch': False,
                                            'class': 'LocalDB',
                                            'root': ''}},
                   'fetch_order': ['local'],
                   'store_order': ['local']}
    dumpconfig('r2d2_config.yaml', r2d2_config)
    os.environ['R2D2_CONFIG'] = 'r2d2_config.yaml'


def config_superob(yamlname, inputfile, outputfile, gridspec):
    yamlconfig = {'binning': {'errors': {'base': 0.0,
                                         'error mean mult': 1.0,
                                         'value stddev mult': 1.0},
                              'mode': 'superob'},
                  'grid': {'filename': gridspec,
                           'lat': 'lat',
                           'lon': 'lon'},
                  'obs input file': inputfile,
                  'obs output file': outputfile}
    dumpconfig(yamlname, yamlconfig)

if __name__ == '__main__':

    # Command line arguments
    description=''
    parser = ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--start', help='start date in iso format yyyy-mm-ddThh:mm:ssZ', type=str,required=True)
    parser.add_argument('--end', help='end date  in iso format yyyy-mm-ddThh:mm:ssZ', type=str,required=True)
    parser.add_argument('--step', help='duration of the input files in iso somthing (P1D, PT10M, ...)', type=str,default='PT10M')
    parser.add_argument('--provider', help='provider of the processed obervations (jcsda_soca)', type=str,required=True)
    parser.add_argument('--experiment', help='descriptor of the usage type (benchmark_v2)', type=str,required=True)
    parser.add_argument('--obstype', help='type of observations: sst_noaa18, adt_c2, ...', type=str,required=True)
    parser.add_argument('--superobout', help='path to where the superobed file will be saved', type=str,required=True)
    args = parser.parse_args()

    # Get runtime env var
    R2D2_DB = os.getenv('R2D2_DB')
    GRIDSPEC = os.getenv('GRIDSPEC')

    # Where am I?
    cwd = os.getcwd()

    # Configure R2D2
    config_r2d2(localdb_path=R2D2_DB)

    # Fetch 10mn sst obs
    fetch_obs(args)

    # Count files, set the number of workers
    lof=glob.glob(args.obstype+'*.nc')
    lof.sort()
    sojobs=[]
    for f in lof:

        # Create yaml config for superob
        bf = os.path.basename(f)
        ymdhm = re.findall("\d+", bf[-16:])[0]
        year, month, day, hour, minute = str2ymdhm(ymdhm)
        outputpath=os.path.join(args.superobout,year,year+month+day)
        if (not os.path.exists(outputpath)):
            os.makedirs(outputpath)
        yamlconf=bf+'.yaml'
        of=args.obstype+'_so025_'+year+month+day+hour+minute+'.nc'
        config_superob(yamlname=yamlconf, 
                       inputfile=os.path.join(cwd,bf), 
                       outputfile=os.path.join(outputpath, of),
                       gridspec=GRIDSPEC)
        
        # Prepare the superobing job
        # TODO (Guillaume): Send process in background
        #command.append('srun', '-n', '1', SUPEROB_BIN, os.path.join(cwd, yamlconf))
        #t = subprocess.run(command)

        # TODO (G): Can't run multiple instances of srun in SLURM without specifying job arrays
        #           might work on orion desktop?
        #command.append('srun -n 1 '+SUPEROB_BIN+' '+os.path.join(cwd, yamlconf)) 
        #sojobs.append(SUPEROB_BIN+' '+os.path.join(cwd, yamlconf)) 

# Run n superobing jobs in parallel
#n = 12 
#for j in range(max(int(len(sojobs)/n), 1)):
#    procs = [subprocess.Popen(i, shell=True) for i in sojobs[j*n: min((j+1)*n, len(sojobs))] ]
#    for p in procs:
#        p.wait()
