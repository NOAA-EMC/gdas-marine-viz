#!/usr/bin/env python3
import sys
from r2d2 import store
from solo.configuration import Configuration
from solo.date import date_sequence
import yaml
from yaml.loader import SafeLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, RawTextHelpFormatter
import os

def obssource(source_dir, year, month, day, obs_type, hour=[], minute=[], ext='.nc', fmttype='ymdhm'):
    # TODO: This is clumsy ... rework that differently. Add exception too ...
    obspath=source_dir
    if fmttype=='ymdhm':
        obspath=os.path.join(obspath, year+month+day,obs_type+'_'+year+month+day+hour+minute+'.'+ext)
    if fmttype=='ymd':
        obspath=os.path.join(obspath, year, year+month+day,obs_type+'_'+year+month+day+'.'+ext)

    return obspath

def store_obs(yaml_file):
    config = Configuration(yaml_file)
    dates = date_sequence(config.start, config.end, config.step)
    obs_types = config.obs_types
    provider = config.provider
    experiment = config.experiment
    type = config.type
    source_dir = config.source_dir
    fmt = config.source_file
    ext = config.source_file_ext
    step = config.step
    database= config.database

    for date in dates:
        strdate = str(date)
        year = strdate[0:4]
        month = strdate[4:6]
        day = strdate[6:8]
        hour = strdate[8:10]
        minute = strdate[10:12]

        for obs_type in obs_types:
            obssrc=obssource(source_dir, year, month, day, obs_type, hour, minute, ext=ext, fmttype=fmt)
            store(
                provider=provider,
                type=type,
                experiment=experiment,
                database=database,
                date=date,
                obs_type=obs_type,
                time_window=step,
                source_file=obssrc,
                ignore_missing=True,
            )

if __name__ == '__main__':
    description = \
    'Example: \n' + \
    './soca_store_obs.py soca_store_obs.py --start 2019-01-01T00:00:00Z \n' + \
    '                                      --end 2019-01-01T01:00:00Z \n' + \
    '                                      --source_dir ./ioda_data/ \n' + \
    '                                      --source_file ymdhm' + \
    '                                      --source_file_ext nc' + \
    '                                      --provider gdas_marine \n' + \
    '                                      --experiment s2s_v1 \n' + \
    '                                      --obstype sst \n' + \
    '                                      --platforms noaa18_l3u \n' + \
    '                                      --storage local \n' + \
    '                                      --step PT10M \n' + \
    'List of observation types and platforms (not complete by a long shot!): \n' + \
    '  - adt: 3a, c2, j2, j3, sa, ... \n' + \
    '  - icec: EMC, ... \n' + \
    '  - icefb: GDR, ... \n' + \
    '  - insitu: hgodas, fnmoc, ... \n' + \
    '  - oc: EMC \n' + \
    '  - sst: noaa19, metopa, gmi, windsat, drifter, ... \n' + \
    '  - sss: smap, smos, aquarius, ... \n'

    # Command line arguments
    parser = ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--start', help='start date in iso format yyyy-mm-ddThh:mm:ssZ', type=str,required=True)
    parser.add_argument('--end', help='end date  in iso format yyyy-mm-ddThh:mm:ssZ', type=str,required=True)
    parser.add_argument('--source_dir', help='path to the original observations in ioda format', type=str,required=True)
    parser.add_argument('--source_file', help='a description of how the path to the file is formed. Only 2 options: ymd or ymdh', type=str, default='ymd')
    parser.add_argument('--source_file_ext', help='source file name extension', type=str, default='nc')
    parser.add_argument('--provider', help='provider of the processed obervations (jcsda_soca)', type=str,required=True)
    parser.add_argument('--experiment', help='descriptor of the usage type (benchmark_v2)', type=str,required=True)
    parser.add_argument('--obstype', help='type of observations: sst, sss, adt, ...',
                        type=str,required=True)
    parser.add_argument('--platforms', help='a descriptor of the platform of the observation type',
                        type=str, nargs='+',required=True)
    parser.add_argument('--step', help='duration of the input files in iso somthing (P1D, PT10M, ...)',
                        type=str,default='P1D')
    parser.add_argument('--shared_db', help='path to the shared database (/work/noaa/marine/marineda/r2d2/obs/ on Orion)',
                        type=str, default='./r2d2-shared/')
    parser.add_argument('--local_db', help='path to the local database (use it for testing)',
                        type=str, default='./r2d2-local/')
    parser.add_argument('--storage', help='local or shared',
                        type=str, default='local')

    args = parser.parse_args()

    # Assemble type/platforms
    tp=[]
    for p in args.platforms:
        tp.append(args.obstype+'_'+p)

    # Configure R2D2
    r2d2_config = {'databases': {'archive': {'bucket': 'archive.jcsda',
                                             'cache_fetch': True,
                                             'class': 'S3DB'},
                                 'local': {'cache_fetch': False,
                                           'class': 'LocalDB',
                                           'root': args.local_db},
                                 'shared': {'cache_fetch': False,
                                            'class': 'LocalDB',
                                            'root': args.shared_db}},
                   'fetch_order': ['shared'],
                   'store_order': [args.storage]}
    
    f = open('r2d2_config.yaml', 'w')
    yaml.dump(r2d2_config, f, sort_keys=False, default_flow_style=False)
    os.environ['R2D2_CONFIG'] = 'r2d2_config.yaml'

    # Create R2D2 database storage configuration
    obsstore = {'start': args.start,
                'end': args.end,
                'step': args.step,
                'source_dir': args.source_dir,
                'source_file': args.source_file,
                'source_file_ext': args.source_file_ext,
                'type': 'ob',
                'provider': args.provider,
                'experiment': args.experiment,
                'obs_types': tp,
                'database': args.storage}
    f = open('store_obs.yaml', 'w')
    yaml.dump(obsstore, f, sort_keys=False, default_flow_style=False)

    # Store obs in R2D2
    store_obs('store_obs.yaml')
