#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
import socket
import yaml
import numpy as np
from netCDF4 import Dataset

# sets the cmap vmin/vmax for each variable
# TODO: this should probably be in a yaml or something
# ... For OmB, OmA
vminmax = {'seaSurfaceTemperature': {'vmin': -2.0, 'vmax': 2.0},
           'seaIceFraction': {'vmin': -0.2, 'vmax': 0.2},
           'seaSurfaceSalinity': {'vmin': -0.2, 'vmax': 0.2},  # TODO: this should be changed
           'absoluteDynamicTopography': {'vmin': -0.2, 'vmax': 0.2},
           'waterTemperature': {'vmin': -2.0, 'vmax': 2.0},
           'salinity': {'vmin': -0.2, 'vmax': 0.2}}
# ... For hofx
vminmax_hofx = {'seaSurfaceTemperature': {'vmin': -5.0, 'vmax': 30.0},
                'seaIceFraction': {'vmin': 0.0, 'vmax': 1.0},
                'seaSurfaceSalinity': {'vmin': 0.0, 'vmax': 50.0},  # TODO: this should be changed
                'absoluteDynamicTopography': {'vmin': 0.0, 'vmax': 1.0},
                'waterTemperature': {'vmin': -5.0, 'vmax': 30.0},
                'salinity': {'vmin': 0.0, 'vmax': 50.0}}
# ... For obs errors
vminmax_error = {'seaSurfaceTemperature': {'vmin': 0.0, 'vmax': 1.0},
                 'seaIceFraction': {'vmin': 0.0, 'vmax': 0.2},
                 'seaSurfaceSalinity': {'vmin': 0.0, 'vmax': 1.0},  # TODO: this should be changed
                 'absoluteDynamicTopography': {'vmin': 0.0, 'vmax': 1.0},
                 'waterTemperature': {'vmin': 0.0, 'vmax': 1.0},
                 'salinity': {'vmin': 0.0, 'vmax': 1.0}}


def marine_eva_post(inputyaml, outputdir, diagdir):
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        with open(inputyaml, 'r') as inputyaml_opened:
            input_yaml_dict = yaml.safe_load(inputyaml_opened)
        logging.info(f'Loading input YAML from {inputyaml}')
    except Exception as e:
        logging.error(f'Error occurred when attempting to load: {inputyaml}, error: {e}')
        return
    for dataset in input_yaml_dict['datasets']:
        # Get filenames
        newfilenames = []
        for filename in dataset['filenames']:
            newfilename = os.path.join(diagdir, os.path.basename(filename))
            newfilenames.append(newfilename)
        dataset['filenames'] = newfilenames
        # Get group names of fields
        newgroupnames = []
        for groupname in dataset['groups']:
            gnames = groupname['name']
            newgroupnames.append(gnames)

    # Set vmin/vmax based on yaml and var_min/var_max
    for graphic in input_yaml_dict['graphics']['figure_list']:
        # This assumes that there is only one variable, or that the
        # variables are all the same
        variable = graphic['batch figure']['variables'][0]

        for plot in graphic['plots']:
            for layer in plot['layers']:
                if layer['type'] == 'MapScatter':
                    # Initialize some variables
                    var_min = -999
                    var_max = -999
                    dynopt = graphic['dynamic options']
                    dynstr = ""
                    # This loop is only used for the ObsTime map plot: to find var_min/var_max of "dateTime"
                    for tdyn in dynopt:
                        sdyn = str(tdyn)
                        dyn = ""
                        # String together "dynamic options" line from yaml
                        for ss in sdyn:
                            dyn += ss
                        # Loop through group names of fields to find "metadata" (for "dateTime" plot)
                        for tname in newgroupnames:
                            invar = ""
                            if str(tname).lower().find('metadata') != -1 and str(dyn).lower().find('obstime') != -1:
                                invar = "dateTime"
                                dynstr = str(tname)
                                # Open diag file to get "dateTime" variable
                                ds = Dataset(dataset['filenames'][0], mode="r")
                                vardata = ds.groups[tname].variables[invar]
                                # Find min/max dateTime (in seconds from epoch)
                                tmin = np.nanmin(vardata)
                                tmax = np.nanmax(vardata)
                                var_min = int(np.round(tmin, decimals=-1))
                                var_max = int(np.round(tmax, decimals=-1))
                                # Close diag file
                                ds.close()
                                break
                    # Set min/max for map plot
                    if dynstr.lower().find('omb') != -1 or dynstr.lower().find('oma') != -1:
                        layer['vmin'] = vminmax[variable]['vmin']
                        layer['vmax'] = vminmax[variable]['vmax']
                    elif dynstr.lower().find('hofx') != -1:
                        layer['vmin'] = vminmax_hofx[variable]['vmin']
                        layer['vmax'] = vminmax_hofx[variable]['vmax']
                    elif dynstr.lower().find('error') != -1:
                        layer['vmin'] = vminmax_error[variable]['vmin']
                        layer['vmax'] = vminmax_error[variable]['vmax']
                    elif dynstr.lower().find('metadata') != -1:
                        layer['vmin'] = var_min
                        layer['vmax'] = var_max

    # first, let us prepend some comments that tell someone this output YAML was generated
    now = datetime.datetime.now()
    prepend_str = ''.join([
        '# This YAML file automatically generated by marine_eva_post.py\n',
        f'# on {socket.gethostname()} at {now.strftime("%Y-%m-%dT%H:%M:%SZ")}\n',
    ])

    outputyaml = os.path.join(outputdir, os.path.basename(inputyaml))
    # open output file for writing and start the find/replace process
    try:
        logging.info(f'Writing modified YAML to {outputyaml}')
        with open(outputyaml, 'w') as yaml_out:
            yaml_out.write(prepend_str)
            yaml.dump(input_yaml_dict, yaml_out)
    except Exception as e:
        logging.error(f'Error occurred when attempting to write: {outputyaml}, error: {e}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputyaml', type=str, help='Input YAML to modify', required=True)
    parser.add_argument('-o', '--outputdir', type=str, help='Directory to send output YAML', required=True)
    parser.add_argument('-d', '--diagdir', type=str, help='Location of diag files', required=True)
    args = parser.parse_args()
    marine_eva_post(args.inputyaml, args.outputdir, args.diagdir)
