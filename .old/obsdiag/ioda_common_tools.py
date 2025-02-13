#!/home/gvernier/anaconda3/bin/python

from netCDF4 import Dataset, num2date, date2num
import numpy as np
import os
import sys
from matplotlib import cm
import re

#------------------------------------------------------------
"""

This module contains all the common functions related to
reading ioda v2 format files.

The functions defined in the code are:
    1. Reading lat lon
    2. Reading data when filename , group name and variable name are available

"""

#------------------------------------------------------------

## get lat and lon
def get_ioda_latlon(filename,lat_name="latitude",lon_name="longitude"):
    ncfile        = Dataset(filename,'r')
    metadata_grp  = ncfile.groups['MetaData']
    lat           = np.squeeze(metadata_grp.variables[lat_name][:]) 
    lon           = np.squeeze(metadata_grp.variables[lon_name][:])
    ncfile.close()
    return lat,lon

# get the data when grpname and varname are provided

def get_var_grp_data(filename, varname,grpname):
    ncfile        = Dataset(filename,'r')
    var_grp       = ncfile.groups[grpname]
    data          = np.squeeze(var_grp.variables[varname][:])
    ncfile.close()
    return data


