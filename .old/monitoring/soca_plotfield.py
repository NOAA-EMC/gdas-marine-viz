#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
from netCDF4 import Dataset, num2date, date2num
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from cmocean import cm as cmo
import os
import yaml
defaultcm = cmo.thermal
from matplotlib import cm
import re
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plothor(x, y, z, map, varname='',
            clim=[-1,1],
            proj_type='global',
            plot_type='contour',
            colormap=defaultcm,
            pretty=False):
    a=float(clim[0])
    b=float(clim[1])
    #print('limit', a, b)
    if  proj_type == 'global' or proj_type == 'hat10':
        proj = ccrs.Robinson()
    if proj_type == 'north':
        proj = ccrs.NorthPolarStereo()
    if proj_type == 'south':
        proj = ccrs.SouthPolarStereo()

    fig = plt.figure(figsize=(13,8))
    #fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    if  proj_type == 'global':
        ax.set_global()
    if  proj_type == 'hat10':
        ax.set_extent([-100, 0, 0, 50], ccrs.PlateCarree())
    if proj_type == 'north':
        ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    if proj_type == 'south':
        ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
    if proj_type=='local':
        ax.set_extent([-80, 30, 50, 87], crs=ccrs.PlateCarree())
    if plot_type == 'contour':
        if abs(a-b)<2: 
            clevs = np.linspace(a, b, 21)
        else:
            clevs = np.linspace(a, b, 41)
        #norm1 = MidpointNormalize(vmin=a, vmax=b, midpoint=0)
        p = ax.contourf(x, y, z, clevs,
                        transform=ccrs.PlateCarree(),
                        cmap=colormap, #norm=norm1,
                        extend='both')

        #line_c = ax.contour(x, y, z, levels=p.levels,
        #                colors=['black'],
        #                transform=ccrs.PlateCarree())
    if plot_type == 'pcolor':
        p = ax.pcolormesh(x, y, z,
                          vmin=clim[0],
                          vmax=clim[1],
                          transform=ccrs.PlateCarree(),
                          cmap=colormap )
    ax.gridlines()
    plt.colorbar(p, ax=ax, shrink=0.5)
    plt.title(varname)
    if pretty == True:
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
        ax.coastlines()

class Grid:
    def __init__(self,fname):
        ncfile = Dataset(fname,'r')
        self.lat=np.squeeze(ncfile.variables['lat'][:])
        self.lon=np.squeeze(ncfile.variables['lon'][:])
        self.mask=np.squeeze(ncfile.variables['mask2d'][:])
        ncfile.close()

def get_var(filename, varname, level=0, aggz=False):
    ncfile = Dataset(filename,'r')
    tmp=np.squeeze(ncfile.variables[varname][:])
    ncfile.close()

    if (aggz):
        tmp=np.sum(tmp[level:,:,:], axis=0)
    else:
        try:
            tmp=tmp[level,:,:]
        except:
            pass

    return tmp

def get_var_zonal(filename, varname, index):
    ncfile = Dataset(filename,'r')
    tmp=np.squeeze(ncfile.variables[varname][:])
    ncfile.close()

    tmp=tmp[:,index,:]

    return tmp

def get_date(filename):
    try:
        yyyymmdd=re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}T[0-9]{2})", filename).group(0)+'Z'
    except:
        yyyymmdd=re.search("([0-9]{4}[0-9]{2}[0-9]{2}[0-9]{2})", filename).group(0)+'Z'
    return yyyymmdd

if __name__ == '__main__':
    description = """Plot field, global and polar stereo:
                     soca_plots -g /path/to/socagrid/soca_gridspec.nc
                                -f /path/to/file/fieldfile.nc
                                -t /path/to/thicknessfile/thicknessfile.nc
                                -s horizontal
                                -y plot.yaml
                  """
    # Command line argument
    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-g',
        '--grid',
        help="soca geometry file, soca_gridspec.nc",
        type=str, required=True)
    parser.add_argument(
        '-f',
        '--file',
        help="soca file containg fields",
        type=str, required=True)
    parser.add_argument(
        '-t',
        '--thicknessfile',
        help="soca file containg the thickness fields (useful when plotting incr)",
        type=str, required=False)
    parser.add_argument(
        '-s',
        '--slice',
        help="slice type: horizontal, meridional, zonal",
        type=str, default="horizontal")
    parser.add_argument(
        '-y',
        '--yaml',
        help="path to yaml file name",
        type=str, required=True)
    parser.add_argument(
        '-p',
        '--pretty',
        help="pretty field plots (land, river ...)",
        type=str, default=True, required=False)

    args = parser.parse_args()

    if (not args.thicknessfile):
        args.thicknessfile=args.file

    # Get date
    try:
        yyyymmdd=get_date(args.file)
    except:
        yyyymmdd="No date"

    # Load Grid
    grid = Grid(fname=args.grid)

    # Arguments parsed from yaml file
    config = yaml.load(open(args.yaml), Loader=yaml.FullLoader)
    args.variable=config["variable"]
    args.clim=[config["clim"]["min"], config["clim"]["max"]]
    args.color=config["color"]
    args.proj=config["projection"]
    args.exp=config["experiment"]
    args.ddir=config["plot_dir"]
    
    head, tail = os.path.split(args.file)
    if (args.slice=="horizontal"):
        args.aggz=config["aggregate"]
        args.level=config["level"]
        var = get_var(filename=args.file, varname=args.variable, level=args.level, aggz=args.aggz)

        plothor(grid.lon, grid.lat, var/grid.mask,
                map,varname=args.variable,
                clim=args.clim,
                proj_type=args.proj,
                colormap=args.color,
                pretty=args.pretty)

        # Yongzuo title & png name
        if args.variable == 'ave_ssh' or args.variable == 'aicen' or args.variable == 'hicen':
             plt.title(args.variable+' '+args.proj+' '+yyyymmdd)
             pngname=args.variable+'.'+args.proj+'.'+yyyymmdd+'.png'
        else:
             plt.title(args.variable+' LEVEL '+str(args.level)+' '+args.proj+' '+yyyymmdd)
             pngname=args.variable+'.'+args.proj+'.'+yyyymmdd+'.png'

        plt.savefig(args.ddir+'/'+pngname,  bbox_inches='tight', pad_inches = 0.02)

    if (args.slice=="zonal"):
        args.lat=config["latitude"]
        args.hname=config["thickness variable"]
        args.maxdepth=config["max depth"]

        index=np.argmin(abs(grid.lat[:,0]-args.lat))
        var = get_var_zonal(args.file, args.variable, index)
        h = get_var_zonal(args.thicknessfile, args.hname, index)
        nlev=np.shape(h)[0]
        y=-np.cumsum(h, axis=0)
        clevs = np.linspace(args.clim[0], args.clim[1], 21)
        x=np.tile(grid.lon[index,:],(nlev,1))
        fig = plt.figure(figsize=(18,6))
        ax = fig.add_subplot(1, 1, 1)
        p = ax.pcolormesh(x, y, var,
                          cmap=plt.get_cmap(args.color),
                          vmin=args.clim[0], vmax=args.clim[1],
                          shading='gouraud')
        clevs = np.linspace(np.min(var[:]), np.max(var[:]), 21)
        p2 = ax.contour(x, y, var, clevs, colors='k',linewidths=0.5)
        plt.colorbar(p, shrink=0.5)
        plt.title(yyyymmdd+' '+args.variable+' lat='+str(args.lat))
        ax.set_ylim((-args.maxdepth, 0))
        if  args.proj == 'hat10':
            #ax.set_extent([-55,-10, 0, 50], ccrs.PlateCarree())
            ax.set_xlim((-55,-15))
        else:
            ax.set_extent([-100, 0, 0, 50], ccrs.PlateCarree())
        pngname=tail+'.'+args.variable+'.'+str(args.lat)+'.'+ str(args.maxdepth)+'.zonal.png'
        plt.savefig(pngname)

