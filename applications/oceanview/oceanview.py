#!/usr/bin/env python
import matplotlib as mpl
import pylab as pl
from pylab import get_current_fig_manager as gcfm
import wx
import numpy as np
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import re

mpl.use('WXAgg')
mpl.interactive(False)

COLORS = ['b', 'r', 'g', 'm', 'k', 'y', 'c', 'orange', 'purple', 'lime', 'brown', 'pink']


class Instrument:
    """
    Represents a simple data structure for an instrument

    Attributes:
        name (str): The name of the instrument.
        instid (int): The instrument ID.
        varid (int): The variable ID.
        zmin (float): The minimum depth.
        zmax (float): The maximum depth.
        color (str, optional): The color of the instrument. Defaults to 'k' (black).
        proj (list, optional): The projection(s) for the instrument. Defaults to ['global'].
    """

    def __init__(self, name, instid, varid, zmin, zmax, color='k', proj=['global']):
        self.name = name
        self.instid = instid      # Instrument ID
        self.varid = varid        # Var ID
        self.zmin = zmin
        self.zmax = zmax
        self.color = color
        self.proj = proj


class VarSpecs:
    def __init__(self, varid, bounds, fignum, units):
        self.varid = varid
        self.bounds = bounds
        self.fignum = fignum
        self.units = units


dict_inst = {'insitu_profile_xbtctd'   :
             Instrument(name='XBT_CTD', instid=504, varid=np.array([101, 102]), zmin=0, zmax=2000),
             'insitu_profile_glider'   :
             Instrument(name='Gliders', instid=504, varid=np.array([101, 102]), zmin=0, zmax=2000),
             'insitu_profile_tropical'   :
             Instrument(name='Tropical_Moorings', instid=505, varid=np.array([101, 102]), zmin=0, zmax=2000),
             'insitu_profile_argo'   :
             Instrument(name='Argo', instid=506, varid=np.array([101, 102]), zmin=0, zmax=2000),
             'insitu_temp_profile_argo'   :
             Instrument(name='Argo', instid=507, varid=np.array([101]), zmin=0, zmax=2000),
             'insitu_surface_drifter'   :
             Instrument(name='drifter', instid=508, varid=np.array([101, 102]), zmin=0, zmax=2000),
             'insitu_profile_bathy'   :
             Instrument(name='bathy', instid=509, varid=np.array([101, 102]), zmin=0, zmax=2000),
             'insitu_profile_tesac'   :
             Instrument(name='tesac', instid=510, varid=np.array([101, 102]), zmin=0, zmax=2000),
             'insitu_profile_tesac_salinity'   :
             Instrument(name='tesac salt', instid=511, varid=np.array([102]), zmin=0, zmax=2000),
             'insitu_surface_trkob'   :
             Instrument(name='trkob', instid=512, varid=np.array([102]), zmin=0, zmax=2000),
             'insitu_surface_trkob_salinity'   :
             Instrument(name='trkob salt', instid=513, varid=np.array([102]), zmin=0, zmax=2000),
             }


class ioda:
    """
    A class representing the ioda object.

    Attributes:
        lon (list): List of longitudes.
        lat (list): List of latitudes.
        oma (list): List of observation minus analysis values.
        omf (list): List of observation minus first guess values.
        hofx (list): List of simulated observations.
        obs (list): List of observations.
        obserror (list): List of observation errors.
        preqc (list): List of pre-quality control values.
        postqc (list): List of post-quality control values.
        lev (list): List of levels/depths.
        instid (list): List of instrument IDs.
        col (list): List of colors.
        time (list): List of times.
        varname (str): Name of the variable.

    Methods:
        __init__(self, iodafnames, varname='waterTemperature'): Initializes the ioda object.
    """

    def __init__(self, iodafnames, varname='waterTemperature'):
        """
        Initializes the ioda object.

        Parameters:
            iodafnames (list): List of iodafname files.
            varname (str, optional): Name of the variable. Defaults to 'waterTemperature'.

        Returns:
            None
        """
        flist = iodafnames
        self.lon = []
        self.lat = []
        self.oma = []
        self.omf = []
        self.hofx = []
        self.obs = []
        self.obserror = []
        self.preqc = []
        self.postqc = []
        self.lev = []
        self.seqnum = []
        self.instid = []
        self.time = []
        self.varname = varname

        # Plot profile
        self.unit = ''
        if varname == 'waterTemperature':
            self.unit = '[^oC]'
        if varname == 'salinity':
            self.unit = '[psu]'

        def get_from_ioda(ncfile, varname, groupname):
            try:
                return ncfile.groups[groupname].variables[varname][:]
            except (KeyError, IndexError):
                return np.array([])

        pattern = re.compile(r'\.\d{10}\.nc4$')
        for iodafname in tqdm(flist):
            ncfile = Dataset(iodafname)
            bufr_subset = pattern.sub('', iodafname.split('.')[0])
            print(f"++++++++++++++++++++++++++++++++    {bufr_subset}")
            dum = get_from_ioda(ncfile, varname, 'ObsValue')
            valid_indices = np.where(abs(dum) < 9999999.9)
            self.obs = np.append(dum[valid_indices], self.obs)

            dum = get_from_ioda(ncfile, varname, 'ombg')
            self.omf = np.append(-dum[valid_indices], self.omf)

            dum = get_from_ioda(ncfile, varname, 'oman')
            self.oma = np.append(-dum[valid_indices], self.oma)

            dum = get_from_ioda(ncfile, varname, 'EffectiveError0')
            self.obserror = np.append(dum[valid_indices], self.obserror)

            dum = get_from_ioda(ncfile, varname, 'EffectiveQC0')
            self.postqc = np.append(dum[valid_indices], self.postqc)

            dum = get_from_ioda(ncfile, 'longitude', 'MetaData')
            self.lon = np.append(dum[valid_indices], self.lon)

            dum = get_from_ioda(ncfile, 'latitude', 'MetaData')
            self.lat = np.append(dum[valid_indices], self.lat)

            try:
                dum = get_from_ioda(ncfile, 'depth', 'MetaData')
                self.lev = np.append(-dum[valid_indices], self.lev)
            except KeyError:
                self.lev = np.append(0 * valid_indices[0], self.lev)

            instnum = dict_inst[bufr_subset].instid
            instid = instnum * np.ones(np.shape(dum))
            self.instid = np.append(instid[valid_indices], self.instid)
            ncfile.close()

        self.time = np.zeros(np.shape(self.lon))


class observation_space(object):
    """
    Represents an observation space for oceanview.

    Attributes:
    - iodafname (str): The name of the iodafname file.
    - varname (str): The name of the variable (default is 'waterTemperature').
    - ioda (ioda): An instance of the ioda class.
    - iodafname (str): The name of the iodafname file.
    - fignum (int): The figure number.
    - figure (Figure): The matplotlib figure.
    - axis (Axes): The matplotlib axis.
    - tooltip (ToolTip): The wxPython tooltip.
    - dataX (ndarray): The longitude data.
    - dataY (ndarray): The latitude data.
    - X (ndarray): The transformed longitude data.
    - Y (ndarray): The transformed latitude data.

    Methods:
    - find_inst(INSTID, dict_inst): Finds the instrument name based on the INSTID.
    - plot_prof(dax, fcst, ana, obs, z, var_name, sigo=None): Plots the profile.
    - draw_map(lonl=-180, lonr=180, proj='global'): Draws the map.
    - _onMotion(event): Handles the motion event.
    - _onClick(event): Handles the click event.
    """

    def __init__(self, iodafname, varname='waterTemperature'):
        """
        Initializes an instance of the observation_space class.

        Parameters:
        - iodafname (str): The name of the iodafname file.
        - varname (str): The name of the variable (default is 'waterTemperature').

        Returns:
        None
        """
        self.ioda = ioda(iodafname, varname=varname)
        self.iodafname = iodafname
        self.fignum = 2
        self.figure = pl.figure(num=1, figsize=(18, 10))
        self.axis = self.figure.add_subplot(111)
        self.tooltip = wx.ToolTip(tip='tip with a long %s line and a newline\n' % (' ' * 100))
        gcfm().canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)
        self.figure.canvas.mpl_connect('motion_notify_event', self._onMotion)
        self.figure.canvas.mpl_connect('button_press_event', self._onClick)
        self.dataX = np.squeeze(self.ioda.lon)
        self.dataY = np.squeeze(self.ioda.lat)

        map0 = self.draw_map(lonl=-180, lonr=180)
        x, y = map0(self.dataX, self.dataY)
        self.X = x
        self.Y = y
        cnt = 0
        alpha = 1.0
        for inst in range(505, 514):
            msize = 5.0
            # Plot obs loc
            valid_index = np.where(self.ioda.instid == inst)
            self.axis.plot(x[valid_index], y[valid_index], linestyle='None', marker='.',
                           markersize=msize,
                           label='myplot',
                           color=COLORS[cnt],
                           alpha=alpha)
            cnt += 1

    def find_inst(self, INSTID, dict_inst):
        """
        Finds the instrument name based on the INSTID.

        Parameters:
        - INSTID (int): The instrument ID.
        - dict_inst (dict): The dictionary of instrument information.

        Returns:
        - inst_name (str): The name of the instrument.
        """
        for instrument in dict_inst:
            if INSTID == dict_inst[instrument].instid:
                inst_name = dict_inst[instrument].name
        return inst_name

    def plot_prof(self, dax, fcst, ana, obs, z, var_name, sigo=None):
        """
        Plots the profile.

        Parameters:
        - dax (Axes): The matplotlib axis to plot on.
        - fcst (ndarray): The forecast data.
        - ana (ndarray): The analysis data.
        - obs (ndarray): The observation data.
        - z (ndarray): The depth data.
        - var_name (str): The name of the variable.
        - sigo (ndarray): The observation error data (default is None).

        Returns:
        None
        """
        # Sort the data based on depth
        sorted_index = sorted(range(len(z)), key=lambda k: z[k])

        # Plot the forecast data
        dax.plot(fcst[sorted_index], z[sorted_index], '-', lw=4, color='g', label='Background')

        # Plot the analysis data
        dax.plot(ana[sorted_index], z[sorted_index], '-', lw=4, color='r', label='Analysis')
        dax.plot(obs[sorted_index], z[sorted_index], '.', alpha=1.0, markersize=5.0, color='b', label='Observation')
        if sigo is not None:
            sigo_tmp = sigo[sorted_index]
            sigo_tmp[abs(sigo_tmp) > 999.9] = np.nan
            dax.plot(obs[sorted_index] - 1.0 * sigo_tmp, z[sorted_index], '--', linewidth=1.0, color='b')
            dax.plot(obs[sorted_index] + 1.0 * sigo_tmp, z[sorted_index], '--', linewidth=1.0, color='b')
        dax.legend()
        dax.set_xlabel(var_name, fontweight='bold')
        dax.grid(True)

    def draw_map(self, lonl=-180, lonr=180, proj='global'):
        """
        Draws the map.

        Parameters:
        - lonl (float): The left longitude boundary (default is -180).
        - lonr (float): The right longitude boundary (default is 180).
        - proj (str): The projection type (default is 'global').

        Returns:
        - map (Basemap): The Basemap object.
        """
        if proj == 'global':
            map = Basemap(projection='robin', lon_0=-180, resolution='c')
        if proj == 'polarn':
            map = Basemap(projection='npstere', boundinglat=60, lon_0=0, resolution='l')
        if proj == 'polars':
            map = Basemap(projection='spstere', boundinglat=-50, lon_0=0, resolution='l')
        map.drawcoastlines()
        map.fillcontinents(color='gray')

        return map

    def _onMotion(self, event):
        """
        Handles the motion event.

        Parameters:
        - event (MotionEvent): The matplotlib motion event.

        Returns:
        None
        """
        collisionFound = False
        if event.xdata is not None and event.ydata is not None:  # mouse is inside the axes
            for i in range(len(self.X)):
                radius = 100000  # Collision radius
                if (abs(event.xdata - self.X[i]) < radius) and (abs(event.ydata - self.Y[i]) < radius):
                    inst_name = self.find_inst(self.ioda.instid[i], dict_inst)
                    tip = (
                        'Lon=%f\n'
                        'Lat=%f\n'
                        'Instrument: %s\n'
                        'Var: %s' % (self.dataX[i], self.dataY[i], inst_name, self.ioda.varname)
                    )
                    self.tooltip.SetTip(tip)
                    self.tooltip.Enable(True)
                    self.i = i
                    collisionFound = True
                    break
        if not collisionFound:
            self.tooltip.Enable(False)

    def _onClick(self, event):
        """
        Handles the click event.

        Parameters:
        - event (MouseEvent): The matplotlib mouse event.

        Returns:
        None
        """
        # Left mouse click: Profile
        # --------------------------
        if event.button == 1:
            self.figure2 = plt.figure(num=self.fignum, figsize=(12, 12))
            self.axis2 = self.figure2.add_axes([0.3, 0.69, 0.4, 0.3])
            map = self.draw_map()
            for shift in [0, 360]:
                x, y = map(self.dataX + shift, self.dataY)
                self.axis2.plot(x[:], y[:],
                                linestyle='None', marker='.', markersize=.1, alpha=0.1, label='myplot', color='b')
                self.axis2.plot(x[self.i], y[self.i],
                                linestyle='None', marker='.', markersize=10, label='myplot', color='k')

            # Identify instrument and variable
            inst_name = self.find_inst(self.ioda.instid[self.i], dict_inst)

            # Prepare axis
            self.axis3 = self.figure2.add_axes([0.1, 0.05, 0.8, 0.6])
            self.axis3.set_ylabel('Depth [m]', fontweight='bold')

            # Get indices of observation pointed by mouth
            valid_index = np.where((self.ioda.lon == self.dataX[self.i]) & (self.ioda.lat == self.dataY[self.i]))
            z = self.ioda.lev[valid_index]
            fcst = self.ioda.obs[valid_index] - self.ioda.omf[valid_index]
            ana = self.ioda.obs[valid_index] - self.ioda.oma[valid_index]
            obsi = self.ioda.obs[valid_index]
            obserrori = self.ioda.obserror[valid_index]

            # Plot profile
            if self.ioda.varname == 'waterTemperature':
                profile_legend = f'Insitu temperature {self.ioda.unit}'
            if self.ioda.varname == 'salinity':
                profile_legend = f'Salinity {self.ioda.unit}'
            self.plot_prof(self.axis3, fcst, ana, obsi, z, profile_legend, sigo=obserrori)

            # Add obs info to the figure
            self.axis5 = self.figure2.add_axes([0.75, 0.75, 0.2, 0.2], frameon=False)
            self.axis5.axis('off')
            strtxt = '{0:10} {1}'.format('Instrument: ', inst_name) + '\n' + \
                     '{0:5} {1:3.2f}'.format('Lon:', self.dataX[self.i]) + '\n' + \
                     '{0:5} {1:3.2f}'.format('Lat:', self.dataY[self.i]) + '\n'
            self.axis5.text(0.01, 0.3, strtxt, fontsize=20, fontweight='bold')

            self.fignum += 1

            plt.show()

        # Middle mouse click: Regression plot for all instruments
        # --------------------------------------------------------
        if event.button == 2:
            # Isolate variable type
            for INSTID in tqdm(np.unique(self.ioda.instid)):
                # Identify instrument
                for instrument in dict_inst:
                    if INSTID == dict_inst[instrument].instid:
                        inst_name = dict_inst[instrument].name

                figure2 = plt.figure(num=self.fignum, figsize=(16, 12))

                def create_hist2d_plot(ax, data, depth, title=None, xlabel=None):
                    """
                    Creates a 2D histogram plot of data vs depth

                    Parameters:
                    - ax (matplotlib.axes): The axis to plot on
                    - data (array): The data for x-axis (omf or oma)
                    - depth (array): The depth data for y-axis
                    - title (str): Optional title for the plot
                    - xlabel (str): Label for x-axis
                    """
                    # Calculate bias and RMSE
                    bias = np.mean(data)
                    rmse = np.sqrt(np.mean(data**2))

                    # Create the 2D histogram
                    ax.hist2d(data, depth, bins=200, norm=LogNorm(), cmap='jet')
                    ax.set_xlim(-6, 6)
                    ax.grid(True)

                    if xlabel:
                        ax.set_xlabel(f'{xlabel} {self.ioda.unit}', fontweight='bold', fontsize=18)

                    if title:
                        ax.set_title(f'{title}: Bias = {bias:.3f}, RMSE = {rmse:.3f}',
                                     fontweight='bold', fontsize=16)

                    return bias, rmse

                # Set up the figure
                axis2 = figure2.add_subplot(121)
                plt.suptitle(inst_name, fontweight='bold', fontsize=18)

                # Plot OMF data
                valid_index = np.where((self.ioda.instid == INSTID)
                                       & (np.abs(self.ioda.omf) < 6.0)
                                       & (self.ioda.lev > -3000.0) & (self.ioda.lev < 0.0))
                yy = self.ioda.lev[valid_index]
                xx = self.ioda.omf[valid_index]
                create_hist2d_plot(axis2, xx, yy, self.ioda.varname, 'omf')
                axis2.set_ylabel('depth [m]', fontweight='bold', fontsize=18)

                # Plot OMA data
                axis3 = figure2.add_subplot(122)
                valid_index = np.where((self.ioda.instid == INSTID)
                                       & (np.abs(self.ioda.oma) < 6.0)
                                       & (self.ioda.lev > -3000.0) & (self.ioda.lev < 0.0))
                yy = self.ioda.lev[valid_index]
                xx = self.ioda.oma[valid_index]
                create_hist2d_plot(axis3, xx, yy, self.ioda.varname, 'oma')

                self.fignum += 1

            plt.show()

        # Right mouse click: Horizontal scatter plot of omf's and oma's for surface
        #                    Vertical scatter for profiles
        # --------------------------------------------------------------------------
        if event.button == 3:
            # Isolate var type
            for INSTID in tqdm(np.unique(self.ioda.instid)):
                # Identify instrument
                for instrument in dict_inst:
                    if INSTID == dict_inst[instrument].instid:
                        inst_name = dict_inst[instrument].name
                        allproj = dict_inst[instrument].proj

                for proj in allproj:
                    figure2 = plt.figure(num=self.fignum, figsize=(16, 12))

                    valid_index = np.where(np.logical_and((self.ioda.instid == INSTID), (self.ioda.lev < 10)))
                    STD = np.std(self.ioda.omf[valid_index])

                    axis2 = figure2.add_subplot(211)
                    map = self.draw_map(proj=proj)
                    for shift in [0, 360]:
                        x, y = map(self.dataX[valid_index] + shift, self.dataY[valid_index])
                        axis2.scatter(x, y, 5, c=self.ioda.omf[valid_index], cmap=cm.bwr,
                                      vmin=-2 * STD, vmax=2 * STD, edgecolor=None, lw=0)
                    titlestr = inst_name + ' OMF'
                    plt.title(titlestr, fontsize=24, fontweight='bold')

                    axis3 = figure2.add_subplot(212)
                    map = self.draw_map(proj=proj)
                    for shift in [0, 360]:
                        x, y = map(self.dataX[valid_index] + shift, self.dataY[valid_index])
                        axis3.scatter(x, y, 5, c=self.ioda.oma[valid_index],
                                      cmap=cm.bwr, vmin=-2 * STD, vmax=2 * STD, edgecolor=None, lw=0)
                    titlestr = inst_name + ' OMA'
                    plt.title(titlestr, fontsize=24, fontweight='bold')
                    self.fignum += 1

                    ax4 = figure2.add_axes([0.15, 0.25, 0.025, 0.5])
                    norm = mpl.colors.Normalize(vmin=-.5 * STD, vmax=.5 * STD)
                    mpl.colorbar.ColorbarBase(ax4, cmap=cm.bwr, norm=norm, orientation='vertical', extend='both')
            plt.show()


if __name__ == '__main__':
    description = """Observation space interactive map:
                     oceanview.py -i prof.out_*.nc adt.out_*.nc sst.out_*.nc"""

    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', help="ioda files from the output of soca DA", type=str, nargs='+', required=True)
    parser.add_argument('-v', '--variable', help="waterTemperature or salinity", type=str, required=True)

    print("""
             ============================================
             === Mouse left click: Profiles
             === Mouse middle click: regression
             === Mouse right click: Horizontal omf's/oma's
             === Usage: oceanview.py -i prof.out_*.nc adt.out_*.nc sst.out_*.nc
             ============================================
          """)
    args = parser.parse_args()
    listoffiles = args.input
    example = observation_space(listoffiles, args.variable)
    plt.show()
