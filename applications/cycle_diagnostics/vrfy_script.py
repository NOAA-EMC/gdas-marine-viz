import os
import numpy as np
import gen_eva_obs_yaml
import marine_eva_post
from multiprocessing import Process
from soca_vrfy import statePlotter, plotConfig
import subprocess
import glob

comout = os.getenv('COM_OCEAN_ANALYSIS')
com_ocean_analysis = os.getenv('COM_OCEAN_ANALYSIS')
com_ice_analysis = os.getenv('COM_ICE_ANALYSIS')
print("comout: ", comout)
comconf = os.getenv('COM_CONF')
# resolve the comout path since it may contain wild cards
matching_paths = glob.glob(comout)
if matching_paths:
    comout = matching_paths[0]  # Assuming you want the first match
    print(comout)
else:
    print(comout)
    print("No matching paths found")
    exit(1)
del matching_paths

com_ice_history = os.getenv('COM_ICE_HISTORY_PREV')
com_ocean_history = os.getenv('COM_OCEAN_HISTORY_PREV')
plot_background = os.getenv('PLOT_BACKGROUND', 'OFF').upper() == 'ON'
if plot_background:
    # resolve the comout path since it may contain wild cards
    matching_paths = glob.glob(com_ice_history)
    if matching_paths:
        com_ice_history = matching_paths[0]  # Assuming you want the first match
        print(com_ice_history)
    else:
        print(com_ice_history)
        print("No matching paths found")
        exit(1)
    del matching_paths
    # resolve the comout path since it may contain wild cards
    matching_paths = glob.glob(com_ocean_history)
    if matching_paths:
        com_ocean_history = matching_paths[0]  # Assuming you want the first match
        print(com_ocean_history)
    else:
        print(com_ocean_history)
        print("No matching paths found")
        exit(1)
    del matching_paths

cyc = os.getenv('cyc')
RUN = os.getenv('RUN')

bcyc = str((int(cyc) - 3) % 24).zfill(2)
gcyc = str((int(cyc) - 6) % 24).zfill(2)
grid_file = os.path.join(comout, f'{RUN}.t' + bcyc + 'z.ocngrid.nc')

layer_file=os.path.join(com_ocean_history, f'{RUN}.t' + gcyc + 'z.inst.f006.nc')

# bkg_err grid file path based on the system's hostname
hpcname = os.getenv('HPCname')
if hpcname.startswith("hera"):
    grid_file_bkgerr = '/scratch1/NCEPDEV/da/common/validation/vrfy/soca_gridspec.bkgerr.nc'
elif hpcname in ["ursa"]:
    grid_file_bkgerr = '/scratch3/NCEPDEV/da/common/validation/vrfy/soca_gridspec.bkgerr.nc'
elif hpcname in ["hercules", "orion"]:
    grid_file_bkgerr = '/work/noaa/da/marineda/validation/vrfy/soca_gridspec.bkgerr.nc'
else:
    print(f"Error: Unrecognized HPC name '{hpcname}'. Aborting.")
    exit(1)

# Check if the file exists, then decide on grid_file
if not os.path.exists(grid_file):
    # TODO: Make this work on other HPC
    grid_file = '/scratch3/NCEPDEV/da/common/validation/vrfy/gdas.t21z.ocngrid.nc'

# for eva
diagdir = os.path.join(comout, 'diags')
letkfdiagdir = os.path.join(comout, 'letkf', 'diags')
HOMEgdasmv = os.getenv('HOMEgdasmv')

# Get flags from environment variables (set in the bash driver)
plot_ensemble_b = os.getenv('PLOT_ENSEMBLE_B', 'OFF').upper() == 'ON'
plot_parametric_b = os.getenv('PLOT_PARAMETRIC_B', 'OFF').upper() == 'ON'
plot_letkf_ensemble = os.getenv('PLOT_LETKF_ENSEMBLE', 'OFF').upper() == 'ON'
plot_increment = os.getenv('PLOT_INCREMENT', 'OFF').upper() == 'ON'
plot_analysis = os.getenv('PLOT_ANALYSIS', 'OFF').upper() == 'ON'
eva_plots = os.getenv('EVA_PLOTS', 'OFF').upper() == 'ON'
eva_letkf_plots = os.getenv('EVA_LETKF_PLOTS', 'OFF').upper() == 'ON'

# output directory
vrfyout = os.getenv('VRFYOUT', './vrfyout')
print('------------------------- vrfyout:', vrfyout)

# Initialize an empty list for the main config
configs = []

# Analysis plotting configuration
if plot_analysis:
    print('Plotting analysis')
    configs_ana = [plotConfig(grid_file=grid_file,
                              data_file=os.path.join(com_ocean_analysis, f'{RUN}.t' + cyc + 'z.jedi_analysis.a006.nc'),
                              variables_horiz={
                                  'ave_ssh': [-1.8, 1.3],
                                  'Temp': [-1.8, 34.0],
                                  'Salt': [32, 40]},
                              colormap='nipy_spectral',
                              vrfyout=os.path.join(vrfyout, 'vrfy', 'ana')),   # ocean surface analysis
                   plotConfig(grid_file=grid_file,
                              data_file=os.path.join(com_ice_analysis, f'{RUN}.t' + cyc + 'z.jedi_analysis.a006.nc'),
                              variables_horiz={'aice_h': [0.0, 1.0],
                                               'hi_h': [0.0, 4.0],
                                               'hs_h': [0.0, 0.5]},
                              colormap='jet',
                              projs=['North', 'South', 'Global'],
                              vrfyout=os.path.join(vrfyout, 'vrfy', 'ana'))]   # sea ice analysis
    configs.extend(configs_ana)

# Ensemble B plotting configuration
if plot_ensemble_b:
    print('Plotting ensemble B SSH diagnostics')
    config_ens = [plotConfig(grid_file=grid_file_bkgerr,
                             data_file=os.path.join(comout, f'{RUN}.t{cyc}z.ocn.recentering_error.nc'),
                             variables_horiz={'ave_ssh': [-0.05, 0.05]},
                             colormap='seismic',
                             vrfyout=os.path.join(vrfyout, 'vrfy', 'recentering_error')),   # recentering error
                  plotConfig(grid_file=grid_file_bkgerr,
                             data_file=os.path.join(comout, f'{RUN}.t{cyc}z.ocn.ssh_steric_stddev.nc'),
                             variables_horiz={'ave_ssh': [0, 0.8]},
                             colormap='gist_ncar',
                             vrfyout=os.path.join(vrfyout, 'vrfy', 'bkgerr', 'ssh_steric_stddev')),  # ssh steric stddev
                  plotConfig(grid_file=grid_file_bkgerr,
                             data_file=os.path.join(comout, f'{RUN}.t{cyc}z.ocn.ssh_unbal_stddev.nc'),
                             variables_horiz={'ave_ssh': [0, 0.8]},
                             colormap='gist_ncar',
                             vrfyout=os.path.join(vrfyout, 'vrfy', 'bkgerr', 'ssh_unbal_stddev')),   # ssh unbal stddev
                  plotConfig(grid_file=grid_file_bkgerr,
                             data_file=os.path.join(comout, f'{RUN}.t{cyc}z.ocn.ssh_total_stddev.nc'),
                             variables_horiz={'ave_ssh': [0, 0.8]},
                             colormap='gist_ncar',
                             vrfyout=os.path.join(vrfyout, 'vrfy', 'bkgerr', 'ssh_total_stddev')),   # ssh total stddev
                  plotConfig(grid_file=grid_file_bkgerr,
                             data_file=os.path.join(comout, f'{RUN}.t{cyc}z.ocn.steric_explained_variance.nc'),
                             variables_horiz={'ave_ssh': [0, 1]},
                             colormap='seismic',
                             vrfyout=os.path.join(vrfyout, 'vrfy', 'bkgerr',
                                                  'steric_explained_variance'))]  # steric explained variance

    configs.extend(config_ens)

# Parametric B plotting configuration
if plot_parametric_b:
    print('Plotting parametric B diagnostics')
    config_bkgerr = [plotConfig(grid_file=grid_file_bkgerr,
                                data_file=os.path.join(comout, os.path.pardir, os.path.pardir,
                                                       'bmatrix', 'ice', f'{RUN}.t' + cyc + 'z.ice.bkgerr_stddev.nc'),
                                variables_horiz={'aice_h': [0.0, 0.3],
                                                 'hi_h': [0.0, 2.0],
                                                 'hs_h': [0.0, 0.2]},
                                colormap='gist_ncar',
                                projs=['North', 'South', 'Global'],
                                vrfyout=os.path.join(vrfyout, 'vrfy', 'bkgerr')),   # sea ice bkgerr stddev
                     plotConfig(grid_file=grid_file_bkgerr,
                                layer_file=grid_file_bkgerr,
                                data_file=os.path.join(comout, os.path.pardir, os.path.pardir,
                                                       'bmatrix', 'ocean', f'{RUN}.t' + cyc + 'z.ocean.bkgerr_stddev.nc'),
                                lats=np.arange(-60, 60, 10),
                                lons=np.arange(-280, 80, 30),
                                variables_zonal={'Temp': [0, 2],
                                                 'Salt': [0, 0.2],
                                                 'u': [0, 0.5],
                                                 'v': [0, 0.5]},
                                variables_meridional={'Temp': [0, 2],
                                                      'Salt': [0, 0.2],
                                                      'u': [0, 0.5],
                                                      'v': [0, 0.5]},
                                variables_horiz={'Temp': [0, 1],
                                                 'Salt': [0, 0.2],
                                                 'u': [0, 0.5],
                                                 'v': [0, 0.5],
                                                 'ave_ssh': [0, 0.1]},
                                colormap='gist_ncar',
                                vrfyout=os.path.join(vrfyout, 'vrfy', 'bkgerr'))]   # ocn bkgerr stddev
    configs.extend(config_bkgerr)

# LETKF ensemble plotting configuration
if plot_letkf_ensemble:
    print('Plotting background ensemble diagnostics')
    config_letkf = [plotConfig(grid_file=grid_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ice.t{cyc}z.ensvar_prior.nc'),
                               variables_horiz={'aice_h': [0.0, 0.3]},
                               colormap='gist_ncar',
                               projs=['North', 'South', 'Global'],
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_bkg_std'),
                               plot_sqrt=True),   # sea ice background ensemble spread
                    plotConfig(grid_file=grid_file,
                               layer_file=layer_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ocean.t{cyc}z.ensvar_prior.nc'),
                               lats=np.arange(-60, 60, 10),
                               lons=np.arange(-280, 80, 30),
                               variables_zonal={'Temp': [0, 2],
                                                'Salt': [0, 0.2],
                                                'u': [0, 0.5],
                                                'v': [0, 0.5]},
                               variables_meridional={'Temp': [0, 2],
                                                     'Salt': [0, 0.2],
                                                     'u': [0, 0.5],
                                                     'v': [0, 0.5]},
                               variables_horiz={'Temp': [0, 1],
                                                'Salt': [0, 0.2],
                                                'u': [0, 0.5],
                                                'v': [0, 0.5],
                                                'ave_ssh': [0, 0.05]},
                               colormap='gist_ncar',
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_bkg_std'),
                               plot_sqrt=True),   # ocn background ensemble spread
                    plotConfig(grid_file=grid_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ice.t{cyc}z.ensvar_post.nc'),
                               variables_horiz={'aice_h': [0.0, 0.3]},
                               colormap='gist_ncar',
                               projs=['North', 'South', 'Global'],
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_ana_std'),
                               plot_sqrt=True),   # sea ice analysis spread
                    plotConfig(grid_file=grid_file,
                               layer_file=layer_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ocean.t{cyc}z.ensvar_post.nc'),
                               lats=np.arange(-60, 60, 10),
                               lons=np.arange(-280, 80, 30),
                               variables_zonal={'Temp': [0, 2],
                                                'Salt': [0, 0.2],
                                                'u': [0, 0.5],
                                                'v': [0, 0.5]},
                               variables_meridional={'Temp': [0, 2],
                                                     'Salt': [0, 0.2],
                                                     'u': [0, 0.5],
                                                     'v': [0, 0.5]},
                               variables_horiz={'Temp': [0, 1],
                                                'Salt': [0, 0.2],
                                                'u': [0, 0.5],
                                                'v': [0, 0.5],
                                                'ave_ssh': [0, 0.05]},
                               colormap='gist_ncar',
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_ana_std'),
                               plot_sqrt=True),   # ocn letkf var
                    plotConfig(grid_file=grid_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ice.t{cyc}z.ensmean_prior.nc'),
                               variables_horiz={'aice_h': [0.0, 1.0]},
                               colormap='jet',
                               projs=['North', 'South', 'Global'],
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_bkg_mean')),   # sea ice mean ensemble background
                    plotConfig(grid_file=grid_file,
                               layer_file=layer_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ocean.t{cyc}z.ensmean_prior.nc'),
                               lats=np.arange(-60, 60, 10),
                               lons=np.arange(-280, 80, 30),
                               variables_zonal={'Temp': [-1.8, 34.0],
                                                'Salt': [32, 40],
                                                'u': [-1.0, 1.0],
                                                'v': [-1.0, 1.0]},
                               variables_meridional={'Temp': [-1.8, 34.0],
                                                     'Salt': [32, 40],
                                                     'u': [-1.0, 1.0],
                                                     'v': [-1.0, 1.0]},
                               variables_horiz={'ave_ssh': [-1.8, 1.3],
                                                'Temp': [-1.8, 34.0],
                                                'Salt': [32, 40],
                                                'u': [-1.0, 1.0],
                                                'v': [-1.0, 1.0]},
                               colormap='nipy_spectral',
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_bkg_mean')),  # ocean mean ensemble background
                    plotConfig(grid_file=grid_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ice.t{cyc}z.ensmean_post.nc'),
                               variables_horiz={'aice_h': [0.0, 1.0]},
                               colormap='jet',
                               projs=['North', 'South', 'Global'],
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_ana_mean')),   # sea ice mean ensemble analysis
                    plotConfig(grid_file=grid_file,
                               layer_file=layer_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ocean.t{cyc}z.ensmean_post.nc'),
                               lats=np.arange(-60, 60, 10),
                               lons=np.arange(-280, 80, 30),
                               variables_zonal={'Temp': [-1.8, 34.0],
                                                'Salt': [32, 40],
                                                'u': [-1.0, 1.0],
                                                'v': [-1.0, 1.0]},
                               variables_meridional={'Temp': [-1.8, 34.0],
                                                     'Salt': [32, 40],
                                                     'u': [-1.0, 1.0],
                                                     'v': [-1.0, 1.0]},
                               variables_horiz={'ave_ssh': [-1.8, 1.3],
                                                'Temp': [-1.8, 34.0],
                                                'Salt': [32, 40],
                                                'u': [-1.0, 1.0],
                                                'v': [-1.0, 1.0]},
                               colormap='nipy_spectral',
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_ana_mean')),  # ocean mean ensemble analysis
                    plotConfig(grid_file=grid_file,
                               layer_file=layer_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ocean.t{cyc}z.ensmean_incr.nc'),
                               lats=np.arange(-60, 60, 10),
                               lons=np.arange(-280, 80, 30),
                               variables_zonal={'Temp': [-0.5, 0.5],
                                                'Salt': [-0.1, 0.1]},
                               variables_horiz={'Temp': [-0.5, 0.5],
                                                'Salt': [-0.1, 0.1],
                                                'ave_ssh': [-0.1, 0.1]},
                               variables_meridional={'Temp': [-0.5, 0.5],
                                                     'Salt': [-0.1, 0.1]},
                               colormap='seismic',
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_incr_mean')),   # ocean mean LETKF increment
                    plotConfig(grid_file=grid_file,
                               data_file=os.path.join(comout, 'letkf', f'enkfgdas.ice.t{cyc}z.ensmean_incr.nc'),
                               lats=np.arange(-60, 60, 10),
                               variables_horiz={'aice_h': [-0.2, 0.2]},
                               colormap='seismic',
                               projs=['North', 'South'],
                               vrfyout=os.path.join(vrfyout, 'vrfy', 'letkf_incr_mean'))]   # sea ice mean LETKF increment
    configs.extend(config_letkf)

# Background plotting configuration
if plot_background:
    print('Plotting background')
    config_bkg = [plotConfig(grid_file=grid_file,
                             data_file=os.path.join(com_ice_history, f'{RUN}.t{gcyc}z.inst.f006.nc'),
                             variables_horiz={'aice_h': [0.0, 1.0],
                                              'hi_h': [0.0, 4.0],
                                              'hs_h': [0.0, 0.5]},
                             colormap='jet',
                             projs=['North', 'South', 'Global'],
                             vrfyout=os.path.join(vrfyout, 'vrfy', 'bkg')),   # sea ice background
                  plotConfig(grid_file=grid_file,
                             layer_file=layer_file,
                             data_file=os.path.join(com_ocean_history, f'{RUN}.t{gcyc}z.inst.f006.nc'),
                             lats=np.arange(-60, 60, 10),
                             lons=np.arange(-280, 80, 30),
                             variables_zonal={'Temp': [-1.8, 34.0],
                                              'Salt': [32, 40],
                                              'u': [-1.0, 1.0],
                                              'v': [-1.0, 1.0]},
                             variables_meridional={'Temp': [-1.8, 34.0],
                                                   'Salt': [32, 40],
                                                   'u': [-1.0, 1.0],
                                                   'v': [-1.0, 1.0]},
                             variables_horiz={'ave_ssh': [-1.8, 1.3],
                                              'Temp': [-1.8, 34.0],
                                              'Salt': [32, 40],
                                              'u': [-1.0, 1.0],
                                              'v': [-1.0, 1.0]},
                             colormap='nipy_spectral',
                             vrfyout=os.path.join(vrfyout, 'vrfy', 'bkg'))]
    configs.extend(config_bkg)

# Increment plotting configuration
if plot_increment:
    print('Plotting increment')
    config_incr = [plotConfig(grid_file=grid_file,
                              layer_file=layer_file,
                              data_file=os.path.join(com_ocean_analysis, f'{RUN}.t' + cyc + 'z.jedi_increment.i006.nc'),
                              lats=np.arange(-60, 60, 10),
                              lons=np.arange(-280, 80, 30),
                              variables_zonal={'Temp': [-0.5, 0.5],
                                               'Salt': [-0.1, 0.1]},
                              variables_horiz={'Temp': [-0.5, 0.5],
                                               'Salt': [-0.1, 0.1],
                                               'ave_ssh': [-0.1, 0.1]},
                              variables_meridional={'Temp': [-0.5, 0.5],
                                                    'Salt': [-0.1, 0.1]},
                              colormap='seismic',
                              vrfyout=os.path.join(vrfyout, 'vrfy', 'incr')),   # ocean increment
                   plotConfig(grid_file=grid_file,
                              data_file=os.path.join(com_ice_analysis, f'{RUN}.t' + cyc + 'z.jedi_increment.i006.nc'),
                              lats=np.arange(-60, 60, 10),
                              variables_horiz={'aice_h': [-0.2, 0.2],
                                               'hi_h': [-0.5, 0.5],
                                               'hs_h': [-0.1, 0.1]},
                              colormap='seismic',
                              projs=['North', 'South'],
                              vrfyout=os.path.join(vrfyout, 'vrfy', 'incr'))]   # sea ice increment
#                   plotConfig(grid_file=grid_file,
#                              data_file=os.path.join(comout, f'{RUN}.t' + cyc + 'z.ice.incr.postproc.nc'),
#                              lats=np.arange(-60, 60, 10),
#                              variables_horiz={'aice_h': [-0.2, 0.2],
#                                               'hi_h': [-0.5, 0.5],
#                                               'hs_h': [-0.1, 0.1]},
#                              colormap='seismic',
#                              projs=['North', 'South'],
#                              vrfyout=os.path.join(vrfyout,
#                                                   'vrfy', 'incr.postproc'))]   # sea ice increment after postprocessing
    configs.extend(config_incr)


# Plot the marine verification figures
def plot_marine_vrfy(config):
    ocnvrfyPlotter = statePlotter(config)
    ocnvrfyPlotter.plot()


# Number of processes
num_processes = len(configs)

# Create a list to store the processes
processes = []

# Iterate over configs
for config in configs[:num_processes]:
    process = Process(target=plot_marine_vrfy, args=(config,))
    process.start()
    processes.append(process)

# Wait for all processes to finish
for process in processes:
    process.join()

# Run EVA
if eva_plots:
    evadir = os.path.join(HOMEgdasmv)
    marinetemplate = os.path.join(evadir, 'configs', 'marine_gdas_plots.yaml')
    varyaml = os.path.join(comconf, 'var.yaml')

    # it would be better to refrence the dirs explicitly with the comout path
    # but eva doesn't allow for specifying output directories
    vrfydir = os.path.join(vrfyout, 'vrfy')
    if not os.path.exists(vrfydir):
        os.makedirs(vrfydir)
    os.chdir(vrfydir)
    if not os.path.exists('preevayamls'):
        os.makedirs('preevayamls')
    if not os.path.exists('evayamls'):
        os.makedirs('evayamls')

    gen_eva_obs_yaml.gen_eva_obs_yaml(varyaml, marinetemplate, 'preevayamls')

    files = os.listdir('preevayamls')
    for file in files:
        infile = os.path.join('preevayamls', file)
        marine_eva_post.marine_eva_post(infile, 'evayamls', diagdir)

    files = os.listdir('evayamls')
    for file in files:
        infile = os.path.join('evayamls', file)
        print('running eva on', infile)
        subprocess.run(['eva', infile], check=True)

# Run EVA on LETKF diags
if eva_letkf_plots:
    evadir = os.path.join(HOMEgdasmv)
    marinetemplate = os.path.join(evadir, 'configs', 'marine_letkf_gdas_plots.yaml')
    letkfyaml = os.path.join(comout, 'letkf', 'letkf.yaml')

    # it would be better to refrence the dirs explicitly with the comout path
    # but eva doesn't allow for specifying output directories
    os.chdir(os.path.join(vrfyout, 'vrfy'))
    if not os.path.exists('preevayamls_letkf'):
        os.makedirs('preevayamls_letkf')
    if not os.path.exists('evayamls_letkf'):
        os.makedirs('evayamls_letkf')

    gen_eva_obs_yaml.gen_eva_obs_yaml(letkfyaml, marinetemplate, 'preevayamls_letkf')

    files = os.listdir('preevayamls_letkf')
    for file in files:
        infile = os.path.join('preevayamls_letkf', file)
        marine_eva_post.marine_eva_post(infile, 'evayamls_letkf', letkfdiagdir)

    files = os.listdir('evayamls_letkf')
    for file in files:
        infile = os.path.join('evayamls_letkf', file)
        print('running eva on', infile)
        subprocess.run(['eva', infile], check=True)

#######################################
# calculate diag statistics
#######################################

# As of 11/12/2024 not working
# diag_statistics.get_diag_stats()
