import os
import glob
import subprocess
import re
import netCDF4
from wxflow import parse_j2yaml

# obs space statistics
print("---------------- Compute basic stats")
comout = os.getenv('COM_OCEAN_ANALYSIS')
HOMEgdasmv = os.getenv('HOMEgdasmv')
HOMEgfs = os.getenv('HOMEgfs')
nens = os.getenv('NENS')
diags_list = glob.glob(os.path.join(os.path.join(comout, 'letkf', 'diags', '*.nc4')))
obsstats_j2yaml = os.path.join(HOMEgdasmv, 'configs', 'obs_stats.yaml.j2')


# function to create a minimalist ioda obs sapce
def create_obs_space(data):
    os_dict = {"obs space": {
               "name": data["obs_space"],
               "obsdatain": {
                   "engine": {"type": "H5File", "obsfile": data["obsfile"]}
               },
               "simulated variables": [data["variable"]]
               },
               "variable": data["variable"],
               "experiment identifier": data["pslot"],
               "csv output": data["csv_output"]
               }
    return os_dict


# get the experiment id
pslot = os.getenv("PSLOT")


# iterate through the obs spaces and generate the yaml for gdassoca_obsstats.x
obs_spaces = []
for obsfile in diags_list:
    # define an obs space name
    obs_space = re.sub(r'\.\d{10}\.nc4$', '', os.path.basename(obsfile))

    # get the variable name, assume 1 variable per file
    nc = netCDF4.Dataset(obsfile, 'r')
    variable = next(iter(nc.groups["ombg"].variables))
    nc.close()

    # filling values for the templated yaml
    data = {'obs_space': os.path.basename(obsfile),
            'obsfile': obsfile,
            'pslot': pslot,
            'variable': variable,
            'csv_output': os.path.join(comout, 'letkf', 'diags', f"{obs_space}.stats.csv")}
    obs_spaces.append(create_obs_space(data))

# create the yaml
data = {'obs_spaces': obs_spaces, 'nens': nens}
conf = parse_j2yaml(path=obsstats_j2yaml, data=data)
stats_yaml = 'diag_stats.yaml'
conf.save(stats_yaml)

# Path to your executable
exe_path = HOMEgfs + '/sorc/gdas.cd/build/bin/gdassoca_obsstats.x'

# Run the executable
result = subprocess.run([exe_path, stats_yaml], capture_output=True, text=True)
