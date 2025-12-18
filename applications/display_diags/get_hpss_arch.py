#!/usr/bin/env python3

from applications.display_diags.extract_last_dir import extract_last_dir
from wxflow import parse_yaml
import os

expconfigs = parse_yaml(path="get_hpss_arch.yaml")

local_hpss_root = expconfigs['local hpss root']

for experiment in expconfigs['experiments']:
   
    pslot = experiment['pslot']
    print("Extracting experiment ", pslot)
    local_hpss_dir = os.path.join(local_hpss_root,pslot)
    if os.path.isdir(local_hpss_dir):
       os.chdir(local_hpss_dir)
    else:
       raise FileNotFoundError(f"Directory does not exist: {local_hpss_path}")

    HPSS_root = experiment['HPSS root']
    hpss_dir = os.path.join(HPSS_root,pslot)
    extract_last_dir(hpss_dir, hsi_output='out.txt', result_file='cycles.txt')



