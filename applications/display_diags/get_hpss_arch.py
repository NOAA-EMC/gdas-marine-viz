#!/usr/bin/env python3

import argparse
import os
from applications.display_diags.extract_last_dir import extract_last_dir
from wxflow import parse_yaml

def main(yaml_file):
    expconfigs = parse_yaml(path=yaml_file)

    local_hpss_root = expconfigs['local hpss root']

    for experiment in expconfigs['experiments']:
        pslot = experiment['pslot']
        print("Extracting experiment ", pslot)
        local_hpss_dir = os.path.join(local_hpss_root, pslot)
        if os.path.isdir(local_hpss_dir):
            os.chdir(local_hpss_dir)
        else:
            raise FileNotFoundError(f"Directory does not exist: {local_hpss_dir}")

        HPSS_root = experiment['HPSS root']
        hpss_dir = os.path.join(HPSS_root, pslot)
        extract_last_dir(hpss_dir, hsi_output='out.txt', result_file='cycles.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HPSS archive.")
    parser.add_argument(
        "--yaml_file",
        type=str,
        default="get_hpss_arch.yaml",
        help="Path to the YAML configuration file (default: get_hpss_arch.yaml)"
    )
    args = parser.parse_args()
    main(args.yaml_file)
