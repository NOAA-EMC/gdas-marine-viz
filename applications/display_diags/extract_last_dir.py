#!/usr/bin/env python3

import argparse
import subprocess
import yaml
from pathlib import Path
from datetime import datetime, timedelta


def extract_last_dir(HPSS_root, hsi_output=None, result_file=None):
    """
    Extract and process cycle directories from HPSS archive.

    Args:
        HPSS_root: Root path in HPSS to search for cycle directories
        hsi_output: Path to temporary file for hsi output (default: "hsi_ls_out.txt")
        result_file: Path to YAML file tracking the last extracted cycle (default: "cycles.yaml")
    """
    # Set defaults for optional parameters
    if hsi_output is None:
        hsi_output = Path("hsi_ls_out.txt")
    else:
        hsi_output = Path(hsi_output)

    if result_file is None:
        result_file = Path("cycles.yaml")
    else:
        result_file = Path(result_file)

    # Convert HPSS_root to Path if string
    HPSS_dir = Path(HPSS_root)

    # Verify result_file exists before reading
    if not result_file.exists():
        raise FileNotFoundError(f"{result_file} not found - this file should contain the last extracted cycle")

    with result_file.open("r", encoding="utf-8") as f:
        cycles_data = yaml.safe_load(f)

    last_extracted_cycle_str = str(cycles_data["last_extracted_cycle"])

    # Validate
    if len(last_extracted_cycle_str) != 10 or not last_extracted_cycle_str.isdigit():
        raise ValueError(
            f"Invalid cycle string '{last_extracted_cycle_str}'; expected YYYYMMDDHH"
        )

    # Parse into datetime (UTC assumed) and start on the next cycle
    first_cycle = datetime.strptime(last_extracted_cycle_str, "%Y%m%d%H") + timedelta(hours=6)

    print(f"Parsed cycle datetime: {first_cycle} (UTC assumed)")

    # Command to list directories in HPSS (output written to hsi_output file)
    cmd = [
        "hsi",
        "-O", str(hsi_output),
        "ls", "-1",
        str(HPSS_dir)
    ]

    # Run the command
    subprocess.run(cmd, check=True)

    # Verify hsi created the output file
    if not hsi_output.exists():
        raise FileNotFoundError(f"{hsi_output} was not created by hsi")

    # Read the paths from the file written by hsi
    lines = [
        line.strip()
        for line in hsi_output.read_text().splitlines()
        if line.strip()
    ]

    if not lines:
        raise RuntimeError(f"No paths found in {hsi_output}")

    # Take the last path
    last_path = lines[-1]

    # Extract deepest directory name
    last_cycle_str = Path(last_path.rstrip("/")).name

    # Validate
    if len(last_cycle_str) != 10 or not last_cycle_str.isdigit():
        raise ValueError(
            f"Invalid cycle string '{last_cycle_str}'; expected YYYYMMDDHH"
        )

    # Parse into datetime (UTC assumed)
    last_cycle = datetime.strptime(last_cycle_str, "%Y%m%d%H")

    # Loop over cycles
    cycle = first_cycle
    while cycle <= last_cycle:
        cycle_str = cycle.strftime("%Y%m%d%H")

        print(f"Processing cycle {cycle_str} **********")

        commands = [
            ["htar", "-xf", f"{HPSS_dir}/{cycle_str}/gdasocean_analysis.tar"],
            ["htar", "-xf", f"{HPSS_dir}/{cycle_str}/gdasocean.tar"],
            ["htar", "-xf", f"{HPSS_dir}/{cycle_str}/gdasice.tar"],
        ]

        cycle_succeeded = True
        for cmd in commands:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error while executing command: {cmd}. Error: {e}")
                print("Breaking out of the loop without incrementing the cycle.")
                cycle_succeeded = False
                break

        if not cycle_succeeded:
            return  # Exit the function early if the error occurs, leaving result_file at the last success

        # Record progress immediately so a later failure doesn't lose completed work
        result_file.write_text(
            yaml.dump({"last_extracted_cycle": cycle_str}, default_flow_style=False)
        )

        cycle += timedelta(hours=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and process cycle directories from HPSS archive"
    )
    parser.add_argument(
        "HPSS_root",
        type=str,
        help="Root path in HPSS to search for cycle directories"
    )
    parser.add_argument(
        "--hsi-output",
        type=str,
        default=None,
        help="Path to temporary file for hsi output (default: hsi_ls_out.txt)"
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        help="Path to YAML file tracking the last extracted cycle (default: cycles.yaml)"
    )

    args = parser.parse_args()

    extract_last_dir(
        HPSS_root=args.HPSS_root,
        hsi_output=args.hsi_output,
        result_file=args.result_file
    )
