#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

logger = logging.getLogger("get_hpss_arch")

# Tar files pulled from each cycle directory in HPSS, in extraction order
CYCLE_TARFILES = [
    "gdasocean_analysis.tar",
    "gdasocean.tar",
    "gdasice.tar",
    "gdas.tar",
]

CYCLE_FMT = "%Y%m%d%H"
CYCLE_STEP = timedelta(hours=6)


def setup_logging(level=logging.INFO):
    """Send timestamped log messages to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False


def _parse_cycle(cycle_str):
    """Validate a YYYYMMDDHH string and return it as a datetime (UTC assumed)."""
    if len(cycle_str) != 10 or not cycle_str.isdigit():
        raise ValueError(f"Invalid cycle string '{cycle_str}'; expected YYYYMMDDHH")
    return datetime.strptime(cycle_str, CYCLE_FMT)


def extract_last_dir(HPSS_root, hsi_output=None, result_file=None):
    """
    Extract and process cycle directories from HPSS archive.

    Args:
        HPSS_root: Root path in HPSS to search for cycle directories
        hsi_output: Path to temporary file for hsi output (default: "hsi_ls_out.txt")
        result_file: Path to YAML file tracking the last extracted cycle (default: "cycles.yaml")
    """
    hsi_output = Path(hsi_output) if hsi_output is not None else Path("hsi_ls_out.txt")
    result_file = Path(result_file) if result_file is not None else Path("cycles.yaml")
    HPSS_dir = Path(HPSS_root)

    # Verify result_file exists before reading
    if not result_file.exists():
        raise FileNotFoundError(f"{result_file} not found - this file should contain the last extracted cycle")

    with result_file.open("r", encoding="utf-8") as f:
        cycles_data = yaml.safe_load(f)

    last_extracted_cycle_str = str(cycles_data["last_extracted_cycle"])
    logger.info("Last extracted cycle (from %s): %s",
                result_file.resolve(), last_extracted_cycle_str)

    # Start on the cycle after the last one extracted
    first_cycle = _parse_cycle(last_extracted_cycle_str) + CYCLE_STEP
    logger.info("First cycle to extract: %s", first_cycle.strftime(CYCLE_FMT))

    # List cycle directories in HPSS (output written to hsi_output file)
    cmd = ["hsi", "-O", str(hsi_output), "ls", "-1", str(HPSS_dir)]
    logger.info("Listing HPSS directory %s", HPSS_dir)
    logger.debug("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Verify hsi created the output file
    if not hsi_output.exists():
        raise FileNotFoundError(f"{hsi_output} was not created by hsi")

    # Read the paths from the file written by hsi
    lines = [line.strip() for line in hsi_output.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No paths found in {hsi_output}")

    # The last listed path is the most recent cycle in HPSS
    last_cycle_str = Path(lines[-1].rstrip("/")).name
    last_cycle = _parse_cycle(last_cycle_str)
    logger.info("Found %d entries in HPSS; latest archived cycle: %s",
                len(lines), last_cycle_str)

    if first_cycle > last_cycle:
        logger.info("Nothing to do: already up to date through %s", last_extracted_cycle_str)
        return

    n_cycles = int((last_cycle - first_cycle) / CYCLE_STEP) + 1
    logger.info("Will extract %d cycle(s): %s through %s into %s",
                n_cycles, first_cycle.strftime(CYCLE_FMT), last_cycle_str, os.getcwd())

    # Loop over cycles
    cycle = first_cycle
    icycle = 0
    while cycle <= last_cycle:
        icycle += 1
        cycle_str = cycle.strftime(CYCLE_FMT)
        logger.info("========== Cycle %s (%d of %d) ==========", cycle_str, icycle, n_cycles)
        cycle_start = time.monotonic()

        for ifile, tarfile in enumerate(CYCLE_TARFILES, start=1):
            tarpath = f"{HPSS_dir}/{cycle_str}/{tarfile}"
            cmd = ["htar", "-xf", tarpath]
            logger.info("Cycle %s: downloading file %d of %d: %s",
                        cycle_str, ifile, len(CYCLE_TARFILES), tarpath)
            file_start = time.monotonic()
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error("Cycle %s: failed to extract %s (exit code %s): %s",
                             cycle_str, tarpath, e.returncode, " ".join(cmd))
                logger.error("Stopping; %s left at last successful cycle", result_file)
                return
            logger.info("Cycle %s: finished %s in %.1f s",
                        cycle_str, tarfile, time.monotonic() - file_start)

        # Record progress immediately so a later failure doesn't lose completed work
        result_file.write_text(
            yaml.dump({"last_extracted_cycle": cycle_str}, default_flow_style=False)
        )
        logger.info("Cycle %s complete in %.1f s; updated %s",
                    cycle_str, time.monotonic() - cycle_start, result_file)

        cycle += CYCLE_STEP

    logger.info("All %d cycle(s) extracted through %s", n_cycles, last_cycle_str)


def main(yaml_file):
    with open(yaml_file) as f:
        expconfigs = yaml.safe_load(f)

    local_hpss_root = expconfigs['local hpss root']

    for experiment in expconfigs['experiments']:
        pslot = experiment['pslot']
        local_hpss_dir = os.path.join(local_hpss_root, pslot)
        HPSS_root = experiment['HPSS root']
        hpss_dir = os.path.join(HPSS_root, pslot)

        logger.info("##### Experiment %s #####", pslot)
        logger.info("HPSS source: %s", hpss_dir)
        logger.info("Local destination: %s", local_hpss_dir)

        if os.path.isdir(local_hpss_dir):
            os.chdir(local_hpss_dir)
        else:
            raise FileNotFoundError(f"Directory does not exist: {local_hpss_dir}")

        extract_last_dir(hpss_dir, hsi_output='out.txt', result_file='cycles.yaml')
        logger.info("Finished experiment %s", pslot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HPSS archive.")
    parser.add_argument(
        "--yaml_file",
        type=str,
        default="get_hpss_arch.yaml",
        help="Path to the YAML configuration file (default: get_hpss_arch.yaml)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also log the exact hsi/htar commands being run"
    )
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    main(args.yaml_file)
