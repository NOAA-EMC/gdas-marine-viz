#!/usr/bin/env python3

import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Files
hsi_output = Path("foo")
result_file = Path("last_dir.txt")
pslot = "cp4.03-parallel-hybrid"
HPSS_root = Path("/NCEPDEV/emc-global/1year/john.steffen/WCOSS2/scratch")
#HPSS_dir = Path(f"/NCEPDEV/emc-global/1year/john.steffen/WCOSS2/scratch/{pslot}")
HPSS_dir = Path(f"/5year/NCEPDEV/emc-global/emc.glopara/WCOSS2/GFSv17/retrov17_01_realtime")


with result_file.open("r", encoding="utf-8") as f:
    first_line = f.readline()
    first_cycle_str = f.readline().strip()

# Validate
if len(first_cycle_str) != 10 or not first_cycle_str.isdigit():
    raise ValueError(
        f"Invalid cycle string '{first_cycle_str}'; expected YYYYMMDDHH"
    )

# Parse into datetime (UTC assumed)
first_cycle = datetime.strptime(first_cycle_str, "%Y%m%d%H")

print(f"Parsed cycle datetime: {first_cycle} (UTC assumed)")

# Command to run (output goes to file "foo")
cmd = [
    "hsi",
    "-O", str(hsi_output),
    "ls", "-1",
    HPSS_dir
]

# Run the command
subprocess.run(cmd, check=True)

# Read the paths from the file written by hsi
lines = [
    line.strip()
    for line in hsi_output.read_text().splitlines()
    if line.strip()
]

if not hsi_output.exists():
    raise FileNotFoundError(f"{hsi_output} was not created by hsi")

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

    for cmd in commands:
        subprocess.run(cmd, check=True)

    cycle += timedelta(hours=6)


# Write result for later use
result_file.write_text(f"{first_cycle_str}\n{last_cycle_str}\n")

