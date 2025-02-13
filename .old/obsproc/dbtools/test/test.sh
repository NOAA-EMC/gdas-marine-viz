#!/bin/bash
set -eu
../bin/godas_store_obs.py --start 2015-01-01T00:00:00Z \
                         --end 2015-01-03T00:00:00Z \
                         --source_dir ./obs-2-move \
                         --source_file ymd \
                         --provider jcsda_soca \
                         --experiment benchmark_v2 \
                         --obstype adt \
                         --platforms c2 j2
