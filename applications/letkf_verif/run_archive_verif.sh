#!/bin/bash
# Cron entry point; arguments are forwarded to archive_and_verify.py.
set -euo pipefail
set +u
source /etc/profile.d/modules.sh
set -u
module purge
module use "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)/modulefiles"
module load EVA/ursa
APP=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
exec python "$APP/archive_and_verify.py" "$@"
