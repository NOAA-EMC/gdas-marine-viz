#!/usr/bin/env python3
"""Precompute the verification cache for ONE experiment.

This is the expensive half of the suite: it reads the multi-GB ensemble
variance, increment, background and observation files and reduces them to a
small per-cycle cache. Everything the comparison page needs afterwards is read
from that cache, so this runs once per experiment and never again unless the
experiment changes.

One experiment per invocation, each writing its own output directory, so they
can run as independent jobs:

    python3 precompute_experiment.py experiments.yaml \\
            --experiment 3dvar_ctl --outdir /scratch/me/verif/3dvar &
    python3 precompute_experiment.py experiments.yaml \\
            --experiment letkf_v1  --outdir /scratch/me/verif/letkf &
    wait

Then build the page across the caches with build_comparison.py.

The cache this writes records that it joined only its own experiment, so the
observation statistics in it are that experiment's own sample. Comparing two of
them needs the cross-experiment join that build_comparison.py rebuilds; it will
say so if you skip it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compute_cycle  # noqa: E402
import preflight  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Precompute the verification cache for one experiment.')
    ap.add_argument('config', nargs='?', default=None,
                    help='path to the experiments yaml (or use --config)')
    ap.add_argument('--config', dest='config_opt', default=None,
                    help='same as the positional argument')
    ap.add_argument('--experiment', required=True,
                    help='which experiment in the registry to precompute')
    ap.add_argument('--outdir', default=None,
                    help='where this experiment\'s cache goes; give each '
                         'experiment its own so parallel runs do not collide')
    ap.add_argument('--root', default=None,
                    help='override the experiment root (or $LETKF_VERIF_ROOT)')
    ap.add_argument('--cycle', action='append', default=None,
                    help='compute only this cycle (repeatable)')
    ap.add_argument('--force', action='store_true',
                    help='recompute cycles already cached')
    ap.add_argument('--jobs', type=int, default=1,
                    help='cycles to compute in parallel; they are independent')
    ap.add_argument('--skip-preflight', action='store_true',
                    help='do not check the environment and inputs first')
    a = ap.parse_args(argv)
    cfg_path = a.config or a.config_opt

    common = []
    if cfg_path:
        common.append(cfg_path)
    for flag, val in (('--outdir', a.outdir), ('--root', a.root)):
        if val:
            common += [flag, val]

    if not a.skip_preflight:
        print('== preflight: %s ==' % a.experiment, flush=True)
        # Scoped to this experiment: a sibling in the same registry may not
        # exist yet, and that must not block precomputing this one. Cheap, and
        # it fails on a missing grid or an unresolvable cycle before the
        # expensive reads start rather than an hour into them.
        if preflight.main(common + ['--experiment', a.experiment]) != 0:
            print('\npreflight failed; fix the above or pass --skip-preflight')
            return 1

    argv_c = common + ['--experiment', a.experiment, '--jobs', str(a.jobs)]
    if a.force:
        argv_c.append('--force')
    for c in a.cycle or []:
        argv_c += ['--cycle', c]

    print('\n== precompute: %s ==' % a.experiment, flush=True)
    rc = compute_cycle.main(argv_c)
    if rc == 0:
        print('\nDone. Build the comparison page with:\n'
              '  python3 build_comparison.py %s --cache <this cache> '
              '--cache <other cache> --outdir <page dir>'
              % (cfg_path or '<config>'))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
