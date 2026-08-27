#!/usr/bin/env python3
"""Build the comparison page from caches precomputed per experiment.

The cheap half of the suite: it reads only the caches that
precompute_experiment.py wrote, so it re-runs in about a minute and can be
re-run freely while tuning what the page shows.

    python3 build_comparison.py experiments.yaml \\
            --cache /scratch/me/verif/3dvar/cache \\
            --cache /scratch/me/verif/letkf/cache \\
            --outdir /scratch/me/verif/page

Stages: rejoin the observations, render the three figure sets, write the
scorecard, then assemble the HTML report.

Why the rejoin comes first
--------------------------
A cache precomputed for one experiment alone has no cross-experiment common
sample -- for it, "common" is just its own observations. Merging two such caches
and comparing them would score each experiment on a different set of
observations, which is exactly what the common sample exists to prevent: at the
sample cycle the LETKF thins sea-ice observations to ~9% of the 3DVar sample,
and the subset it keeps is the easier one, so an own-sample comparison
understates its disadvantage roughly five-fold.

So this re-reads only the observation files and rebuilds a genuine join across
every experiment on show -- seconds per cycle, against minutes for a full pass.
`--no-rejoin` skips it when those files are not reachable; the scorecard and the
report then say plainly that they are own-sample numbers.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_report  # noqa: E402
import compute_cycle  # noqa: E402
import plot_obsspace  # noqa: E402
import plot_statespace  # noqa: E402
import plot_timeseries  # noqa: E402
import scorecard  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Build the comparison page from precomputed caches.')
    ap.add_argument('config', nargs='?', default=None,
                    help='path to the experiments yaml (or use --config)')
    ap.add_argument('--config', dest='config_opt', default=None,
                    help='same as the positional argument')
    ap.add_argument('--cache', action='append', default=None,
                    help='a precomputed cache directory (repeatable); all are '
                         'read and merged')
    ap.add_argument('--outdir', default=None,
                    help='where the figures, scorecard and report go')
    ap.add_argument('--root', default=None,
                    help='override the experiment root (or $LETKF_VERIF_ROOT)')
    ap.add_argument('--no-rejoin', action='store_true',
                    help='do not rebuild the cross-experiment observation '
                         'join; scores fall back to each experiment\'s own '
                         'sample and are confounded by thinning and QC')
    ap.add_argument('--skip', action='append', default=None,
                    choices=['rejoin', 'obsspace', 'statespace', 'timeseries',
                             'scorecard', 'report'],
                    help='skip a stage (repeatable)')
    a = ap.parse_args(argv)
    cfg_path = a.config or a.config_opt
    skip = set(a.skip or [])
    if a.no_rejoin:
        skip.add('rejoin')

    common = []
    if cfg_path:
        common.append(cfg_path)
    for flag, val in (('--outdir', a.outdir), ('--root', a.root)):
        if val:
            common += [flag, val]
    # The page's own cache goes FIRST, because the first entry is where new
    # results are written and the rejoin writes a merged cycle. Passing the
    # per-experiment caches first would overwrite one experiment's precomputed
    # cache with a merged one, quietly destroying the input to this step.
    if a.outdir:
        common += ['--cache', os.path.join(a.outdir, 'cache')]
    for c in a.cache or []:
        common += ['--cache', c]

    stages = [
        ('rejoin', 'rejoining observations across experiments',
         lambda: compute_cycle.main(common + ['--rejoin'])),
        ('obsspace', 'observation-space figures',
         lambda: plot_obsspace.main(common)),
        ('statespace', 'state-space figures',
         lambda: plot_statespace.main(common)),
        ('timeseries', 'cycling figures',
         lambda: plot_timeseries.main(common)),
        ('scorecard', 'scorecard', lambda: scorecard.main(common)),
        ('report', 'HTML report', lambda: build_report.main(common)),
    ]
    todo = [s for s in stages if s[0] not in skip]
    for i, (name, label, run) in enumerate(todo, 1):
        print('\n== [%d/%d] %s ==' % (i, len(todo), label), flush=True)
        rc = run()
        if rc:
            print('\n%s failed (exit %s)' % (name, rc))
            return rc

    if 'rejoin' in skip:
        print('\nNote: the observation join was skipped, so the scores are on '
              'each experiment\'s OWN sample and are confounded by differences '
              'in thinning and QC. Re-run without --no-rejoin where the '
              'observation files are reachable.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
