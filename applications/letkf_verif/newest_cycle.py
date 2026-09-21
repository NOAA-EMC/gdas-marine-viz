#!/usr/bin/env python3
"""The newest complete cycle on disk for a config's experiments -- and,
with --set-stop, advance the config's `cycles: stop:` to it.

Driven by the same registry the verification uses (stems, background
offsets, increment patterns), so it works for any experiments.yaml. A cycle
is COMPLETE for an experiment when its analysis directory exists, its ocean
increment resolves, and the background valid at the NEXT cycle (that cycle's
own f006 forecast) resolves -- the forecast following the analysis is what
the next cycle's verification needs, and a cycle whose forecast is still
running would otherwise be scored half-written and, worse, cached that way.

    python3 newest_cycle.py experiments.yaml            # report per experiment
    python3 newest_cycle.py experiments.yaml --set-stop # and edit `stop:`

Exit 0 when a cycle newer than the config's stop exists (printed on the last
line as YYYYMMDDHH), 3 when nothing newer is complete, 2 on a config that
uses an explicit cycle list (only the {start, stop, step} form can be
advanced). --all requires every experiment to be complete; the default
(--any) takes the newest cycle any of them has, which is what a comparison
with an archived reference wants: the newer cycles run with the experiments
that have them.
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml  # noqa: E402
from lv_common import load_config, parse_cycle  # noqa: E402

CYCLE_FMT = '%Y%m%d%H'


def complete(exp, cycle, step_hours):
    """Whether ``cycle`` can be verified for this experiment now."""
    if not os.path.isdir(exp.dir_for(cycle)):
        return False
    if exp.increment(cycle, 'ocean') is None:
        return False
    nxt = (parse_cycle(cycle) + timedelta(hours=step_hours)).strftime(CYCLE_FMT)
    return (exp.background(nxt, 'ocean') is not None
            or exp.background(nxt, 'ice') is not None)


def newest(exp, start, step_hours, horizon):
    """Newest complete cycle at or after ``start``, scanning to ``horizon``;
    None when none is. Scans forward so a gap does not stop the search."""
    t, best = parse_cycle(start), None
    step = timedelta(hours=step_hours)
    while t <= horizon:
        c = t.strftime(CYCLE_FMT)
        if complete(exp, c, step_hours):
            best = c
        t += step
    return best


def set_stop(path, cycle):
    """Rewrite only the `stop:` value of the yaml's cycles block, keeping
    comments and layout; the loader re-validates on the next read."""
    text = open(path).read()
    new, n = re.subn(r'^(\s*stop:\s*)(\S+)', r'\g<1>%s' % cycle, text,
                     count=1, flags=re.M)
    if n != 1:
        raise SystemExit('%s: could not find a single `stop:` line' % path)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(new)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config')
    ap.add_argument('--set-stop', action='store_true',
                    help='advance `cycles: stop:` in the config to the result')
    ap.add_argument('--all', action='store_true',
                    help='require every experiment to be complete at the cycle')
    ap.add_argument('--horizon-days', type=float, default=45.0,
                    help='how far past the current stop to look (default 45)')
    a = ap.parse_args(argv)

    with open(a.config) as fh:
        raw = yaml.safe_load(fh)
    spec = raw.get('cycles')
    if not isinstance(spec, dict):
        print('%s: `cycles:` is an explicit list; only the {start, stop, step} '
              'form can be advanced' % a.config)
        return 2
    step_hours = float(spec.get('step', 6))
    stop = str(spec['stop'])
    cfg = load_config(a.config, None, None, None)
    horizon = min(datetime.utcnow(),
                  parse_cycle(stop) + timedelta(days=a.horizon_days))

    per_exp = {}
    for e in cfg['experiments']:
        per_exp[e.name] = newest(e, stop, step_hours, horizon)
        print('  %-28s newest complete cycle: %s'
              % (e.name, per_exp[e.name] or 'none at or after %s' % stop))
    found = [c for c in per_exp.values() if c]
    if not found or (a.all and len(found) < len(per_exp)):
        print('nothing complete beyond stop %s' % stop)
        return 3
    cycle = min(found) if a.all else max(found)
    if cycle <= stop:
        print('stop %s is already the newest complete cycle' % stop)
        return 3
    if a.set_stop:
        set_stop(a.config, cycle)
        print('%s: stop %s -> %s' % (a.config, stop, cycle))
    print(cycle)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
