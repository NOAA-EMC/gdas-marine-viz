#!/usr/bin/env python3
"""Check that this machine can run the suite, before you spend an hour on it.

Verifies the python stack, the grid files, the cartopy coastline data, and
that every experiment's files for every configured cycle actually resolve.

    python3 preflight.py
    python3 preflight.py --root /scratch/me/expts   # override experiment root
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, BAD, WARN = 'ok  ', 'FAIL', 'warn'
_status = {'fail': 0, 'warn': 0}


def say(state, label, detail=''):
    if state == BAD:
        _status['fail'] += 1
    elif state == WARN:
        _status['warn'] += 1
    print('  [%s] %-46s %s' % (state, label, detail))


def check_python():
    print('python stack')
    say(OK, 'interpreter', sys.executable)
    for mod, need in (('numpy', True), ('netCDF4', True), ('yaml', True),
                      ('matplotlib', True), ('cartopy', True),
                      ('scipy', False)):
        try:
            m = importlib.import_module(mod)
            say(OK, mod, getattr(m, '__version__', 'present'))
        except Exception as e:
            say(BAD if need else WARN, mod,
                ('required -- ' if need else 'optional -- ')
                + type(e).__name__ + ': ' + str(e)[:70])


def check_coastlines():
    print('coastline data (cartopy)')
    try:
        import cartopy
    except Exception:
        say(BAD, 'cartopy absent', 'required for the map figures')
        return
    root = os.path.join(cartopy.config['data_dir'], 'shapefiles',
                        'natural_earth', 'physical')
    missing = [n for n in ('ne_110m_land.shp', 'ne_110m_coastline.shp')
               if not os.path.exists(os.path.join(root, n))]
    if missing:
        say(BAD, 'natural_earth 110m shapefiles', 'missing %s in %s'
            % (', '.join(missing), root))
        print('         cartopy downloads these on first use; if the machine is '
              'offline, copy ~208 KB from one that has them:')
        print('         rsync -a <host>:%s/ne_110m_{land,coastline}.* %s/'
              % (root, root))
    else:
        say(OK, 'natural_earth 110m shapefiles', root)


def check_config(args):
    from lv_common import Grid, load_config, select_experiments
    print('configuration')
    try:
        cfg = load_config(args.config, args.root, args.outdir, args.cache)
        cfg = select_experiments(cfg, getattr(args, 'experiment', None))
    except Exception as e:
        say(BAD, 'load experiments.yaml', '%s: %s' % (type(e).__name__, e))
        return None
    say(OK, 'experiments.yaml',
        '%d experiments, %d cycles' % (len(cfg['experiments']),
                                       len(cfg['cycles'])))

    print('grid')
    p = cfg['grid']
    say(OK if os.path.exists(p) else BAD, 'grid',
        ('%.0f MB  %s' % (os.path.getsize(p) / 1e6, p))
        if os.path.exists(p) else 'not found: %s' % p)
    if not os.path.exists(cfg['grid']):
        return cfg
    try:
        g = Grid(cfg['grid'])
        say(OK, 'grid loads', '%d x %d, ocean area %.3e m2'
            % (g.shape[0], g.shape[1], float(g.wgt.sum())))
        # depth comes per cycle from the background; without one the profile
        # plots fall back to the level index
        bkg = any(e.background(cfg['cycles'][0], 'ocean')
                  for e in cfg['experiments']) if cfg['cycles'] else False
        say(OK if bkg else WARN, 'depth axis',
            'from background h, per cycle' if bkg
            else 'no background found; profiles will use the level index')
    except Exception as e:
        say(BAD, 'grid loads', '%s: %s' % (type(e).__name__, e))

    import lv_woa
    woa = lv_woa.configured(cfg)
    if woa is None:
        say(OK, 'WOA23 climatology', 'no `woa:` in the config; skipped')
    else:
        wvars = [v for v in (cfg.get('background_vars') or {}).get('ocean', [])
                 if v in lv_woa.VARS]
        missing = lv_woa.missing_files(woa, wvars) if wvars else []
        if not wvars:
            say(WARN, 'WOA23 climatology',
                'configured, but background_vars.ocean names neither %s'
                % ' nor '.join(lv_woa.VARS))
        elif missing:
            say(BAD, 'WOA23 climatology',
                '%d of %d expected file(s) absent under %s, first %s'
                % (len(missing), 13 * len(wvars), woa['path'],
                   os.path.basename(missing[0])))
        else:
            say(OK, 'WOA23 climatology',
                '%d files for %s under %s'
                % (13 * len(wvars), '+'.join(wvars), woa['path']))

    sec = cfg.get('sections') or {}
    if sec.get('zonal') or sec.get('meridional'):
        from lv_statespace import section_warnings
        bad = section_warnings(cfg)
        say(WARN if bad else OK, 'vertical sections',
            '%d zonal, %d meridional%s'
            % (len(sec.get('zonal') or []), len(sec.get('meridional') or []),
               '; %d north of the tripolar seam' % len(bad) if bad else ''))
        for w in bad:
            print('         %s' % w)
    else:
        # Said out loud rather than passed over: an absent `sections:` is a
        # legitimate choice, but it is indistinguishable in the finished
        # report from a broken one -- every section figure is simply never
        # written, with no error anywhere.
        say(OK, 'vertical sections',
            'no `sections:` in the config; section figures will be skipped')
    return cfg


def check_data(cfg):
    if cfg is None:
        return
    from lv_common import open_workdir
    print('experiment data')
    work = open_workdir(cfg)
    for cycle in cfg['cycles']:
        for e in cfg['experiments']:
            d = e.dir_for(cycle)
            if not os.path.isdir(d):
                say(BAD, '%s / %s' % (e.name, cycle), 'no such directory: %s' % d)
                continue
            try:
                files = e.obs_files(cycle, work)
            except Exception as ex:
                say(BAD, '%s / %s obs' % (e.name, cycle),
                    '%s: %s' % (type(ex).__name__, str(ex)[:80]))
                files = {}
            incr = [r for r in ('ocean', 'ice') if e.increment(cycle, r)]
            ens = [r for r in ('ocean', 'ice')
                   if e.ensvar(cycle, r, 'prior')]
            state = OK if files and incr else WARN
            say(state, '%s / %s' % (e.name, cycle),
                '%d obs types, increments %s%s'
                % (len(files), incr or 'NONE',
                   ', ensvar %s' % ens if ens else ''))


def check_increment_vars(cfg):
    """Does each experiment's increment file actually hold its state_vars?

    A pattern that resolves to the WRONG file still 'finds an increment' --
    it just contains none of the configured variables, so every panel built
    from it comes out empty with no error raised anywhere. That is invisible
    until someone reads the finished report, which is an expensive way to
    find out. One header read per experiment/realm catches it here instead.

    Checked on the first cycle that resolves, not every cycle: the filename
    pattern is the same for all of them, so one is representative.
    """
    if cfg is None:
        return
    from netCDF4 import Dataset
    print('increment contents')
    svars = cfg.get('state_vars', {})
    for e in cfg['experiments']:
        for realm in ('ocean', 'ice'):
            want = svars.get(realm) or []
            if not want:
                continue
            label = '%s / %s increment' % (e.name, realm)
            path = next((p for p in (e.increment(c, realm)
                                     for c in cfg['cycles']) if p), None)
            if path is None:
                say(WARN, label, 'no file matched in any configured cycle')
                continue
            try:
                with Dataset(path) as d:
                    have = [v for v in want if v in d.variables]
            except Exception as ex:
                say(BAD, label, '%s: %s' % (type(ex).__name__, str(ex)[:60]))
                continue
            if have:
                say(OK, label, '%s -- has %s'
                    % (os.path.basename(path), '+'.join(have)))
            else:
                say(BAD, label,
                    '%s holds NONE of %s -- check `increment_pattern`; every '
                    'panel built from it will be silently empty'
                    % (os.path.basename(path), ', '.join(want)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('config', nargs='?', default=None,
                    help='path to the experiments yaml (or use --config)')
    ap.add_argument('--config', dest='config_opt', default=None,
                    help='same as the positional argument')
    ap.add_argument('--root', default=None,
                    help='override every experiment root (or $LETKF_VERIF_ROOT)')
    ap.add_argument('--outdir', default=None,
                    help='send cache/figs/report/scorecard here '
                         '(or $LETKF_VERIF_OUTDIR)')
    ap.add_argument('--experiment', action='append', default=None,
                    help='check only this experiment (repeatable); useful '
                         'before a per-experiment precompute, where a sibling '
                         'experiment need not be present at all')
    ap.add_argument('--cache', action='append', default=None,
                    help='extra cache directory to read and merge (repeatable); '
                         'the first is where new results are written')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt

    check_python()
    check_coastlines()
    cfg = check_config(a)
    check_data(cfg)
    check_increment_vars(cfg)

    print()
    if _status['fail']:
        print('%d blocking problem(s); fix these before running compute_cycle.py'
              % _status['fail'])
    elif _status['warn']:
        print('ready, with %d warning(s) -- the suite will run and degrade '
              'gracefully' % _status['warn'])
    else:
        print('ready')
    return 1 if _status['fail'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
