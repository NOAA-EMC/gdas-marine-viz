#!/usr/bin/env python3
"""Compute every verification diagnostic for one or more cycles.

This is the only expensive step: it reads the multi-GB ensemble-variance and
increment files and the full observation diagnostics, and reduces them to a
small JSON (scalars, profiles, histograms) plus an NPZ of subsampled maps.
Everything downstream reads only the cache, so re-plotting is instant.

    python3 compute_cycle.py                 # every cycle in experiments.yaml
    python3 compute_cycle.py --cycle 2026042206
    python3 compute_cycle.py --force         # recompute cached cycles
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lv_obsspace  # noqa: E402
import lv_plot  # noqa: E402
import lv_statespace  # noqa: E402
from lv_common import (Grid, ObsSet, join_obs, load_config,  # noqa: E402
                       open_workdir, release_workdir,
                       select_experiments)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f else f          # NaN is not valid JSON
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def cycle_is_present(cfg, cycle):
    """Report which experiments have a usable directory for this cycle.

    Over a week of cycles some will be absent or half-written; those must be
    skipped without leaving a hollow cache file that later poisons the plots
    and the scorecard.
    """
    missing = [e.name for e in cfg['experiments']
               if not os.path.isdir(e.dir_for(cycle))]
    return missing


def observation_block(cfg, cycle, work, verbose=True):
    """Join every experiment's observations and reduce them.

    Shared by a full pass and by --rejoin, which needs exactly this and
    nothing else.

    An obs type is joined across whichever experiments PRESENT this cycle
    actually have it, not every experiment registered in the config, and not
    every experiment present this cycle either. Requiring every present
    experiment to share an obs type would silently drop it from the cache
    the moment even one experiment lacks it (a different QC config, a sensor
    one system doesn't assimilate) -- exactly the obs types most worth
    comparing. Each type's 'own' stats always cover whoever has it; its
    'common' stats cover the join across that same subset, so the sample is
    honest about who it actually spans (see common_experiments and
    lv_plot.type_sample, which reads it).
    """
    present = [e for e in cfg['experiments']
              if e.name not in cycle_is_present(cfg, cycle)]
    per_type = {}
    for e in present:
        try:
            files = e.obs_files(cycle, work)
        except FileNotFoundError as err:
            print('  ! %s: %s' % (e.name, err))
            continue
        for t, p in files.items():
            per_type.setdefault(t, {})[e.name] = p

    out, skipped = {}, []
    for obstype in sorted(per_type):
        paths = per_type[obstype]
        if verbose:
            print('  obs  %-32s' % obstype, end='', flush=True)
        own = {n: ObsSet(p, obstype) for n, p in paths.items()}
        aligned, counts, common_pass = join_obs(own)
        out[obstype] = lv_obsspace.compute(
            obstype, aligned, counts, common_pass, own, cfg)
        if verbose:
            n = list(counts.values())[0]['n_matched']
            missing = {e.name for e in present} - set(paths)
            note = ('  (missing from %s)' % sorted(missing)) if missing else ''
            print(' matched %8d  common-pass %8d%s'
                 % (n, common_pass.sum(), note))
        del own, aligned
    return out, skipped


def compute_cycle(cfg, cycle, grid, work, verbose=True):
    t0 = time.time()
    out = {'cycle': cycle,
           'experiments': {e.name: {'label': e.label, 'kind': e.kind}
                           for e in cfg['experiments']},
           'reference': cfg.get('reference'),
           # recorded so the plot layer subsamples the grid exactly the way
           # the cached maps were written, even if the config changes later
           'map_stride': int(cfg.get('map_stride', 2)),
           'obs': {}, 'state': {}}
    maps = {}

    out['obs'], _ = observation_block(cfg, cycle, work, verbose)

    for e in cfg['experiments']:
        if verbose:
            print('  state %-32s' % e.name, end='', flush=True)
        res, m = lv_statespace.compute(grid, e, cycle, cfg)
        out['state'][e.name] = res
        for k, v in m.items():
            maps['%s/%s' % (e.name, k)] = v
        if verbose:
            realms = [k for k in res if isinstance(res[k], dict)]
            bkg = [k for k in realms if res[k].get('has_background')]
            print(' %d realms, %d maps%s' % (len(realms), len(m),
                  (', bkg %s' % '+'.join(bkg)) if bkg else ', no background'))

    out['elapsed_sec'] = round(time.time() - t0, 1)
    return out, maps


def rejoin_one(cfg, cycle, work, verbose=True):
    """Recompute only the observation block, for every registered experiment.

    Merging caches from separate runs cannot rebuild the common sample, because
    each run only ever joined its own experiment. This re-reads the observation
    files (~650 MB per cycle against ~14 GB for a full pass) and rewrites the
    obs block with a genuine cross-experiment join, leaving the cached state
    and maps untouched.
    """
    jpath = os.path.join(cfg['cache'], '%s.json' % cycle)
    merged = None
    for root in cfg['caches']:
        p = os.path.join(root, '%s.json' % cycle)
        if os.path.exists(p):
            with open(p) as f:
                merged = lv_plot.merge_cycle(merged, json.load(f))
    if merged is None:
        return 'absent', 'nothing cached for this cycle'

    obs, _ = observation_block(cfg, cycle, work, verbose)
    if not obs:
        return 'empty', 'no observation types resolved'
    merged['obs'] = obs
    merged['experiments'] = {e.name: {'label': e.label, 'kind': e.kind}
                             for e in cfg['experiments']}
    merged['reference'] = cfg.get('reference')
    tmp = jpath + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(_jsonable(merged), f, indent=1)
    os.replace(tmp, jpath)
    return 'ok', '%s (obs block rejoined)' % jpath


def run_one(cfg, cycle, grid, work, force=False, verbose=True,
            keep_work=False):
    """Compute and cache one cycle. Returns (status, detail)."""
    jpath = os.path.join(cfg['cache'], '%s.json' % cycle)
    if os.path.exists(jpath) and not force:
        return 'cached', 'already computed (--force to redo)'

    # NOT gated on cycle_is_present() here: that only checks each
    # experiment's OWN analysis directory for this cycle, but a var-kind
    # experiment's background is looked up under a DIFFERENT (earlier)
    # cycle's directory entirely (see Experiment.background). An experiment
    # with no analysis of its own this cycle can still have a real
    # background -- 3dvar-rt fetched only 00Z, so 06/12/18Z have no analysis
    # directory, but DO have a background whenever the previous 00Z cycle's
    # forecast is on disk. compute_cycle() already handles a missing
    # directory per lookup (each returns None, nothing crashes); gating the
    # whole cycle up front on analysis-directory presence discarded that
    # background before it was ever looked for.
    out, maps = compute_cycle(cfg, cycle, grid, work, verbose)
    has_state = any(r.get('has_background') or r.get('has_increment')
                    for exp_state in out['state'].values()
                    for r in exp_state.values() if isinstance(r, dict))
    if not out['obs'] and not has_state:
        missing = cycle_is_present(cfg, cycle)
        if missing:
            return 'absent', 'no directory for %s' % ', '.join(missing)
        return 'empty', 'no observation types resolved; nothing cached'

    tmp = jpath + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(_jsonable(out), f, indent=1)
    os.replace(tmp, jpath)                 # never leave a half-written cache
    if maps:
        mp = os.path.join(cfg['cache'], '%s_maps.npz' % cycle)
        np.savez_compressed(mp + '.tmp.npz', **maps)
        os.replace(mp + '.tmp.npz', mp)
    if not keep_work:
        # the extracted observations are ~1 GB per cycle and nothing
        # downstream reads them; only the cache is needed from here on
        release_workdir(work, cycle, [e.name for e in cfg['experiments']])
    return 'ok', '%s (%.0fs)' % (jpath, out['elapsed_sec'])


def _rejoin_worker(args):
    """Top-level so it pickles; each process loads its own config and workdir.

    Mirrors _worker()'s reasoning below: every option that shapes the config
    must be carried here explicitly -- a parallel --rejoin silently dropping
    --experiment would rejoin the full registry instead of the requested
    subset -- and each process needs its own workdir handle rather than
    sharing the parent's (not picklable, and the observation extraction it
    manages is per-process scratch space anyway).
    """
    config, root, outdir, cache, experiment, cycle, keep_work = args
    cfg = load_config(config, root, outdir, cache)
    cfg = select_experiments(cfg, experiment)
    work = open_workdir(cfg)
    try:
        status, detail = rejoin_one(cfg, cycle, work, verbose=False)
    except Exception as e:                 # one bad cycle must not kill the run
        status, detail = 'failed', '%s: %s' % (type(e).__name__, str(e)[:120])
    if status == 'ok' and not keep_work:
        release_workdir(work, cycle, [e.name for e in cfg['experiments']])
    return cycle, status, detail


def _worker(args):
    """Top-level so it pickles; each process loads its own config and grid.

    Every option that shapes the config must be carried here explicitly. It
    used to take neither `cache` nor `experiment`, so `--jobs > 1` silently
    dropped both: a per-experiment precompute computed every experiment into
    what was meant to be one experiment's cache. The parallel branch needs more
    than one cycle, so a single-cycle test never reaches it.
    """
    (config, root, outdir, cache, experiment,
     cycle, force, no_maps, keep_work) = args
    cfg = load_config(config, root, outdir, cache)
    cfg = select_experiments(cfg, experiment)
    if no_maps:
        cfg['map_levels'] = []
    grid = Grid(cfg['grid'])
    work = open_workdir(cfg)
    try:
        return (cycle,) + run_one(cfg, cycle, grid, work, force,
                                  verbose=False, keep_work=keep_work)
    except Exception as e:                 # one bad cycle must not kill the run
        return cycle, 'failed', '%s: %s' % (type(e).__name__, str(e)[:120])


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
    ap.add_argument('--cache', action='append', default=None,
                    help='extra cache directory to read and merge (repeatable); '
                         'the first is where new results are written')
    ap.add_argument('--cycle', action='append', default=None,
                    help='cycle YYYYMMDDHH (repeatable); default: all in config')
    ap.add_argument('--force', action='store_true',
                    help='recompute cycles that are already cached')
    ap.add_argument('--no-maps', action='store_true',
                    help='skip the 2-D map extraction (faster)')
    ap.add_argument('--keep-work', action='store_true',
                    help='keep the extracted observations after caching '
                         '(~1 GB per cycle); they are removed by default')
    ap.add_argument('--rejoin', action='store_true',
                    help='recompute only the observation block for the full '
                         'registry and merge it into the cache; use after '
                         'merging caches from separate runs')
    ap.add_argument('--experiment', action='append', default=None,
                    help='restrict to this experiment (repeatable); the whole '
                         'registry is used when omitted')
    ap.add_argument('--jobs', type=int, default=1,
                    help='cycles to compute in parallel (also applies to '
                         '--rejoin); they are independent, and the work is '
                         'I/O bound')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt

    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    cfg = select_experiments(cfg, a.experiment)
    cycles = [str(c) for c in a.cycle] if a.cycle else cfg['cycles']
    os.makedirs(cfg['cache'], exist_ok=True)
    print('%d cycle(s), cache %s' % (len(cycles), cfg['cache']))

    results = []
    parallel = a.jobs > 1 and len(cycles) > 1
    if a.rejoin and parallel:
        # rejoin re-reads observation files (~650 MB/cycle) but touches
        # neither the grid nor cached state/maps -- the same
        # independent-and-I/O-bound reasoning --jobs already applies to a
        # full compute, just against rejoin_one() instead of run_one().
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(a.config, a.root, a.outdir, a.cache, a.experiment,
                  c, a.keep_work) for c in cycles]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for cycle, status, detail in ex.map(_rejoin_worker, tasks):
                results.append((cycle, status, detail))
                print('  %-11s %-8s %s' % (cycle, status, detail))
    elif a.rejoin:
        work = open_workdir(cfg)
        for cycle in cycles:
            print('%s:' % cycle)
            try:
                status, detail = rejoin_one(cfg, cycle, work)
            except Exception as e:
                status, detail = 'failed', '%s: %s' % (type(e).__name__, e)
            if status == 'ok' and not a.keep_work:
                release_workdir(work, cycle, [e.name for e in cfg['experiments']])
            results.append((cycle, status, detail))
            print('  %s: %s' % (status, detail))
    elif parallel:
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(a.config, a.root, a.outdir, a.cache, a.experiment,
                  c, a.force, a.no_maps, a.keep_work) for c in cycles]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for cycle, status, detail in ex.map(_worker, tasks):
                results.append((cycle, status, detail))
                print('  %-11s %-8s %s' % (cycle, status, detail))
    else:
        grid = Grid(cfg['grid'])
        work = open_workdir(cfg)
        if a.no_maps:
            cfg['map_levels'] = []
        for cycle in cycles:
            print('%s:' % cycle)
            try:
                status, detail = run_one(cfg, cycle, grid, work, a.force,
                                         keep_work=a.keep_work)
            except Exception as e:
                status, detail = 'failed', '%s: %s' % (type(e).__name__, e)
            results.append((cycle, status, detail))
            print('  %s: %s' % (status, detail))

    tally = {}
    for _, s, _d in results:
        tally[s] = tally.get(s, 0) + 1
    print('\nsummary: ' + ', '.join('%d %s' % (v, k)
                                    for k, v in sorted(tally.items())))
    bad = [(c, d) for c, s, d in results if s in ('failed', 'empty')]
    for c, d in bad:
        print('  ! %s %s' % (c, d))
    return 1 if any(s == 'failed' for _, s, _d in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
