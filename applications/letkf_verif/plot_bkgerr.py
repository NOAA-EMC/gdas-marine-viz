#!/usr/bin/env python3
"""Maps and sections of the parametric background error, D in B = K D C D K^T.

For each drawn cycle and each control variable in the D files (lv_bkgerr):

    maps        ocean: one row per `map_levels:` level, one column per
                experiment; ice: one figure per hemisphere
    sections    the 3-D ocean variables along the configured `sections:`
                transects, one figure per `sections: depth_views:` entry

D is a standard deviation, so every panel is on a sequential scale from zero,
shared across the experiments in a row (maps) or across every transect
(sections) -- the same sharing rule the background sections use, and the
same colormap (plot_statespace._field_cmap).

Cycles are chosen like the other per-date stages: every configured cycle at
`--hours` plus the latest. The D files are read directly, not through the
precompute cache, so turning this section on costs no re-precompute.

Run with the gdas-marine-viz venv and OMP_NUM_THREADS=1.

    python3 plot_bkgerr.py experiments.yaml --outdir page --hours 00
"""

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')

import lv_bkgerr as B  # noqa: E402
import lv_statespace as S  # noqa: E402
import plot_statespace as PS  # noqa: E402
from lv_common import Fresh, Grid, load_config  # noqa: E402


def _level_label(depth, var, k, nk):
    if nk <= 1:
        return var
    if depth is not None and k < len(depth):
        z = float(depth[k])
        return '%s\nlevel %d\n(~%s m)' % (var, k,
                                          ('%.0f' % z) if z >= 10 else
                                          ('%.1f' % z))
    return '%s\nlevel %d' % (var, k)


def _upper(fields, q=99.0):
    vals = [f[np.isfinite(f)].ravel() for f in fields
            if f is not None and np.any(np.isfinite(f))]
    if not vals:
        return 1.0
    return float(np.percentile(np.concatenate(vals), q)) or 1.0


def draw_cycle(cfg, grid, cycle, files, names, levels, lines, stride):
    """Every figure for one cycle. ``files`` is {(exp, realm): path}."""
    out = []
    PS.TAG = '_%s' % cycle
    exps = {e.name: e for e in cfg['experiments']}

    # The depth axis, for labels and sections: the first experiment with an
    # ocean background at this cycle.
    bkg = next((p for p in (exps[n].background(cycle, 'ocean') for n in names)
                if p), None)
    depth = S.depth_from_h(grid, bkg) if bkg else None

    for realm in B.REALMS:
        srcs = {}
        for n in names:
            p = files.get((n, realm))
            if p:
                srcs[n] = B.BkgerrSource(p, grid)
        if not srcs:
            continue
        try:
            variables = B.sort_vars({v for n in srcs
                                     for v in B.variables(files[(n, realm)])})
            if realm == 'ocean' and lines:
                axes_raw, wet = S.section_geometry(grid, bkg, lines, stride)
                axes = {}
                for key, val in axes_raw.items():
                    what, _, line = key.partition('_')
                    x, d = axes.get(line, (None, None))
                    axes[line] = (val if what == 'x' else x,
                                  val if what == 'depth' else d)
            for var in variables:
                unit = B.UNITS.get(var, '')
                nk = max(srcs[n].levels(var) or 0 for n in srcs)
                ks = [k for k in levels if k < nk] if nk > 1 else [0]

                # -- maps -------------------------------------------------
                rows, limits = [], {}
                for k in ks:
                    fields = []
                    for n in names:
                        f = (srcs[n].level(var, k) if n in srcs
                             and srcs[n].levels(var) else None)
                        if f is not None:
                            f = np.where(grid.mask, f, np.nan)[::stride,
                                                              ::stride]
                        fields.append(f)
                    key = '%s_k%d' % (var, k)
                    rows.append((key, fields))
                    limits[key] = (0.0, _upper(fields), PS._field_cmap(var))
                labels = {'%s_k%d' % (var, k): _level_label(depth, var, k, nk)
                          for k in ks}
                for view in PS.views_for(realm):
                    p = PS.map_grid(
                        cfg, grid, rows, names,
                        'parametric background error D (std dev): %s, %s'
                        % (var, PS.cycle_row_label(cycle)),
                        'bkgerr_maps_%s_%s.png' % (realm, var), view,
                        limits=limits, cb_label='sigma_b %s (%s)' % (var, unit),
                        row_label=lambda key: labels.get(key, key))
                    if p:
                        out.append(p)

                # -- sections -------------------------------------------------
                if realm != 'ocean' or nk <= 1 or not lines:
                    continue
                per = {}
                for n in names:
                    if n not in srcs or not srcs[n].levels(var):
                        continue
                    for key, plane in S.section_planes(
                            grid, srcs[n], [var], lines,
                            stride=stride).items():
                        m = wet.get(key.rpartition('_')[2])
                        per[(n, key.rpartition('_')[2])] = (
                            plane if m is None else
                            np.where(m, plane, np.nan).astype('f4'))
                srows = [(ln[0], [per.get((n, ln[0])) for n in names])
                         for ln in lines]
                lim = (0.0, _upper([f for _k, fs in srows for f in fs]),
                       PS._field_cmap(var))
                for cut, suffix, label in PS.section_depth_views(cfg):
                    p = PS.section_grid(
                        cfg, srows, names, axes,
                        'parametric background error D (std dev) along '
                        'vertical sections - %s (%s) - %s'
                        % (var, label, PS.cycle_row_label(cycle)),
                        'bkgerr_sections_%s%s.png' % (var, suffix), lim,
                        cb_label='sigma_b %s (%s)' % (var, unit),
                        depth_cut=cut, note=PS._tripolar_note)
                    if p:
                        out.append(p)
        finally:
            for s in srcs.values():
                s.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config', nargs='?', default=None)
    ap.add_argument('--config', dest='config_opt', default=None)
    ap.add_argument('--root', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--cache', action='append', default=None)
    ap.add_argument('--cycle', default=None,
                    help='draw this one cycle (overrides --hours)')
    ap.add_argument('--hours', default='00',
                    help='draw every configured cycle at these UTC hours '
                         '(comma-separated), plus the latest; "all" for '
                         'every cycle')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt

    t_all = time.time()
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    if a.cycle:
        todo = [str(a.cycle)]
    else:
        todo = [str(c) for c in cfg['cycles']]
        if a.hours.lower() != 'all':
            hours = {h.strip().zfill(2) for h in a.hours.split(',')
                     if h.strip()}
            todo = [c for c in todo if c[8:10] in hours or c == todo[-1]]

    names = [e.name for e in cfg['experiments']]
    plan = {}
    for c in todo:
        files = {(e.name, r): B.bkgerr_file(e, c, r)
                 for e in cfg['experiments'] for r in B.REALMS}
        files = {k: v for k, v in files.items() if v}
        if files:
            plan[c] = files
    if not plan:
        print('background error: no *bkgerr_parametric_stddev.nc under any '
              'experiment for the selected cycles -- nothing drawn',
              flush=True)
        return 0

    grid = Grid(cfg['grid'])
    stride = int(cfg.get('map_stride', 2))
    levels = [int(k) for k in (cfg.get('map_levels') or [0])]
    lines = S.section_lines(grid, cfg, stride)
    os.makedirs(cfg['figs'], exist_ok=True)
    fresh = Fresh(cfg, 'bkgerr', force=a.force, script=__file__)
    print('background error: %d cycle(s), %d experiment(s)'
          % (len(plan), len(names)), flush=True)

    written = []
    for c, files in plan.items():
        inputs = sorted(files.values()) + [cfg['grid'], B.__file__]
        params = {'names': names, 'levels': levels, 'stride': stride,
                  'lines': [ln[0] for ln in lines],
                  'views': [v[1] for v in PS.section_depth_views(cfg)]}
        if fresh.ok('cycle:%s' % c, inputs, params):
            print('  %s: up to date' % c, flush=True)
            continue
        t = time.time()
        new = draw_cycle(cfg, grid, c, files, names, levels, lines, stride)
        fresh.record('cycle:%s' % c, inputs, params, new)
        written += new
        print('  %s: %d figure(s) in %.0fs' % (c, len(new), time.time() - t),
              flush=True)
    fresh.save()
    print('background error: %d figure(s) in %.0fs'
          % (len(written), time.time() - t_all), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
