#!/usr/bin/env python3
"""Binned-departure and regression figures. Reads '<cycle>_obsbins.npz'.

Per obs type, three figure families, each drawn for every --hours cycle and
once for every cached cycle pooled ('all'):

    figs/obsbins_map_<type>[_<cycle>|_all].png
        one row per statistic (obs count, mean O-B, RMS O-B, mean O-A,
        RMS O-A, assigned and effective obs error, RMS(O-B) over each),
        one column per experiment -- on the 1-degree bins compute_cycle.py
        wrote. Ice types are drawn polar.
    figs/obsbins_reg_<type>[_<cycle>|_all].png
        observation against background and against analysis, as a log-
        density histogram with the 1:1 line and the least-squares fit.
    figs/obsbins_sec_<type>[_<cycle>|_all].png   (profile types)
        the map columns on depth x latitude instead.

Colour scales are set from the pooled figure and reused for every date of
that type, so stepping through dates in the report compares like with like.
An experiment with nothing at a date keeps its row, blank and labelled, so
its absence reads as absence rather than as a shifted layout.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
import lv_obsbins as B  # noqa: E402
import lv_plot as P  # noqa: E402
import plot_statespace as PS  # noqa: E402
from lv_common import Fresh, load_config  # noqa: E402
from lv_obsspace import is_ice, is_profile  # noqa: E402

TAG = ''
DPI = 100

# (key in lv_obsbins.derived, title, scale kind)
COLUMNS = [('count', 'obs count', 'count'),
           ('ombg_mean', 'mean O$-$B', 'mean'),
           ('ombg_rms', 'RMS O$-$B', 'rms'),
           ('oman_mean', 'mean O$-$A', 'mean'),
           ('oman_rms', 'RMS O$-$A', 'rms'),
           ('r_rms', 'assigned obs error', 'err_r'),
           # EffectiveError1 (0 when a file has no final one): the error the
           # analysis actually weighted with, after QC and any inflation --
           # the ratio against it is the one the DA "sees"
           ('reff_rms', 'effective obs error', 'err_reff'),
           ('ombg_over_r', 'RMS(O$-$B) / assigned error', 'ratio'),
           ('ombg_over_reff', 'RMS(O$-$B) / effective error', 'ratio')]


def _save(fig, cfg, base):
    stem, ext = os.path.splitext(base)
    return P.save(fig, cfg, '%s%s%s' % (stem, TAG, ext), dpi=DPI, quiet=True)


def _hemi(obstype):
    if not is_ice(obstype) and 'icefb' not in obstype:
        return None
    return 'sh' if obstype.endswith('south') else 'nh'


def _round(v):
    """Two significant digits: keeps the scale stable across reruns while a
    cycle's worth of new data barely moves the pooled percentile."""
    if not np.isfinite(v) or v == 0:
        return v
    e = np.floor(np.log10(abs(v)))
    return float(np.round(v / 10 ** (e - 1)) * 10 ** (e - 1))


def span(field):
    """(lo, hi) spanning the 2nd-98th percentile of one field, so a
    nearly-flat error (a 0.1 m ADT sigma-o everywhere, with structure of a
    few per cent) still uses the whole ramp; a field that is exactly
    constant gets a small symmetric band around its value."""
    v = field[np.isfinite(field)]
    if not v.size:
        return (0.0, 1.0)
    lo, hi = float(np.percentile(v, 2)), float(np.percentile(v, 98))
    if hi - lo < 1e-3 * max(abs(hi), 1e-12):
        mid = 0.5 * (lo + hi)
        return (_round(mid * 0.9), _round(mid * 1.1) or 1.0)
    return (_round(lo), _round(hi))


# Statistics whose colour scale is set per PANEL, from that panel's own
# spread, with a colourbar of its own: the observation errors. Two
# experiments can assign errors two orders of magnitude apart (0.1 m against
# metres for ADT), and on any scale shared between them one is flat; what
# is wanted from these rows is the STRUCTURE of each system's sigma-o, and
# the numbers on the bar carry the comparison between them.
PER_PANEL = ('err_r', 'err_reff')


def scales(derived_all):
    """Colour limits per column kind from the pooled fields of one type."""
    def pct(keys, q):
        v = np.concatenate([d[k][np.isfinite(d[k])].ravel()
                            for d in derived_all for k in keys if k in d]
                           or [np.zeros(1)])
        return float(np.percentile(np.abs(v), q)) if v.size else 1.0
    # The departures share one scale (rms) across mean/RMS O-B and O-A; an
    # assigned ADT error of metres on that 0.1 m scale flattened the O-B
    # rows to nothing, hence the errors' own, per-panel treatment above.
    return {'mean': _round(pct(['ombg_mean', 'oman_mean'], 98)) or 1.0,
            'rms': _round(pct(['ombg_rms', 'oman_rms'], 98)) or 1.0}


def _norm(kind, lim, field):
    if kind == 'count':
        top = np.nanmax(field) if np.any(np.isfinite(field)) else 1.0
        return LogNorm(vmin=1, vmax=max(top, 2)), 'YlGnBu'
    if kind == 'mean':
        return plt.Normalize(-lim['mean'], lim['mean']), P.DIVERGING
    if kind == 'rms':
        return plt.Normalize(0, lim['rms']), P.SEQUENTIAL
    if kind in PER_PANEL:
        lo, hi = span(field)
        return plt.Normalize(lo, hi), P.SEQ_TEAL
    return TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=2.0), P.DIVERGING


def _blank(ax, name, polar=False):
    """A row for an experiment with nothing at this date: blank, labelled."""
    ax.set_facecolor(P.SURFACE)
    ax.text(0.5, 0.5, '%s: no data' % name, ha='center', va='center',
            fontsize=9, color=P.MUTED, transform=ax.transAxes)
    for s in ax.spines.values():
        s.set_edgecolor(P.GRID)
    ax.set_xticks([])
    ax.set_yticks([])


def _cbar(fig, ax, handle, kind, deg, per, small=False):
    """A colourbar in its own axes just right of ``ax`` (the row's last
    panel): fig.colorbar(ax=...) steals space from cartopy axes unevenly.
    ``small`` is the per-panel form, tucked inside the panel's gutter."""
    pos = ax.get_position()
    if small:
        cax = fig.add_axes([pos.x1 + 0.004, pos.y0 + 0.1 * pos.height, 0.007,
                            0.8 * pos.height])
        cb = fig.colorbar(handle, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.ax.yaxis.set_offset_position('left')
        return
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0 + 0.05 * pos.height, 0.012,
                        0.9 * pos.height])
    cb = fig.colorbar(handle, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    cb.set_label({'count': 'obs per %s' % per, 'mean': 'obs units',
                  'rms': 'obs units', 'ratio': 'ratio'}[kind], fontsize=7.5)


def _stat_grid(obstype, per_exp, names, cfg, lim, title, deg, kind_of_panel,
               fname, polar_ok=True):
    """Rows: the statistics in COLUMNS; columns: experiments.

    Transposed from the obvious layout on purpose: two or three experiments
    side by side fit the report's page width, while seven statistics side
    by side made each map a thumbnail. Each row has its own colour scale,
    shared across the experiments so they compare directly.
    """
    hemi = _hemi(obstype) if polar_ok else None
    polar = hemi is not None
    if polar:
        _label, proj, extent, _cut = PS.HEMIS[hemi]
        w, h = 3.0, 3.0
    elif polar_ok:
        proj, extent = ccrs.PlateCarree(central_longitude=PS.DATA_LON0), None
        w, h = 4.2, 2.3
    else:
        proj, extent = None, None
        w, h = 4.2, 2.0
    nrow = len(COLUMNS)
    fig, axes = plt.subplots(nrow, len(names),
                             figsize=(w * len(names) + 1.0, h * nrow + 0.6),
                             squeeze=False,
                             subplot_kw=(dict(projection=proj) if proj else {}))
    fig.subplots_adjust(left=0.07, right=0.9, top=0.96, bottom=0.03,
                        wspace=0.12, hspace=0.1)
    for r, (key, rtitle, kind) in enumerate(COLUMNS):
        handle = None
        for c, name in enumerate(names):
            ax = axes[r][c]
            if r == 0:
                ax.set_title(name, fontsize=10, color=P.INK)
            d = per_exp.get(name)
            if d is None:
                _blank(ax, name)
            else:
                norm, cmap = _norm(kind, lim, d[key])
                h_ = kind_of_panel(ax, d[key], norm, cmap, proj, extent,
                                   polar)
                if kind in PER_PANEL and h_ is not None:
                    _cbar(fig, ax, h_, kind, deg, '', small=True)
                else:
                    handle = h_ or handle
            if c == 0:
                (ax.text(-0.04, 0.5, rtitle, transform=ax.transAxes,
                         rotation=90, ha='right', va='center', fontsize=9,
                         color=P.INK2) if proj else
                 ax.set_ylabel(rtitle, fontsize=9, color=P.INK2))
        if handle is not None:
            _cbar(fig, axes[r][-1], handle, kind, deg,
                  '%g$^\\circ$ bin' % deg if proj else 'depth bin x %g$^\\circ$'
                  % deg)
    fig.suptitle(title, fontsize=11.5, color=P.INK, y=0.995)
    return _save(fig, cfg, fname)


def fig_maps(obstype, per_exp, names, cfg, lim, title, deg):
    lon_e, lat_e = B.lon_edges(deg), B.lat_edges(deg)

    def panel(ax, field, norm, cmap, proj, extent, polar):
        handle = ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(field),
                               norm=norm, cmap=cmap,
                               transform=ccrs.PlateCarree(), rasterized=True,
                               shading='flat', zorder=1)
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            PS._circular_boundary(ax)
        else:
            ax.set_global()
        PS._decorate(ax, polar=polar)
        return handle
    return _stat_grid(obstype, per_exp, names, cfg, lim, title, deg, panel,
                      'obsbins_map_%s.png' % P.slug(obstype))


def fig_sections(obstype, per_exp, names, cfg, lim, title, deg):
    edges = np.asarray(cfg.get('depth_bins') or [0, 6000], dtype='f8')
    lat_e = B.lat_edges(deg)
    y = np.arange(len(edges))

    def panel(ax, field, norm, cmap, proj, extent, polar):
        handle = ax.pcolormesh(lat_e, y, np.ma.masked_invalid(field),
                               norm=norm, cmap=cmap, rasterized=True,
                               shading='flat')
        ax.set_ylim(len(edges) - 1, 0)
        ax.set_yticks(y[:-1] + 0.5)
        ax.set_yticklabels(['%g-%g' % (lo, hi) for lo, hi
                            in zip(edges[:-1], edges[1:])], fontsize=6.5)
        ax.set_xlim(-80, 90)
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.set_xticklabels(['60S', '30S', '0', '30N', '60N'], fontsize=7)
        ax.tick_params(length=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(P.GRID)
        return handle
    return _stat_grid(obstype, per_exp, names, cfg, lim, title, deg, panel,
                      'obsbins_sec_%s.png' % P.slug(obstype), polar_ok=False)


def fig_regression(obstype, per_exp, names, cfg, fits, title):
    fig, axes = plt.subplots(len(names), 2, figsize=(8.2, 3.7 * len(names)),
                             squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.86, top=0.9, bottom=0.07,
                        hspace=0.4, wspace=0.3)
    col = P.color_map(names)
    handle = None
    for r, name in enumerate(names):
        a = per_exp.get(name)
        for c, (side, label) in enumerate((('bkg', 'background'),
                                           ('ana', 'analysis'))):
            ax = axes[r][c]
            if a is None or ('reg/%s' % side) not in a:
                _blank(ax, name)
                continue
            edges = a['reg/edges']
            hist = a['reg/%s' % side].astype('f8')
            hist[hist == 0] = np.nan
            handle = ax.pcolormesh(edges, edges, np.ma.masked_invalid(hist).T,
                                   norm=LogNorm(vmin=1, vmax=max(
                                       2, np.nanmax(hist))),
                                   cmap='YlGnBu', rasterized=True,
                                   shading='flat')
            lo, hi = edges[0], edges[-1]
            ax.plot([lo, hi], [lo, hi], color=P.MUTED, lw=0.9, ls=(0, (4, 3)),
                    zorder=3)
            f = (fits.get(name) or {}).get(side) or {}
            f = {k: (np.nan if v is None else v) for k, v in f.items()}
            if np.isfinite(f.get('slope', np.nan)):
                xs = np.array([lo, hi])
                ax.plot(xs, f['slope'] * xs + f['intercept'], color=col[name],
                        lw=1.6, zorder=4)
                ax.text(0.03, 0.97,
                        'slope %.3f\nintercept %+.3f\nr %.4f\nN %s'
                        % (f['slope'], f['intercept'], f['r'],
                           '{:,}'.format(int(f['n']))),
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=8, color=P.INK,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec='none', alpha=0.85))
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
            ax.set_title('%s: obs vs %s' % (name, label), fontsize=9.5,
                         color=P.INK)
            ax.set_xlabel('observation', fontsize=8.5)
            ax.set_ylabel('model (H(x))', fontsize=8.5)
            ax.tick_params(labelsize=7.5)
            P.tidy(ax, xgrid=True)
    if handle is not None:
        cb = fig.colorbar(handle, cax=fig.add_axes([0.89, 0.3, 0.018, 0.4]))
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
        cb.set_label('pairs per cell', fontsize=8)
    fig.suptitle(title, fontsize=11.5, color=P.INK, y=0.975)
    return _save(fig, cfg, 'obsbins_reg_%s.png' % P.slug(obstype))


# --------------------------------------------------------------------------

def per_type(arrays):
    """Split one npz (or a pooled total) into {type: {exp: {rest: array}}}."""
    out = {}
    for k, v in arrays.items():
        if not k.startswith('obsbins/'):
            continue
        _o, obstype, exp, rest = k.split('/', 3)
        out.setdefault(obstype, {}).setdefault(exp, {})[rest] = v
    return out


def _fits(cycles, obstype, name, which):
    """Regression fits: one cycle's from its JSON, or refitted from the
    pooled histogram when ``which`` is 'all'."""
    if which != 'all':
        return (cycles[which].get('obs', {}).get(obstype, {})
                .get('regression', {}).get(name) or {})
    return {}


def _fit_from_hist(hist, edges):
    """Least-squares line and r from a 2-D histogram (bin centres)."""
    c = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype('f8')
    n = w.sum()
    if n < 3:
        return {'slope': np.nan, 'intercept': np.nan, 'r': np.nan, 'n': int(n)}
    X, Y = np.meshgrid(c, c, indexing='ij')
    xm, ym = (w * X).sum() / n, (w * Y).sum() / n
    sxx = (w * (X - xm) ** 2).sum()
    syy = (w * (Y - ym) ** 2).sum()
    sxy = (w * (X - xm) * (Y - ym)).sum()
    slope = sxy / sxx if sxx > 0 else np.nan
    return {'slope': float(slope), 'intercept': float(ym - slope * xm),
            'r': float(sxy / np.sqrt(sxx * syy)) if sxx > 0 and syy > 0
            else np.nan, 'n': int(n)}


def draw(which, split, names, cycles, cfg, lims, deg):
    """All figures for one date (or 'all') from the split arrays."""
    global TAG
    TAG = '_%s' % which
    when = ('every cached cycle pooled' if which == 'all'
            else PS.cycle_row_label(which))
    written = []
    for obstype in sorted(split):
        exps = split[obstype]
        derived = {n: B.derived(exps[n], 'map') for n in names if n in exps}
        derived = {n: d for n, d in derived.items() if d is not None}
        lim = lims.get(obstype) or {'mean': 1.0, 'rms': 1.0}
        short = P.short(obstype)
        if derived:
            written.append(fig_maps(
                obstype, derived, names, cfg, lim,
                '%s: binned departures, common sample - %s' % (short, when),
                deg))
        if is_profile(obstype):
            secs = {n: B.derived(exps[n], 'sec') for n in names if n in exps}
            secs = {n: d for n, d in secs.items() if d is not None}
            if secs:
                written.append(fig_sections(
                    obstype, secs, names, cfg, lim,
                    '%s: departures by depth and latitude, common sample - %s'
                    % (short, when), deg))
        fits = {}
        for n in names:
            if n not in exps or 'reg/bkg' not in exps[n]:
                continue
            f = _fits(cycles, obstype, n, which)
            if not f:
                f = {side: _fit_from_hist(exps[n]['reg/%s' % side],
                                          exps[n]['reg/edges'])
                     for side in ('bkg', 'ana')}
            fits[n] = f
        if fits:
            written.append(fig_regression(
                obstype, exps, names, cfg, fits,
                '%s: observation against model, common sample - %s'
                % (short, when)))
    TAG = ''
    return written


_SHARED = {}


def _date_worker(c):
    """One date's figures, in a (forked) worker; returns (paths, seconds)."""
    t = time.time()
    sh = _SHARED
    arrays = P.load_obsbins(sh['cfg'], c)
    written = draw(c, per_type(dict(arrays)), sh['names'], sh['cycles'],
                   sh['cfg'], sh['lims'], sh['deg'])
    return written, time.time() - t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config', nargs='?', default=None)
    ap.add_argument('--config', dest='config_opt', default=None)
    ap.add_argument('--root', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--cache', action='append', default=None)
    ap.add_argument('--hours', default=None,
                    help='per-date figures for cycles at these UTC hours '
                         '(plus the latest); "all" for every cycle; default '
                         'the latest only. The pooled figures are always drawn')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--jobs', type=int, default=1,
                    help='dates to draw in parallel')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt
    t_all = time.time()
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    cycles = P.load_cycles(cfg)
    order = sorted(cycles)
    names = P.all_exp_names(cfg, cycles)
    deg = B.bin_deg(cfg)
    os.makedirs(cfg['figs'], exist_ok=True)
    fresh = Fresh(cfg, 'obsbins', force=a.force, script=__file__)

    # pass 1: pool every cycle (each npz loaded once), and set the scales
    inputs = []
    total = {}
    have = []
    for c in order:
        arrays = P.load_obsbins(cfg, c)
        if arrays is None:
            continue
        have.append(c)
        inputs += [os.path.join(root, '%s_obsbins.npz' % c)
                   for root in cfg['caches']
                   if os.path.exists(os.path.join(root, '%s_obsbins.npz' % c))]
        B.accumulate(total, dict(arrays))
    if not have:
        print('no binned departures cached -- run compute_cycle.py (or its '
              '--rejoin) first')
        return 0
    split = per_type(total)
    lims = {}
    for obstype, exps in split.items():
        ds = [B.derived(v, 'map') for v in exps.values()]
        lims[obstype] = scales([d for d in ds if d])
    params = {'lims': lims, 'deg': deg, 'names': names}
    print('binned departures: %d obs type(s), %d cycle(s) pooled'
          % (len(split), len(have)), flush=True)

    todo = [have[-1]]
    if a.hours and a.hours.lower() == 'all':
        todo = have
    elif a.hours:
        hours = {h.strip().zfill(2) for h in a.hours.split(',') if h.strip()}
        todo = [c for c in have if c[8:10] in hours or c == have[-1]]

    t = time.time()
    if fresh.ok('all', inputs, params):
        print('  pooled: up to date, skipped', flush=True)
    else:
        written = draw('all', split, names, cycles, cfg, lims, deg)
        fresh.record('all', inputs, params, written)
        print('  pooled: %d figure(s) in %.1fs'
              % (len(written), time.time() - t), flush=True)
    del total, split
    pending = []
    for c in todo:
        cin = [os.path.join(root, '%s_obsbins.npz' % c) for root in cfg['caches']
               if os.path.exists(os.path.join(root, '%s_obsbins.npz' % c))]
        if fresh.ok('cycle:%s' % c, cin, params):
            print('  %s: up to date, skipped' % c, flush=True)
        else:
            pending.append((c, cin))
    _SHARED.update(cfg=cfg, cycles=cycles, names=names, lims=lims, deg=deg)
    jobs = max(1, min(a.jobs, len(pending)))
    if jobs == 1:
        results = map(_date_worker, [c for c, _cin in pending])
    else:
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(max_workers=jobs)
        results = ex.map(_date_worker, [c for c, _cin in pending])
    for (c, cin), (written, secs) in zip(pending, results):
        fresh.record('cycle:%s' % c, cin, params, written)
        print('  %s: %d figure(s) in %.1fs' % (c, len(written), secs),
              flush=True)
    if jobs > 1:
        ex.shutdown()
    fresh.save()
    print('done in %.1fs (%d unit(s) up to date and skipped)'
          % (time.time() - t_all, fresh.skipped), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
