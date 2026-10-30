#!/usr/bin/env python3
"""T-S (water-mass) diagrams. Reads '<cycle>_tsdiag.npz'.

Per profile platform (Argo, ...), region and depth range, two figures, drawn
once for every cached cycle pooled ('all') and once per calendar month when
the cache spans more than one:

    figs/ts_density_<platform>_<region>_<depth>_<period>.png
        the observed theta-S distribution as a log-density histogram, then
        per experiment the background (top row) and the analysis (bottom
        row) with the observations behind them: the observed 50 and 90 %
        contours in black, the model's in the experiment's colour. sigma0 isopycnals
        throughout. A water mass the model puts at the wrong temperature
        or salinity shows as a contour that does not sit on the black one.
    figs/ts_bias_<platform>_<region>_<depth>_<period>.png
        model minus observation, S and theta, averaged in sigma0 classes of
        the OBSERVED water -- classes pooled until each holds enough levels,
        so the resolution follows the data: background solid with a +/-1 sd
        band, analysis dashed, counts beside. The compact view for
        comparing experiments.

Depth ranges are `ts_diagram: {depth_ranges: [[top, bottom], ...]}` (default
0-6000, 0-500 and 500-2000 m); regions are 'global' plus the configured
basins and boxes. Every cycle's points are reduced to fixed-grid sums as it
is read, so memory does not grow with the number of cycles.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import lv_plot as P  # noqa: E402
import lv_tsdiag as TS  # noqa: E402
from lv_common import Fresh, load_config, region_list  # noqa: E402

DPI = 100

# Fewest paired levels a figure is drawn from; below it a region/depth is
# noise and left out (the report then simply has no panel for it).
MIN_POINTS = 200
# Fewest levels a (merged) sigma0 class holds in the bias figure, and the
# most classes it is split into: each class holds at least the larger of
# MIN_CLASS and total / MAX_CLASSES (see lv_tsdiag.merge_classes).
MIN_CLASS = 200
MAX_CLASSES = 40
# Fractions of the points the density contours enclose.
CONTOUR_FRACTIONS = (0.9, 0.5)
# Contour cells across the plotted window, each way (see _coarse).
CONTOUR_CELLS = 60


def tsdiag_path(cfg, cycle):
    """The cycle's T-S pairs from the first cache that has them -- the page
    cache first, so a rejoined common sample beats an experiment's own.
    Not merged key by key as the bins are: the observation and model arrays
    of one file are aligned row for row, and another file's are not."""
    for root in cfg.get('caches', [cfg['cache']]):
        p = os.path.join(root, '%s_tsdiag.npz' % cycle)
        if os.path.exists(p):
            return p
    return None


def _load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _save(fig, cfg, base):
    return P.save(fig, cfg, base, dpi=DPI, quiet=True)


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _smooth(h):
    """3x3 box sum, so contours of a sparse histogram are not single cells."""
    p = np.pad(h.astype('f8'), 1)
    return sum(p[i:i + h.shape[0], j:j + h.shape[1]]
               for i in range(3) for j in range(3))


def _factor(lim, edges, cells=CONTOUR_CELLS):
    """Block size that puts about ``cells`` contour cells across ``lim``."""
    return max(1, int(round((lim[1] - lim[0]) / np.diff(edges[:2])[0]
                            / cells)))


def _coarse(h, ft, fs):
    """Block sums of ``h`` (ft x fs cells), then smoothed: the contouring
    grid. The fixed histogram grid is fine enough for a 2 degC window; over
    a full column's 30 degC the same cells leave the 90 % contour as specks
    around single cells, so the contours are drawn on blocks sized to the
    plotted window instead."""
    nt, ns = h.shape[0] // ft * ft, h.shape[1] // fs * fs
    b = h[:nt, :ns].reshape(nt // ft, ft, ns // fs, fs).sum(axis=(1, 3))
    return _smooth(b)


def _coarse_centres(edges, f):
    e = edges[:(edges.size - 1) // f * f + 1:f]
    return 0.5 * (e[:-1] + e[1:])


def contour_levels(h, fractions=CONTOUR_FRACTIONS):
    """Density thresholds enclosing each fraction of the points, increasing
    (for contour()); None where the histogram is too thin to contour."""
    v = np.sort(h.ravel())[::-1]
    tot = v.sum()
    if tot <= 0:
        return None
    cum = np.cumsum(v) / tot
    lv = sorted({float(v[min(np.searchsorted(cum, f), v.size - 1)])
                 for f in fractions})
    lv = [x for x in lv if x > 0]
    return lv or None


def _limits(h, edges, axis, lo=0.005, hi=0.995, pad=0.05):
    """Axis range holding the central lo..hi of the observed distribution."""
    m = h.sum(axis=axis).astype('f8')
    c = np.cumsum(m) / m.sum()
    i0 = int(np.searchsorted(c, lo))
    i1 = int(min(np.searchsorted(c, hi) + 1, edges.size - 1))
    a, b = edges[i0], edges[i1]
    w = max(b - a, 1e-3)
    return a - pad * w, b + pad * w


def _isopycnals(ax, slim, tlim):
    s = np.linspace(slim[0], slim[1], 120)
    t = np.linspace(tlim[0], tlim[1], 120)
    S, T = np.meshgrid(s, t)
    sig = TS.sigma0(T, S)
    cs = ax.contour(S, T, sig, levels=8, colors=P.MUTED, linewidths=0.6,
                    zorder=1)
    ax.clabel(cs, fontsize=7, fmt='%.1f')


def fig_density(acc, names, col, title):
    obs = acc['obs/hist']
    have = [n for n in names if '%s/hist_b' % n in acc]
    tlim = _limits(obs, TS.THETA_EDGES, 1)
    slim = _limits(obs, TS.SALT_EDGES, 0)
    ft, fs = _factor(tlim, TS.THETA_EDGES), _factor(slim, TS.SALT_EDGES)
    sc = _coarse_centres(TS.SALT_EDGES, fs)
    tc = _coarse_centres(TS.THETA_EDGES, ft)
    # The observations on the left, spanning both rows; then one column per
    # experiment, background on the top row and analysis below. Drawn in
    # one panel the two lie almost on each other and the analysis vanishes.
    ncol = 1 + len(have)
    fig = plt.figure(figsize=(3.5 * ncol, 6.9))
    gs = fig.add_gridspec(2, ncol)
    ax0 = fig.add_subplot(gs[:, 0])
    grid = [[fig.add_subplot(gs[r, c + 1], sharex=ax0, sharey=ax0)
             for c in range(len(have))] for r in range(2)]
    dens = np.ma.masked_equal(obs, 0)
    norm = LogNorm(vmin=1, vmax=max(2, int(obs.max())))
    obs_s = _coarse(obs, ft, fs)
    obs_lv = contour_levels(obs_s)

    pm = ax0.pcolormesh(TS.SALT_EDGES, TS.THETA_EDGES, dens,
                        cmap=P.SEQUENTIAL, norm=norm, zorder=0)
    fig.colorbar(pm, ax=ax0, shrink=0.6, pad=0.02, label='levels per cell')
    ax0.set_title('observations (n = %d)' % int(obs.sum()), fontsize=9.5)
    ax0.set_ylabel('potential temperature ($^\\circ$C)')
    panels = [ax0]
    for r, (side, word) in enumerate((('b', 'background'),
                                      ('a', 'analysis'))):
        for c, n in enumerate(have):
            ax = grid[r][c]
            ax.pcolormesh(TS.SALT_EDGES, TS.THETA_EDGES, dens, cmap='Greys',
                          norm=norm, alpha=0.45, zorder=0)
            if obs_lv:
                ax.contour(sc, tc, obs_s, levels=obs_lv, colors=P.INK,
                           linewidths=1.1, zorder=3)
            h = _coarse(acc['%s/hist_%s' % (n, side)], ft, fs)
            lv = contour_levels(h)
            if lv:
                ax.contour(sc, tc, h, levels=lv, colors=col[n],
                           linewidths=1.5, zorder=4)
            ax.set_title('%s: %s' % (n, word), fontsize=9, color=col[n])
            if c == 0:
                ax.set_ylabel('$\\theta$ ($^\\circ$C)')
            else:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.tick_params(labelbottom=False)
            panels.append(ax)
    for ax in panels:
        _isopycnals(ax, slim, tlim)
        P.tidy(ax, xgrid=True)
    ax0.set_xlim(slim)
    ax0.set_ylim(tlim)
    ax0.set_xlabel('salinity')
    for ax in grid[1]:
        ax.set_xlabel('salinity')
    fig.suptitle(title, fontsize=11, color=P.INK)
    if have:
        handles = [Line2D([], [], color=P.INK, lw=1.1, label='observed')] + [
            Line2D([], [], color=col[n], lw=1.5, label=n) for n in have]
        fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                   fontsize=8, frameon=False,
                   title='contours enclose %s and %s %% of levels'
                   % tuple(int(100 * f) for f in CONTOUR_FRACTIONS),
                   title_fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return fig


def fig_bias(acc, names, col, title):
    total = int(acc['n'].sum())
    groups = TS.merge_classes(acc['n'], max(MIN_CLASS, total // MAX_CLASSES))
    if len(groups) < 2:
        return None
    have = [x for x in names if '%s/dtheta_b' % x in acc]
    lo, hi = TS.class_edges(groups)
    # each merged class drawn at its count-weighted sigma0, not its middle:
    # a wide class at the light end is mostly its densest water
    fine = 0.5 * (TS.SIGMA_EDGES[:-1] + TS.SIGMA_EDGES[1:])
    centres = np.array([np.average(fine[a:b], weights=acc['n'][a:b])
                        for a, b in groups])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1, 0.45]})
    for ax, var, label in ((axes[0], 'salt', 'S, model $-$ observed'),
                           (axes[1], 'theta',
                            '$\\theta$, model $-$ observed ($^\\circ$C)')):
        for x in have:
            _n, m, sd = TS.class_stats(acc, x, var, 'b', groups)
            ax.fill_betweenx(centres, m - sd, m + sd, color=col[x],
                             alpha=0.10, linewidth=0)
            ax.plot(m, centres, 'o-', color=col[x], lw=1.6, ms=3,
                    label='%s background' % x)
            _n, ma, _sa = TS.class_stats(acc, x, var, 'a', groups)
            ax.plot(ma, centres, color=col[x], lw=1.0, ls='dashed',
                    label='%s analysis' % x)
        ax.axvline(0.0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)))
        ax.set_xlabel(label)
        P.tidy(ax, xgrid=True)
    n = np.array([acc['n'][a:b].sum() for a, b in groups])
    axes[2].barh(0.5 * (lo + hi), n, height=hi - lo, color=P.MUTED,
                 alpha=0.6, edgecolor=P.SURFACE, linewidth=0.6)
    axes[2].set_xscale('log')
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xlabel('levels per class')
    P.tidy(axes[2], xgrid=True)
    axes[0].set_ylim(hi[-1], lo[0])
    axes[0].set_ylabel('$\\sigma_0$ of the observed water (kg m$^{-3}$)')
    fig.suptitle(title, fontsize=11, color=P.INK)
    # one legend column per experiment, background above analysis
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', fontsize=8,
                   ncol=max(1, len(have)), frameon=False)
    fig.tight_layout(rect=(0, 0.11 if handles else 0, 1, 0.97))
    return fig


def _label(plat, region, dslug, period):
    if period == 'all':
        when = 'all cycles'
    else:
        import calendar
        when = '%s %s' % (calendar.month_abbr[int(period[4:6])], period[:4])
    return '%s, %s, %s, %s' % (plat, region.replace('_', ' '),
                               dslug[1:].replace('-', '$-$').replace('m', ' m'),
                               when)


def draw(total, period, names, cfg):
    """Every figure for one period's pooled sums; returns the paths."""
    col = P.color_map(names)
    written = []
    for (plat, region, dslug), acc in sorted(total.items()):
        if acc['obs/hist'].sum() < MIN_POINTS:
            continue
        lbl = _label(plat, region, dslug, period)
        fig = fig_density(acc, names, col,
                          '$\\theta$-S, common sample: %s' % lbl)
        written.append(_save(fig, cfg, TS.fig_name(
            'density', plat, P.slug(region), dslug, period)))
        fig = fig_bias(acc, names, col,
                       'water-mass bias by density class: %s' % lbl)
        if fig is not None:
            written.append(_save(fig, cfg, TS.fig_name(
                'bias', plat, P.slug(region), dslug, period)))
    return written


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config', nargs='?', default=None)
    ap.add_argument('--config', dest='config_opt', default=None)
    ap.add_argument('--root', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--cache', action='append', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt
    t_all = time.time()
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    cycles = P.load_cycles(cfg)
    names = P.all_exp_names(cfg, cycles)
    os.makedirs(cfg['figs'], exist_ok=True)
    fresh = Fresh(cfg, 'tsdiag', force=a.force, script=__file__)

    paths = {c: tsdiag_path(cfg, c) for c in sorted(cycles)}
    paths = {c: p for c, p in paths.items() if p}
    if not paths:
        print('no T-S pairs cached -- run compute_cycle.py (or its --rejoin) '
              'first')
        return 0
    ranges = TS.depth_ranges(cfg)
    by_month = {}
    for c in paths:
        by_month.setdefault(c[:6], []).append(c)
    months = sorted(by_month)
    periods = {'all': sorted(paths)}
    if len(months) > 1:
        periods.update(by_month)
    params = {'names': names, 'ranges': ranges,
              'regions': ['global'] + region_list(cfg),
              'min_points': MIN_POINTS, 'min_class': MIN_CLASS,
              'max_classes': MAX_CLASSES}
    stale = {p: not fresh.ok(p, [paths[c] for c in cs], params)
             for p, cs in periods.items()}
    print('T-S diagrams: %d cycle(s), %d period(s), %d to draw'
          % (len(paths), len(periods), sum(stale.values())), flush=True)

    # one pass, month by month: each cycle is read once and added to its
    # month and (when that is being redrawn) to the whole-period total
    whole = {}
    for m in months:
        if not (stale.get(m) or stale['all']):
            continue
        t = time.time()
        month = {}
        for c in by_month[m]:
            part = TS.reduce_cycle(_load(paths[c]), cfg, ranges)
            if stale['all']:
                TS.accumulate(whole, part)
            if stale.get(m):
                TS.accumulate(month, part)
        if stale.get(m):
            written = draw(month, m, names, cfg)
            fresh.record(m, [paths[c] for c in by_month[m]], params, written)
            print('  %s: %d figure(s) in %.1fs'
                  % (m, len(written), time.time() - t), flush=True)
    if stale['all']:
        t = time.time()
        written = draw(whole, 'all', names, cfg)
        fresh.record('all', [paths[c] for c in periods['all']], params,
                     written)
        print('  all cycles: %d figure(s) in %.1fs'
              % (len(written), time.time() - t), flush=True)
    fresh.save()
    print('done in %.1fs (%d period(s) up to date and skipped)'
          % (time.time() - t_all, fresh.skipped), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
