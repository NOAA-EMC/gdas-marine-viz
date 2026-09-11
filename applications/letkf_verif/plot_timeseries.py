#!/usr/bin/env python3
"""Cycling figures: how the headline metrics evolve, and whether they drift.

With a single cycle these degrade to a dot plot; they become informative once
several cycles are cached. Drift is the thing a single cycle cannot show and
the thing that most often decides whether a LETKF configuration is usable.
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
import lv_plot as P  # noqa: E402
import lv_verif as LV  # noqa: E402
from lv_common import Grid, load_config, region_list  # noqa: E402

# (cache key, panel title, reference line or None). Each mean sits next to
# the RMS it belongs with: the pair separates a run that is merely scattered
# from one that is systematically offset, and only the second is a bias the
# system can be asked to correct.
PANELS = [
    ('ombg_rms', 'RMS(O$-$B)', None),
    ('ombg_mean', 'mean(O$-$B), bias', 0.0),
    ('oman_rms', 'RMS(O$-$A)', None),
    ('oman_mean', 'mean(O$-$A), bias', 0.0),
    ('spread_b', 'prior ensemble spread', None),
    ('consistency_ratio', 'consistency ratio', 1.0),
    ('spread_skill', 'spread / skill', 1.0),
    ('desroziers_R_ratio', 'Desroziers $R$ / assigned $R$', 1.0),
]

# Columns in the fig_timeseries grid. The row count follows from len(PANELS),
# so adding a metric above does not also mean editing the subplot call.
PANEL_NCOL = 4


def _finite(t, v):
    """The cycles where the series has a value, and those values.

    Observation types that arrive daily rather than every cycle leave NaN at
    the cycles in between, and matplotlib breaks a line at every NaN -- so a
    series present at one cycle in four has every point isolated between gaps
    and draws as bare dots with nothing joining them. Dropping the gaps instead
    connects consecutive AVAILABLE cycles, which is what the line is for.

    The segment then spans the cycles that have no data, so a long gap shows as
    one long straight leg; the markers are what say where the data actually is.
    """
    v = np.asarray(v, dtype='f8')
    ok = np.isfinite(v)
    return [t[i] for i in np.flatnonzero(ok)], v[ok]


def _times(cycles):
    return [dt.datetime.strptime(c, '%Y%m%d%H') for c in sorted(cycles)]


def _series(cycles, obstype, name, key, sample='common'):
    return np.array([P.get(cycles[c].get('obs', {}).get(obstype, {}),
                           sample, name, 'all', key)
                     for c in sorted(cycles)], dtype='f8')


def fig_timeseries(cycles, cfg, obstype, sample='common'):
    names = P.all_exp_names(cfg, cycles)
    col = P.color_map(names)
    t = _times(cycles)
    single = len(t) == 1
    nc = PANEL_NCOL
    nr = int(np.ceil(len(PANELS) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.5 * nc, 3.2 * nr),
                             squeeze=False)
    for a in axes.ravel()[len(PANELS):]:
        a.set_visible(False)
    drawn = False
    for i, (key, label, target) in enumerate(PANELS):
        ax = axes[i // nc][i % nc]
        for n in names:
            v = _series(cycles, obstype, n, key, sample)
            if not np.any(np.isfinite(v)):
                continue
            tt, vv = _finite(t, v)
            ax.plot(tt, vv, 'o' if single else 'o-', color=col[n], label=n,
                    ms=7 if single else 5, mec=P.SURFACE, mew=1.0)
            drawn = True
        if target is not None:
            ax.axhline(target, color=P.MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.set_title(label, fontsize=10)
        P.tidy(ax)
        if len(t) <= 1:
            ax.set_xticks(t)
            ax.set_xticklabels([t[0].strftime('%Y-%m-%d %HZ')], fontsize=8)
    if not drawn:
        plt.close(fig)
        return None
    if not single:
        _date_labels(fig, axes, t)
    P.maybe_legend(axes[0][0], fontsize=8.5)
    note = ('  (single cycle - add more cached cycles to see evolution)'
            if single else '')
    fig.suptitle('%s: metric evolution across cycles%s'
                 % (P.short(obstype), note), y=1.01, fontsize=11.5,
                 color=P.INK)
    fig.tight_layout()
    return P.save(fig, cfg, 'cycle_%s.png' % obstype)


def fig_obs_counts(cycles, cfg):
    """Assimilated observation counts against cycle, one figure per obs type.

    One PNG per obs type rather than one big grid of tiny panels: with 20+
    obs types configured that grid squeezed every type into a sliver too
    narrow to read -- the same problem fig_obs_fit had. build_report.py
    embeds all of them behind an obs-type dropdown; it derives the same
    type list from the cache independently (every type present qualifies
    here, so there is no presence filter to keep in sync -- just the list
    itself).

    Thinning and QC both move this number, and a count that steps or
    collapses mid-run is usually the first sign that something upstream
    broke. The common sample is drawn alongside so the size of the
    comparable subset is visible at the same time.
    """
    order = sorted(cycles)
    names = P.all_exp_names(cfg, cycles)
    col = P.color_map(names)
    types = sorted({t for d in cycles.values() for t in d.get('obs', {})})
    if not types:
        return []
    t = _times(cycles)
    single = len(t) == 1
    mk = 'o' if single else 'o-'

    written = []
    for ot in types:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        top = 0.0
        drawn = False
        # The common sample goes down first: it often sits exactly on top of
        # the most-thinned experiment, and the experiment must stay visible.
        common = np.array([P.get(cycles[c], 'obs', ot, 'counts', names[0],
                                 'n_common_pass') for c in order], dtype='f8')
        if np.any(np.isfinite(common)):
            tt, cc = _finite(t, common)
            ax.plot(tt, cc, mk, color=P.MUTED, label='common sample',
                    ms=7 if single else 5, lw=1.2, mec=P.SURFACE, mew=0.8,
                    zorder=1)
            top = max(top, np.nanmax(common))
            drawn = True
        for n in names:
            v = np.array([P.get(cycles[c], 'obs', ot, 'counts', n,
                                'n_pass_own') for c in order], dtype='f8')
            if not np.any(np.isfinite(v)):
                continue
            tt, vv = _finite(t, v)
            ax.plot(tt, vv, mk, color=col[n], label=n, ms=5 if single else 4,
                    mec=P.SURFACE, mew=1.0, zorder=3)
            top = max(top, np.nanmax(v))
            drawn = True
        if not drawn:
            plt.close(fig)
            continue
        ax.set_ylim(0, 1.1 * top if top > 0 else 1)
        ax.set_ylabel('observations passing QC')
        P.tidy(ax)
        if single:
            ax.set_xticks(t)
            ax.set_xticklabels([t[0].strftime('%Y-%m-%d %HZ')], fontsize=8)
        else:
            _date_labels(fig, np.array([[ax]]), t)
        P.maybe_legend(ax, fontsize=8)
        fig.suptitle('%s: observations assimilated per cycle%s'
                     % (P.short(ot),
                        '  (single cycle - add cycles to see evolution)'
                        if single else ''),
                     y=1.03, fontsize=11.5, color=P.INK)
        fig.tight_layout()
        written.append(P.save(fig, cfg, 'obscount_type_%s.png' % P.slug(ot)))
    return written


def _date_labels(fig, axes, t=None):
    """Finish the time axis on every visible panel.

    Rotates the labels, and pins every panel to the same span when given the
    full list of cycles. Both matter for a grid of panels:

    - Not fig.autofmt_xdate(): it hides the labels on every axis that is not in
      the last grid ROW, so when the final row is short the panels above the
      blanks silently lose their date axis.
    - Panels autoscale to the points they hold, so an observation type that
      arrives daily -- or stops early -- would get a narrower axis than its
      neighbours and could not be read across the row. A shared span also shows
      the absence honestly: the line simply stops.
    """
    for ax in np.ravel(axes):
        if not ax.get_visible():
            continue
        if t is not None and len(t) > 1:
            ax.set_xlim(t[0], t[-1])
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_horizontalalignment('right')


def _time_lines(ax, t, series, col, single):
    """One solid (background) and one dashed (analysis) line per experiment.

    Marker and linestyle are set explicitly rather than through a format
    string: a format string plus an `ls=` keyword fight each other, and with a
    single cycle the keyword wins and draws a line through one point.
    """
    drawn = False
    for n, (b, a) in series.items():
        for v, tag in ((b, 'background'), (a, 'analysis')):
            if v is None or not np.any(np.isfinite(v)):
                continue
            bkg = tag == 'background'
            # The analysis marker is hollow, so its EDGE must carry the colour.
            # Giving it the surface colour on both face and edge, as the
            # filled background marker does, makes it invisible.
            tt, vv = _finite(t, v)
            ax.plot(tt, vv, marker='o', ms=6 if single else 4,
                    color=col[n], mew=1.2,
                    mfc=col[n] if bkg else P.SURFACE,
                    mec=P.SURFACE if bkg else col[n],
                    ls='none' if single else ('-' if bkg else (0, (3, 2))),
                    lw=1.8 if bkg else 1.5,
                    alpha=1.0 if bkg else 0.85,
                    label='%s %s' % (n, tag))
            drawn = True
    return drawn


def fig_obs_fit(cycles, cfg):
    """Background and analysis fit in observation space, one figure per type.

    Two panels: RMS departure, and the signed mean beside it. Both are drawn
    from the same per-type sample (see P.type_sample), which the title names
    because it is not always the common one.

    One PNG per obs type rather than one big grid of tiny panels: with 20+
    obs types configured that grid squeezed every type into a sliver too
    narrow to read -- the same problem fig_verif_series had with regions.
    build_report.py embeds all of them behind an obs-type picker; it derives
    the same type list from the cache independently (every type present
    qualifies here, unlike the region list in fig_verif_series, so there is
    no presence filter to keep in sync -- just the list itself).
    """
    names = P.all_exp_names(cfg, cycles)
    col = P.color_map(names)
    types = sorted({t for d in cycles.values() for t in d.get('obs', {})})
    if not types:
        return []
    t = _times(cycles)
    single = len(t) == 1

    written = []
    for ot in types:
        # each obs type samples whoever actually shares it, not the run-wide
        # common/own choice -- one type missing an experiment shouldn't push
        # every OTHER, fully-shared type onto its own (unjoined) sample too
        sample = P.type_sample(cycles, ot)
        rms = {n: (_series(cycles, ot, n, 'ombg_rms', sample),
                   _series(cycles, ot, n, 'oman_rms', sample)) for n in names}
        # RMS cannot distinguish a run that is scattered from one that is
        # systematically offset, and only the second is a bias the system can
        # be asked to correct -- so the signed mean goes beside it, from the
        # same sample and the same cycles.
        bias = {n: (_series(cycles, ot, n, 'ombg_mean', sample),
                    _series(cycles, ot, n, 'oman_mean', sample)) for n in names}
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6), squeeze=False)
        ax_r, ax_b = axes[0]
        if not _time_lines(ax_r, t, rms, col, single):
            plt.close(fig)
            continue
        # A departure mean is signed, so zero is the whole reference: an
        # experiment straddling this line is unbiased at this scale.
        ax_b.axhline(0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        if not _time_lines(ax_b, t, bias, col, single):
            ax_b.set_visible(False)
        ax_r.set_ylabel('RMS departure (obs units)')
        ax_b.set_ylabel('mean departure (obs units)')
        ax_r.set_title('RMS departure', fontsize=10)
        ax_b.set_title('bias (mean departure)', fontsize=10)
        for ax in axes[0]:
            if not ax.get_visible():
                continue
            P.tidy(ax)
            if single:
                ax.set_xticks(t)
                ax.set_xticklabels([t[0].strftime('%Y-%m-%d %HZ')], fontsize=8)
        if not single:
            _date_labels(fig, axes, t)
        P.maybe_legend(ax_r, fontsize=7.5)
        # Name the sample on the figure itself. It is picked per obs type and
        # silently falls back to 'own' where no genuine cross-experiment join
        # exists, which changes what a comparison between runs MEANS -- the
        # section note says this can happen, but only the figure can say
        # whether it happened to this type.
        fig.suptitle('%s: fit to observations, solid O$-$B / dashed O$-$A'
                     '  [%s]%s'
                     % (P.short(ot),
                        'common sample' if sample == 'common'
                        else "each experiment's own sample",
                        '  (single cycle - add cycles to see evolution)'
                        if single else ''),
                     y=1.03, fontsize=11.5, color=P.INK)
        fig.tight_layout()
        written.append(P.save(fig, cfg, 'obsfit_type_%s.png' % P.slug(ot)))
    return written


def fig_verif_series(cycles, cfg):
    """Fit to the gridded products against cycle, one figure per region.

    One PNG per region rather than the old single wide grid (one row per
    product, one column per region): with several basins configured, that
    grid squeezed every region into a sliver too narrow to read. build_report
    .py embeds all of them behind a region picker and shows one at a time --
    it derives the same product/region lists from the cache independently,
    so both sides must apply the identical presence filter below.
    """
    order = sorted(cycles)
    names = P.all_exp_names(cfg, cycles)
    col = P.color_map(names)
    prods = [p for p in LV.PRODUCTS
             if any(P.get(d, 'state', n, 'ocean', 'verif', p, default=None)
                    for d in cycles.values() for n in names)]
    if not prods:
        return []
    regions = ['global'] + region_list(cfg)
    regions = [r for r in regions
               if any(P.get(d, 'state', n, 'ocean', 'verif', p, 'bkg', r,
                            default=None)
                      for d in cycles.values() for n in names for p in prods)]
    if not regions:
        return []
    t = _times(cycles)
    single = len(t) == 1

    def rms(n, p, st, reg):
        return np.array([P.get(cycles[c], 'state', n, 'ocean', 'verif', p, st,
                               reg, 'rms') for c in order], dtype='f8')

    written = []
    for reg in regions:
        fig, axes = plt.subplots(1, len(prods),
                                 figsize=(4.2 * len(prods), 3.4),
                                 squeeze=False)
        drawn = False
        for c, p in enumerate(prods):
            spec = LV.PRODUCTS[p]
            ax = axes[0][c]
            series = {n: (rms(n, p, 'bkg', reg), rms(n, p, 'ana', reg))
                      for n in names}
            if not _time_lines(ax, t, series, col, single):
                ax.set_visible(False)
                continue
            drawn = True
            ax.set_title(spec['label'], fontsize=10, color=P.INK)
            ax.set_ylabel('RMS (%s)' % spec['units'])
            P.tidy(ax)
            if single:
                ax.set_xticks(t)
                ax.set_xticklabels([t[0].strftime('%m-%d %HZ')], fontsize=7.5)
        if not drawn:
            plt.close(fig)
            continue
        P.maybe_legend(axes[0][0], fontsize=7.5)
        if not single:
            _date_labels(fig, axes, t)
        fig.suptitle('%s: fit to gridded analyses, solid background / '
                     'dashed analysis%s'
                     % (reg.replace('_', ' '),
                        '  (single cycle - add cycles to see evolution)'
                        if single else ''),
                     y=1.03, fontsize=11.5, color=P.INK)
        fig.tight_layout()
        written.append(P.save(fig, cfg, 'verif_region_%s.png' % P.slug(reg)))
    return written


def fig_drift(cycles, cfg):
    """RMS(O-B) for every obs type, normalised, to expose slow divergence."""
    if len(cycles) < 3:
        return None
    names = P.all_exp_names(cfg, cycles)
    col = P.color_map(names)
    types = sorted({t for d in cycles.values() for t in d.get('obs', {})})
    t = _times(cycles)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for n in names:
        stack = []
        for ot in types:
            v = _series(cycles, ot, n, 'ombg_rms', P.type_sample(cycles, ot))
            if np.isfinite(v[0]) and v[0] > 0:
                stack.append(v / v[0])
        if not stack:
            continue
        tt, vv = _finite(t, np.nanmean(stack, axis=0))
        ax.plot(tt, vv, 'o-', color=col[n], label=n,
                ms=5, mec=P.SURFACE, mew=1.0)
    ax.axhline(1.0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.set_ylabel('RMS(O$-$B) relative to first cycle')
    ax.set_title('Cycling stability - a rising line means the system is '
                 'degrading with time')
    P.tidy(ax)
    P.maybe_legend(ax)
    fig.autofmt_xdate(rotation=30)
    return P.save(fig, cfg, 'cycle_stability.png')


def _bkg_series(cycles, order, name, realm, var, level=0):
    """Area-weighted mean of a background field, per cycle."""
    out = []
    for c in order:
        p = P.get(cycles[c], 'state', name, realm, 'bkg_mean', var, default=None)
        out.append(p[level] if p and len(p) > level and p[level] is not None
                   else np.nan)
    return np.array(out, dtype='f8')


def _ice_series(cycles, order, name, key):
    return np.array([P.get(cycles[c], 'state', name, 'ice', 'ice_totals', key)
                     for c in order], dtype='f8')


def _incr_series(cycles, order, name, hemi):
    """Area-integrated sea-ice-area increment per cycle."""
    return np.array([P.get(cycles[c], 'state', name, 'ice',
                           'ice_area_increment', hemi) for c in order],
                    dtype='f8')


def fig_background_drift(cycles, cfg, grid):
    """Global background means against cycle -- the model-drift monitor.

    Departures and increments can both look healthy while the mean state walks
    away; this is the only view here that catches that.
    """
    order = sorted(cycles)
    names = P.all_exp_names(cfg, cycles)
    col = P.color_map(names)
    t = _times(cycles)
    single = len(t) == 1
    mk = 'o' if single else 'o-'

    panels = [
        ('mean SST (Temp level 0)', 'ocean', 'Temp', 0, None),
        ('mean SSS (Salt level 0)', 'ocean', 'Salt', 0, None),
        ('mean SSH (ave_ssh)', 'ocean', 'ave_ssh', 0, None),
        ('mean MLD', 'ocean', 'MLD', 0, None),
        ('mean Temp, level 30', 'ocean', 'Temp', 30, None),
        ('sea-ice extent (10$^6$ km$^2$)', 'ice', None, 0, 'extent'),
        ('sea-ice area (10$^6$ km$^2$)', 'ice', None, 0, 'area'),
        ('sea-ice volume (10$^3$ km$^3$)', 'ice', None, 0, 'volume'),
        ('sea-ice area increment (10$^6$ km$^2$)', 'ice', None, 0, 'incr'),
    ]
    ncol = 5
    fig, axes = plt.subplots(2, ncol, figsize=(4.4 * ncol, 7.0), squeeze=False)
    for a in axes.ravel()[len(panels):]:
        a.set_visible(False)
    drawn = False
    for i, (label, realm, var, lev, icekey) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        any_here = False
        if icekey == 'incr':
            # an increment is signed: mark the no-change line
            ax.axhline(0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for n in names:
            if icekey:
                # Hemispheres share the experiment's colour, so the marker
                # (filled NH / open SH) carries the distinction even when a
                # single cycle leaves no line to dash.
                for hemi, ls, mfc in (('nh', '-', None),
                                      ('sh', (0, (4, 2)), P.SURFACE)):
                    v = (_incr_series(cycles, order, n, hemi.upper())
                         if icekey == 'incr' else
                         _ice_series(cycles, order, n, '%s_%s' % (icekey, hemi)))
                    if not np.any(np.isfinite(v)):
                        continue
                    tt, vv = _finite(t, v)
                    ax.plot(tt, vv, 'o', color=col[n], ms=5.5, mec=col[n],
                            mew=1.4, mfc=mfc or col[n],
                            label='%s %s' % (n, hemi.upper()))
                    if not single:
                        ax.plot(tt, vv, ls=ls, color=col[n], lw=1.8)
                    any_here = True
            else:
                v = _bkg_series(cycles, order, n, realm, var, lev)
                if not np.any(np.isfinite(v)):
                    continue
                tt, vv = _finite(t, v)
                ax.plot(tt, vv, mk, color=col[n], label=n,
                        ms=6 if single else 5, mec=P.SURFACE, mew=1.0)
                any_here = True
        if not any_here:
            ax.set_visible(False)
            continue
        drawn = True
        ax.set_title(label, fontsize=10)
        P.tidy(ax)
        if single:
            ax.set_xticks(t)
            ax.set_xticklabels([t[0].strftime('%Y-%m-%d %HZ')], fontsize=8)
    if not drawn:
        plt.close(fig)
        return None
    for ax in axes.ravel():
        if ax.get_visible():
            P.maybe_legend(ax, fontsize=7.5)
    if not single:
        _date_labels(fig, axes, t)
    note = ('  (single cycle - add cycles to see drift)' if single else '')
    fig.suptitle('Background global means across cycles%s' % note, y=1.01,
                 fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return P.save(fig, cfg, 'cycle_background_drift.png')


def fig_increment_hovmoller(cycles, cfg, grid):
    """RMS increment as depth against cycle, one column per experiment.

    Reads only the cached profiles, so it covers every cycle cheaply -- the
    across-date view of the increments that map figures cannot give at scale.
    """
    if len(cycles) < 2:
        return None
    order = sorted(cycles)
    names = P.all_exp_names(cfg, cycles)
    t = _times(cycles)
    vars3d = [v for v in cfg.get('state_vars', {}).get('ocean', [])
              if any(len(P.get(cycles[c], 'state', n, 'ocean', 'incr_rms', v,
                               default=[]) or []) > 1
                     for c in order for n in names)]
    if not vars3d:
        return None

    fig, axes = plt.subplots(len(vars3d), len(names),
                             figsize=(4.6 * len(names), 3.4 * len(vars3d)),
                             squeeze=False)
    for r, var in enumerate(vars3d):
        panels = {}
        for n in names:
            cols = [P.get(cycles[c], 'state', n, 'ocean', 'incr_rms', var,
                          default=None) for c in order]
            if not any(cols):
                continue
            nk = max(len(c) for c in cols if c)
            M = np.full((nk, len(order)), np.nan)
            for j, c in enumerate(cols):
                if c:
                    M[:len(c), j] = c
            panels[n] = M
        if not panels:
            continue
        vmax = np.nanpercentile(np.concatenate(
            [m[np.isfinite(m)].ravel() for m in panels.values()]), 99)
        import plot_statespace as PS
        y, _ = PS.depth_axis(cycles[order[-1]], grid)
        for j, n in enumerate(names):
            ax = axes[r][j]
            if n not in panels:
                ax.set_visible(False)
                continue
            M = panels[n]
            yy = y[:M.shape[0]]
            h = ax.pcolormesh(t, yy, M, cmap=P.SEQUENTIAL,
                              vmin=0, vmax=vmax, shading='nearest',
                              rasterized=True)
            # Set the limits explicitly rather than calling invert_yaxis():
            # autoscaling from pcolormesh cell edges and then inverting used
            # to lose the shallow half of the profile.
            ax.set_ylim(float(np.nanmax(yy)), float(np.nanmin(yy)))
            P.depth_limit(ax, cfg)
            ax.grid(False)
            if r == 0:
                ax.set_title(n, fontsize=10, color=P.INK)
            if j == 0:
                ax.set_ylabel('%s\nnominal depth (m)' % var)
        cb = fig.colorbar(h, ax=list(axes[r]), fraction=0.02, pad=0.01)
        cb.outline.set_visible(False)
        cb.set_label('RMS increment (%s)' % var, fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
    fig.autofmt_xdate(rotation=30)
    fig.suptitle('Increment magnitude against depth and cycle', y=1.0,
                 fontsize=11.5, color=P.INK)
    return P.save(fig, cfg, 'cycle_increment_hovmoller.png')


def fig_increment_2d(cycles, cfg):
    """RMS increment of the 2-D fields against cycle."""
    if len(cycles) < 2:
        return None
    order = sorted(cycles)
    names = P.all_exp_names(cfg, cycles)
    col = P.color_map(names)
    t = _times(cycles)
    flat = ([('ocean', v) for v in cfg.get('state_vars', {}).get('ocean', [])]
            + [('ice', v) for v in cfg.get('state_vars', {}).get('ice', [])])
    panels = []
    for realm, var in flat:
        series = {}
        for n in names:
            v = np.array([(lambda p: p[0] if p and len(p) == 1 else np.nan)(
                P.get(cycles[c], 'state', n, realm, 'incr_rms', var,
                      default=None)) for c in order], dtype='f8')
            if np.any(np.isfinite(v)) and np.nanmax(np.abs(v)) > 0:
                series[n] = v
        if series:
            panels.append((realm, var, series))
    if not panels:
        return None

    nc = min(3, len(panels))
    nr = int(np.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.2 * nr), squeeze=False)
    for a in axes.ravel()[len(panels):]:
        a.set_visible(False)
    for i, (realm, var, series) in enumerate(panels):
        ax = axes[i // nc][i % nc]
        for n, v in series.items():
            tt, vv = _finite(t, v)
            ax.plot(tt, vv, 'o-', color=col[n], label=n, ms=4, mec=P.SURFACE,
                    mew=1.0)
        ax.set_title('%s  (%s)' % (var, realm), fontsize=10)
        # One field per panel, so the scale stays linear: a log axis flattens
        # exactly the cycle-to-cycle changes this figure exists to show.
        ax.set_ylim(bottom=0)
        P.tidy(ax)
        if i % nc == 0:
            ax.set_ylabel('RMS increment')
    P.maybe_legend(axes[0][0], fontsize=8.5)
    _date_labels(fig, axes, t)
    fig.suptitle('2-D field increment magnitude across cycles', y=1.01,
                 fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return P.save(fig, cfg, 'cycle_increment_2d.png')


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
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    cycles = P.load_cycles(cfg)
    types = sorted({t for d in cycles.values() for t in d.get('obs', {})})
    if any(P.type_sample(cycles, ot) != 'common' for ot in types):
        print('  ! some obs types have no genuine cross-experiment common '
              'sample -- falling back to each experiment\'s own sample for '
              'those (run compute_cycle.py --rejoin to fix this where '
              'possible)')
    print('cycling figures over %d cycle(s)' % len(cycles))
    for ot in types:
        fig_timeseries(cycles, cfg, ot, P.type_sample(cycles, ot))
    fig_obs_counts(cycles, cfg)
    fig_obs_fit(cycles, cfg)
    fig_verif_series(cycles, cfg)
    fig_drift(cycles, cfg)
    grid = Grid(cfg['grid'])
    fig_background_drift(cycles, cfg, grid)
    fig_increment_hovmoller(cycles, cfg, grid)
    fig_increment_2d(cycles, cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
