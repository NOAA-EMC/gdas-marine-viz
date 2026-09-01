#!/usr/bin/env python3
"""Observation-space figures. Reads only the cache written by compute_cycle.py."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
import lv_plot as P  # noqa: E402
from lv_common import load_config, region_list  # noqa: E402


def _grouped_bar(ax, types, names, values, colors, ylabel):
    """values[name][i] aligned with types."""
    x = np.arange(len(types))
    w = 0.8 / max(len(names), 1)
    for j, n in enumerate(names):
        v = np.array([values[n][i] for i in range(len(types))], dtype='f8')
        # 2px surface gap between adjacent bars
        ax.bar(x - 0.4 + w * (j + 0.5), v, w * 0.86, color=colors[n],
               label=n, edgecolor=P.SURFACE, linewidth=1.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([P.short(t) for t in types], rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    P.tidy(ax)


def fig_departures(data, cfg):
    """OmB and OmA RMS per obs type, common sample."""
    types = sorted(data['obs'])
    names = P.exp_names(data)
    col = P.color_map(names)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, key, title in (
            (axes[0], 'ombg_rms', 'Background fit  RMS(O$-$B)'),
            (axes[1], 'oman_rms', 'Analysis fit  RMS(O$-$A)')):
        vals = {n: [P.get(data['obs'][t], 'common', n, 'all', key)
                    for t in types] for n in names}
        _grouped_bar(ax, types, names, vals, col, 'RMS (obs units)')
        ax.set_title(title)
    P.maybe_legend(axes[0], loc='upper left', ncols=len(names))
    fig.suptitle('Departure statistics, common sample (obs passing QC in every '
                 'experiment) - %s' % data['cycle'],
                 y=1.03, fontsize=11.5, color=P.INK)
    return P.save(fig, cfg, 'obs_departures.png')


def fig_desroziers(data, cfg):
    """Are the assumed observation errors and the ensemble spread consistent?"""
    types = sorted(data['obs'])
    names = P.exp_names(data)
    col = P.color_map(names)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    y = np.arange(len(types))

    ax = axes[0]
    for j, n in enumerate(names):
        v = [P.get(data['obs'][t], 'common', n, 'all', 'desroziers_R_ratio')
             for t in types]
        ax.plot(v, y + (j - (len(names) - 1) / 2) * 0.16, 'o', color=col[n],
                label=n, ms=8, mec=P.SURFACE, mew=1.2, zorder=3)
    ax.axvline(1.0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.annotate('assumed R is correct', xy=(1.0, 1.0),
                xycoords=('data', 'axes fraction'), xytext=(4, -12),
                textcoords='offset points', fontsize=8, color=P.MUTED)
    ax.set_xscale('log')
    ax.set_xlabel('Desroziers $R$ / assigned $R$   (<1 = assigned error too large)')
    ax.set_title('Observation error')

    ax = axes[1]
    for j, n in enumerate(names):
        v = [P.get(data['obs'][t], 'common', n, 'all', 'desroziers_HBHt_ratio')
             for t in types]
        if not np.any(np.isfinite(np.array(v, dtype='f8'))):
            continue
        ax.plot(v, y + (j - (len(names) - 1) / 2) * 0.16, 'o', color=col[n],
                label=n, ms=8, mec=P.SURFACE, mew=1.2, zorder=3)
    ax.axvline(1.0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.annotate('spread matches implied $HBH^T$', xy=(1.0, 1.0),
                xycoords=('data', 'axes fraction'), xytext=(4, -12),
                textcoords='offset points', fontsize=8, color=P.MUTED)
    ax.set_xlabel('Desroziers $HBH^T$ / ensemble spread')
    ax.set_title('Background spread (ensemble experiments only)')

    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels([P.short(t) for t in types])
        ax.invert_yaxis()
        P.tidy(ax, xgrid=True)
        ax.grid(axis='y', visible=False)
    P.maybe_legend(axes[0], loc='lower right')
    fig.suptitle('Desroziers consistency diagnostics - %s' % data['cycle'],
                 y=1.02, fontsize=11.5, color=P.INK)
    return P.save(fig, cfg, 'obs_desroziers.png')


def fig_consistency(data, cfg):
    """RMS(O-B) against the spread that should match it, per obs type.

    If the ensemble and the assigned observation error are consistent then
    RMS(O-B) equals sqrt(sigma_b^2 + R^2). Putting them on one axis, with the
    two contributions beside them, turns the consistency ratio into a visible
    gap rather than a number to look up on another page.
    """
    types = sorted(data['obs'])
    names = [n for n in P.exp_names(data)
             if any(np.isfinite(P.get(data['obs'][t], 'common', n, 'all',
                                      'spread_b')) for t in types)]
    if not names:
        return None
    col = P.color_map(P.exp_names(data))
    ncol = min(3, len(types))
    nrow = int(np.ceil(len(types) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.5 * nrow),
                             squeeze=False)
    for a in axes.ravel()[len(types):]:
        a.set_visible(False)

    bars = [('RMS(O$-$B)', 'ombg_rms', 1.0),
            (r'$\sqrt{\sigma_b^2+R^2}$', None, 1.0),
            (r'ens spread $\sigma_b$', 'spread_b', 0.55),
            ('obs error $R$', 'R_assigned', 0.55)]
    for i, t in enumerate(types):
        ax = axes[i // ncol][i % ncol]
        x = np.arange(len(bars))
        w = 0.8 / max(len(names), 1)
        for j, n in enumerate(names):
            g = P.get(data['obs'][t], 'common', n, 'all', default={})
            sb = g.get('spread_b', np.nan)
            r = g.get('R_assigned', np.nan)
            vals = [g.get('ombg_rms', np.nan),
                    float(np.sqrt(sb ** 2 + r ** 2))
                    if np.isfinite(sb) and np.isfinite(r) else np.nan, sb, r]
            # The first two bars are the pair that must agree; the two
            # contributions behind the total are drawn lighter.
            for k, ((_lbl, _key, alpha), val) in enumerate(zip(bars, vals)):
                ax.bar(x[k] - 0.4 + w * (j + 0.5), val, w * 0.86, color=col[n],
                       label=n if k == 0 else None, edgecolor=P.SURFACE,
                       linewidth=1.0, zorder=3, alpha=alpha)
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars], fontsize=8)
        ax.set_title(P.short(t), fontsize=10)
        if i % ncol == 0:
            ax.set_ylabel('observation units')
        P.tidy(ax)
    P.maybe_legend(axes[0][0], fontsize=8)
    fig.suptitle('Departure against the spread that should match it '
                 '(first two bars agree when calibrated) - %s' % data['cycle'],
                 y=1.01, fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return P.save(fig, cfg, 'obs_consistency.png')


def fig_spread(data, cfg):
    """Consistency ratio and posterior/prior spread collapse."""
    types = sorted(data['obs'])
    names = [n for n in P.exp_names(data)
             if any(P.get(data['obs'][t], 'common', n, 'all', 'spread_b')
                    == P.get(data['obs'][t], 'common', n, 'all', 'spread_b')
                    for t in types)]
    if not names:
        return None
    col = P.color_map(P.exp_names(data))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    vals = {n: [P.get(data['obs'][t], 'common', n, 'all', 'consistency_ratio')
                for t in types] for n in names}
    _grouped_bar(axes[0], types, names, vals, col,
                 r'$(\sigma_b^2 + R^2)\,/\,\overline{(O-B)^2}$')
    axes[0].set_yscale('log')
    P.ref_line(axes[0], 1.0, 'calibrated = 1')
    axes[0].set_title('Consistency ratio\n<1 under-dispersive, >1 over-dispersive '
                      'or R too large')

    vals = {n: [P.get(data['obs'][t], 'common', n, 'all', 'spread_ratio')
                for t in types] for n in names}
    _grouped_bar(axes[1], types, names, vals, col,
                 r'$\sigma_a / \sigma_b$ at observation points')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Posterior / prior ensemble spread\n'
                      'a very small ratio means the analysis is over-confident')
    P.maybe_legend(axes[0], loc='upper left', ncols=len(names))
    fig.suptitle('Ensemble calibration in observation space - %s' % data['cycle'],
                 y=1.05, fontsize=11.5, color=P.INK)
    return P.save(fig, cfg, 'obs_spread.png')


def fig_rank(data, cfg):
    """Rank histograms: U-shaped means the prior ensemble is under-dispersive."""
    types = [t for t in sorted(data['obs']) if data['obs'][t].get('rank')]
    if not types:
        return None
    names = P.exp_names(data)
    col = P.color_map(names)
    nc = min(4, len(types))
    nr = int(np.ceil(len(types) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.7 * nr),
                             squeeze=False)
    for a in axes.ravel()[len(types):]:
        a.set_visible(False)
    for i, t in enumerate(types):
        ax = axes[i // nc][i % nc]
        for n in names:
            rh = data['obs'][t]['rank'].get(n)
            if not rh:
                continue
            c = np.array(rh['counts'], dtype='f8')
            ax.step(np.arange(len(c)), c / rh['flat'], where='mid',
                    color=col[n], lw=2.0, label=n)
        ax.axhline(1.0, color=P.MUTED, lw=1.1, ls=(0, (4, 3)))
        ax.set_title(P.short(t), fontsize=9.5)
        ax.set_xlabel('rank of obs among members')
        if i % nc == 0:
            ax.set_ylabel('count / flat')
        P.tidy(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(labels) >= 2:
        fig.legend(handles, labels, loc='lower center', ncols=len(labels),
                   bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Rank histograms - flat is calibrated, U-shaped is '
                 'under-dispersive (%s)' % data['cycle'],
                 y=1.01, fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return P.save(fig, cfg, 'obs_rank_histograms.png')


def fig_spread_skill(data, cfg):
    """Binned spread against realised error; the 1:1 line is calibration."""
    types = [t for t in sorted(data['obs']) if data['obs'][t].get('spread_skill')]
    if not types:
        return None
    names = P.exp_names(data)
    col = P.color_map(names)
    nc = min(4, len(types))
    nr = int(np.ceil(len(types) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.9 * nr), squeeze=False)
    for a in axes.ravel()[len(types):]:
        a.set_visible(False)
    for i, t in enumerate(types):
        ax = axes[i // nc][i % nc]
        lim = 0.0
        for n in names:
            ss = data['obs'][t]['spread_skill'].get(n)
            if not ss:
                continue
            ax.plot(ss['spread'], ss['rmse'], 'o-', color=col[n], ms=6,
                    mec=P.SURFACE, mew=1.0, label='%s (slope %.2f)'
                    % (n, ss['slope']))
            lim = max(lim, max(ss['spread']), max(ss['rmse']))
        if lim:
            ax.plot([0, lim], [0, lim], color=P.MUTED, lw=1.1, ls=(0, (4, 3)),
                    zorder=1)
        ax.set_title(P.short(t), fontsize=9.5)
        ax.set_xlabel(r'$\sqrt{\sigma_b^2 + R^2}$')
        if i % nc == 0:
            ax.set_ylabel('RMS(O$-$B)')
        ax.legend(fontsize=7.5, loc='upper left')
        P.tidy(ax, xgrid=True)
    fig.suptitle('Spread-skill relationship - points on the dashed 1:1 line are '
                 'calibrated (%s)' % data['cycle'],
                 y=1.01, fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return P.save(fig, cfg, 'obs_spread_skill.png')


def _profile_band(key, rec):
    """Half-width of the shaded band for a profile metric, or None.

    For the mean departure it is the observation-to-observation scatter in that
    region and depth bin, sqrt(rms^2 - mean^2) -- the spread the regional mean
    is hiding. For the RMS it is the sampling uncertainty of an RMS estimate,
    rms/sqrt(2n), which matters once the sample is sliced by band and depth and
    the deep bins get thin.
    """
    n = np.array([r.get('n', 0) or 0 for r in rec], dtype='f8')
    rms = np.array([r.get('ombg_rms', np.nan) for r in rec], dtype='f8')
    if key == 'ombg_mean':
        mean = np.array([r.get('ombg_mean', np.nan) for r in rec], dtype='f8')
        return np.sqrt(np.maximum(rms ** 2 - mean ** 2, 0.0))
    if key == 'ombg_rms':
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n > 1, rms / np.sqrt(2.0 * n), np.nan)
    return None


# The departure error budget, drawn on one axis: the first two must agree when
# the ensemble and the assigned observation error are consistent, and the two
# behind them say which side is responsible when they do not.
_BUDGET = (
    ('ombg_rms', 'RMS(O$-$B)', '-', 2.0),
    ('total', r'$\sqrt{\sigma_b^2+R^2}$', (0, (4, 2)), 1.8),
    ('spread_b', r'ens spread $\sigma_b$', (0, (1, 1.6)), 1.5),
    ('R_assigned', 'obs error $R$', (0, (5, 1.5, 1, 1.5)), 1.5),
)


def _budget_series(rec, key):
    if key == 'total':
        sb = np.array([r.get('spread_b', np.nan) for r in rec], dtype='f8')
        rr = np.array([r.get('R_assigned', np.nan) for r in rec], dtype='f8')
        return np.sqrt(sb ** 2 + rr ** 2)
    return np.array([r.get(key, np.nan) for r in rec], dtype='f8')


def _budget_panel(ax, data, t, names, col, labels, centres, pre):
    """RMS(O-B), the total spread, and its two parts, against depth."""
    drawn = 0
    for n in names:
        rec = [P.get(data['obs'][t], 'common', n, pre + L, default={})
               for L in labels]
        for key, lbl, ls, lw in _BUDGET:
            v = _budget_series(rec, key)
            if not np.any(np.isfinite(v)):
                continue
            if key == 'ombg_rms':
                band = _profile_band(key, rec)
                if band is not None:
                    ax.fill_betweenx(centres, v - band, v + band,
                                     color=col[n], alpha=0.16, linewidth=0)
            ax.plot(v, centres, color=col[n], ls=ls, lw=lw,
                    label='%s %s' % (n, lbl))
            drawn += 1
    return drawn


_PROFILE_METRICS = (
    ('ombg_mean', 'mean O$-$B (bias), band = $\\pm$1 sd of departures'),
    ('budget', 'departure and the spread that should match it'),
    ('consistency_ratio', 'consistency ratio'))


def fig_profiles(data, cfg):
    """Argo temperature and salinity statistics against depth, one PNG per
    region rather than one big grid with a column per region: with several
    basins configured that grid squeezed every region into a sliver too
    narrow to read, the same problem fig_verif_series had. build_report.py
    embeds all of them behind a region picker; it derives the same region
    list from the cache independently by checking which obs_profiles_region_
    *.png files this wrote, so it needs no separate presence filter of its
    own.
    """
    types = [t for t in sorted(data['obs']) if data['obs'][t].get('is_profile')]
    if not types:
        return []
    names = P.exp_names(data)
    col = P.color_map(names)
    bins = cfg.get('depth_bins', [])
    centres = [0.5 * (a + b) for a, b in zip(bins[:-1], bins[1:])]
    labels = ['depth_%g_%g' % (a, b) for a, b in zip(bins[:-1], bins[1:])]

    # Regions the cache actually carries: 'global' plus any latitude band that
    # was stratified at compute time.
    have = set()
    for t in types:
        for n in names:
            have |= set(P.get(data['obs'][t], 'common', n, default={}))
    regions = ['global'] + [name for name in region_list(cfg)
                            if any(('%s/%s' % (name, L)) in have
                                   for L in labels)]

    written = []
    for reg in regions:
        pre = '' if reg == 'global' else reg + '/'
        fig, axes = plt.subplots(len(types), len(_PROFILE_METRICS),
                                 figsize=(3.5 * len(_PROFILE_METRICS),
                                          3.4 * len(types)),
                                 squeeze=False)
        drawn_any = False
        for ri, t in enumerate(types):
            for c, (key, title) in enumerate(_PROFILE_METRICS):
                ax = axes[ri][c]
                if key == 'budget':
                    drawn = _budget_panel(ax, data, t, names, col, labels,
                                          centres, pre)
                else:
                    drawn = 0
                    for n in names:
                        rec = [P.get(data['obs'][t], 'common', n, pre + L,
                                     default={}) for L in labels]
                        v = np.array([rr.get(key, np.nan) for rr in rec],
                                     dtype='f8')
                        if not np.any(np.isfinite(v)):
                            continue
                        band = _profile_band(key, rec)
                        if band is not None:
                            ax.fill_betweenx(centres, v - band, v + band,
                                             color=col[n], alpha=0.16,
                                             linewidth=0)
                        ax.plot(v, centres, 'o-', color=col[n], label=n, ms=5,
                                mec=P.SURFACE, mew=1.0)
                        drawn += 1
                if not drawn:
                    ax.set_visible(False)
                    continue
                drawn_any = True
                if key == 'ombg_mean':
                    ax.axvline(0.0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)))
                if key == 'consistency_ratio':
                    ax.axvline(1.0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)))
                    ax.set_xscale('log')
                ax.invert_yaxis()
                P.depth_limit(ax, cfg)
                ax.set_xlabel(title)
                if c == 0:
                    ax.set_ylabel('%s\ndepth (m)' % P.short(t))
                P.tidy(ax, xgrid=True)
            P.maybe_legend(axes[ri][0], fontsize=7)
        if not drawn_any:
            plt.close(fig)
            continue
        fig.suptitle('%s: profile observations against depth, common sample '
                     '- %s' % (reg.replace('_', ' '), data['cycle']),
                     y=1.0, fontsize=11.5, color=P.INK)
        fig.tight_layout()
        written.append(P.save(fig, cfg,
                              'obs_profiles_region_%s.png' % P.slug(reg)))
    return written


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
    ap.add_argument('--cycle', default=None)
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    cycles = P.load_cycles(cfg, [a.cycle] if a.cycle else None)
    cycle = a.cycle or sorted(cycles)[-1]
    data = cycles[cycle]
    print('obs-space figures for %s' % cycle)
    # fig_desroziers is still generated: it is off the report page but remains
    # the quickest read on whether an assigned observation error is wrong.
    for f in (fig_departures, fig_consistency, fig_desroziers, fig_spread,
              fig_rank, fig_spread_skill, fig_profiles):
        f(data, cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
