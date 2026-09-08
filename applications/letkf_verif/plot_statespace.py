#!/usr/bin/env python3
"""State-space figures. Reads only the cache written by compute_cycle.py."""

import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.path as mpath  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import lv_plot as P  # noqa: E402
import lv_verif as LV  # noqa: E402
from lv_common import (Grid, basin_at, basin_regions,  # noqa: E402
                       load_config, region_box, region_list)
import lv_woa  # noqa: E402
from lv_statespace import TRIPOLAR_LAT, section_warnings  # noqa: E402

LAND = '#cfcdc6'
COAST = '#8d8b84'

# The gridspec longitude runs -300..60 and is monotonic along i, with no jump
# anywhere. Centring the projection on the middle of that span puts the map
# seam exactly at the array's own edge, so no cell ever straddles it and no
# seam masking is needed. (Wrapping longitude into -180..180 first, which is
# the obvious move, *introduces* a ragged discontinuity that is not in the
# data.)
DATA_LON0 = -120.0


TAG = ''


MAP_DPI = 100          # rasterised map panels; profiles keep the default

# The verification section carries six map figures -- product-and-model plus
# model-minus-product, for three products -- and the report has a hard 16 MB
# publishing ceiling. These are comparison maps read for pattern, not for
# detail, so they get a lower resolution than the analysis maps.
VERIF_MAP_DPI = 78

# Background fields that take both signs and so want a diverging scale centred
# on zero rather than the sequential ramp the other states use. Temperature is
# deliberately not here: it goes below zero in polar water, but it is still a
# state whose absolute value is what you read, not a departure from zero.
SIGNED_FIELDS = {'u', 'v'}

# Sequential colormap by variable family, keyed on the row-key base name
# (before "_kN"). Every family is on jet (P.SEQ_BKG) for now, as is the
# fallback for anything not listed -- ssh, MLD and speed. The per-family
# structure is kept so a single field can be moved back to one of the
# perceptually-uniform ramps (P.SEQ_WARM/SEQ_TEAL/SEQ_ICE) on its own.
FIELD_CMAP = {
    'Temp': P.SEQ_BKG,
    'Salt': P.SEQ_BKG,
    'aice_h': P.SEQ_BKG,
    'hi_div_aice_h': P.SEQ_BKG,
    'hs_div_aice_h': P.SEQ_BKG,
}

# Row keys with no real vertical structure. CICE/MOM6 2-D diagnostics still
# come out of background_maps() as "<var>_k0" (the level-keyed cache format
# forces a k0 suffix on every field), but level_label()'s depth lookup reads
# the *ocean* h-based depth axis -- meaningless for these, and actively wrong
# for the ice fields (a "~1 m" tacked onto an ice concentration). Their row
# label is just the bare variable name.
NO_LEVEL_LABEL = {'ave_ssh', 'MLD', 'aice_h', 'hi_div_aice_h', 'hs_div_aice_h'}


# The climatology column / line. Not an experiment: a fixed neutral colour
# from OUTSIDE P.color_map()'s cycle and a distinct dash, so it can never be
# misread as one of the runs being compared.
WOA_NAME = 'WOA23'
WOA_STYLE = dict(color=P.INK2, ls=(0, (1, 1.6)), lw=2.0, zorder=5)


def _first(fields):
    """The first present field in a per-experiment list, or None.

    Every experiment caches the same climatology (rebuilt per experiment onto
    its own depth axis, which differ by centimetres), so whichever has it is
    authoritative -- the rule fig_verif_maps already uses for its product.
    """
    return next((f for f in (fields or ()) if f is not None), None)


def _field_cmap(key):
    return FIELD_CMAP.get(key.rsplit('_k', 1)[0], P.SEQ_BKG)


def _fixed_limits(cfg, realm, key, hemi=None):
    """(vmin, vmax, cmap) from cfg['map_limits'], or None if not declared.

    Fixed rather than computed from data, so the same variable renders on the
    same scale in every figure, every cycle and every experiment -- including
    experiments that never share a cycle (cp06.torchbalance.dual's Jan window
    never overlaps cp06.torchbalance/3dvar-rt's Dec one, so there is no single
    figure to pool a percentile from across all three).

    A dict value is a per-hemisphere override, e.g. {'nh': [0, 4], 'sh':
    [0, 2]} -- only sea ice thickness/snow depth need this: Arctic and
    Antarctic ice sit on very different scales.
    """
    entry = ((cfg.get('map_limits') or {}).get(realm) or {}).get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        entry = entry.get(hemi)
        if entry is None:
            return None
    lo, hi = entry
    return float(lo), float(hi), _field_cmap(key)


def _fixed_incr_limits(cfg, realm, key, hemi=None, block='increment'):
    """Symmetric (-lim, lim, DIVERGING) from a signed cfg['map_limits'] block.

    ``block`` selects which one: 'increment' for the analysis step,
    'woa_bias' for the departure from the climatology. They are kept apart
    because they sit on very different sizes -- one analysis increment against
    a whole model-minus-ocean departure.
    """
    entry = (((cfg.get('map_limits') or {}).get(block) or {})
             .get(realm) or {}).get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        entry = entry.get(hemi)
        if entry is None:
            return None
    lim = float(entry)
    return -lim, lim, P.DIVERGING


def _save(fig, cfg, base, dpi=None):
    """Save, inserting the per-cycle tag before the extension when set.

    Quiet: render_cycle() prints one line per figure with its timing.
    """
    stem, ext = os.path.splitext(base)
    return P.save(fig, cfg, '%s%s%s' % (stem, TAG, ext), dpi=dpi, quiet=True)


def _is3d(prof):
    return isinstance(prof, list) and len(prof) > 1


def global_proj():
    return ccrs.PlateCarree(central_longitude=DATA_LON0)


def _circular_boundary(ax):
    """Clip a polar-stereographic panel to a disc, not the default square."""
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.column_stack([0.5 + 0.5 * np.cos(theta),
                             0.5 + 0.5 * np.sin(theta)])
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)


def _decorate(ax, polar=False):
    ax.add_feature(cfeature.LAND, facecolor=LAND, edgecolor='none', zorder=2)
    ax.coastlines(resolution='110m', linewidth=0.35, color=COAST, zorder=3)
    ax.spines['geo'].set_edgecolor(P.GRID)
    ax.spines['geo'].set_linewidth(0.8)
    gl = ax.gridlines(linewidth=0.3, color=P.GRID, alpha=0.8, zorder=4)
    gl.top_labels = gl.right_labels = False
    if not polar:
        gl.left_labels = gl.bottom_labels = False


def depth_axis(data, grid, n=None):
    """Per-cycle mid-depths from the background layer thickness.

    The reference experiment's axis is used for shared y-axes. A cycle with no
    background falls back to the level index rather than a nominal depth in
    metres: a wrong depth in metres is worse than an honest level number.
    """
    ref = data.get('reference')
    order = ([ref] if ref in data.get('state', {}) else []) + [
        k for k in data.get('state', {}) if k != ref]
    for name in order:
        d = data['state'][name].get('depth')
        if d:
            d = np.asarray(d, dtype='f8')
            return (d[:n] if n else d), data['state'][name].get(
                'depth_source', 'background h')
    return (np.arange(1, (n or 75) + 1, dtype='f8'), 'level index')


def level_depth(data, k):
    """Approximate depth of model level ``k`` in metres, or None.

    `map_levels:` are INDICES into a 75-level column, not depths, and on this
    grid level 10 is 21 m while level 30 is 102 m -- a reader given only
    "level 30" has no way to tell whether that is 100 m or 1000 m. The depth
    is already cached per experiment, so every panel that names a level can
    say what it means.

    Approximate because the model uses z*: the depth of a level moves with the
    free surface, and differs slightly between experiments. The reference
    experiment's column is used.
    """
    ref = data.get('reference')
    order = ([ref] if ref in data.get('state', {}) else []) + [
        n for n in data.get('state', {}) if n != ref]
    for name in order:
        d = data['state'][name].get('depth')
        if d and 0 <= k < len(d):
            return float(d[k])
    return None


def level_label(data, k, sep=' '):
    """'level 30 (~102 m)', falling back to 'level 30' with no depth cached."""
    z = level_depth(data, k)
    if z is None:
        return 'level %d' % k
    return 'level %d%s(~%s m)' % (k, sep, ('%.0f' % z) if z >= 10 else ('%.1f' % z))


def _map_row_label(data, key):
    """'Temp_k30' -> 'Temp / level 30 (~102 m)'. Keys without _k pass through.

    NO_LEVEL_LABEL fields (ssh, MLD, every ice field) have no real vertical
    axis, so they get the bare variable name with no fabricated depth.
    """
    base, _, lvl = key.rpartition('_k')
    if not base or not lvl.isdigit():
        return key
    if base in NO_LEVEL_LABEL:
        return base
    return '%s\n%s' % (base, level_label(data, int(lvl), sep='\n'))


def _stride(data, cfg):
    """Subsample factor the cached maps were written with.

    Recorded per experiment as well as per cycle so that caches written with
    different strides can still be merged; the shape guard in _map_panel is
    the backstop if they genuinely disagree.
    """
    for rec in (data.get('state') or {}).values():
        if isinstance(rec, dict) and rec.get('map_stride'):
            return int(rec['map_stride'])
    return int(data.get('map_stride') or cfg.get('map_stride', 2))


def _map_panel(ax, lon, lat, f, cmap, vmin, vmax, polar=False):
    """Draw one field on a GeoAxes. ``lon`` must be the raw gridspec longitude."""
    if f.shape != lon.shape:
        raise ValueError(
            'cached map is %s but the grid subsamples to %s -- the cache was '
            'written with a different map_stride; rerun compute_cycle.py '
            '--force' % (f.shape, lon.shape))
    fm = np.ma.masked_invalid(f)
    # The tripolar coordinates are curvilinear by construction, so matplotlib's
    # monotonicity warning does not apply; 'nearest' treats them as centres.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        h = ax.pcolormesh(lon, lat, fm, cmap=cmap, vmin=vmin, vmax=vmax,
                          shading='nearest', rasterized=True, zorder=1,
                          transform=ccrs.PlateCarree())
    _decorate(ax, polar)
    return h


def _robust(vals, q=99.0):
    v = np.concatenate([np.abs(x[np.isfinite(x)]).ravel() for x in vals
                        if x is not None and np.any(np.isfinite(x))])
    return float(np.percentile(v, q)) if v.size else 1.0


# --------------------------------------------------------------------------

def region_names(data, cfg, block):
    """Regions present in the cache, global first, in config order."""
    have = set()
    for n in P.exp_names(data):
        have |= set(P.get(data['state'], n, 'ocean', block, default={}))
    order = ['global'] + region_list(cfg)
    return [r for r in order if r in have]


def profile_panel(ax, data, grid, names, col, block, var, region,
                  key='mean', sd_key='sd', band=True, ref=None, floor=None,
                  ls=None, label='%s', cfg=None, extra=None):
    """One depth profile per experiment, with the in-region spatial spread.

    Shared by the increment, spread and background profile figures: they differ
    only in which cached block and which series inside it they read.

    ``extra`` is [(label, values, style)] drawn after the experiments -- for
    the WOA23 climatology, which is ONE curve for the whole panel rather than
    one per experiment. Routing it through ``names`` would draw N identical
    lines, since every experiment caches the same field.
    """
    drawn = 0
    for n in names:
        rec = P.get(data['state'], n, 'ocean', block, region, var, default=None)
        if not rec or not _is3d(rec.get(key)):
            continue
        m = np.asarray(rec[key], dtype='f8')
        y, _ = depth_axis(data, grid, len(m))
        style = dict(color=col[n], label=label % n)
        if ls is not None:
            style.update(ls=ls)
        elif ref is not None and n != ref:
            style.update(ls=(0, (4, 2)), lw=1.6)
        ax.plot(m, y, **style)
        if band and sd_key and rec.get(sd_key):
            sd = np.asarray(rec[sd_key], dtype='f8')
            lo = m - sd
            if floor is not None:      # RMS and spread are non-negative
                lo = np.maximum(lo, floor)
            ax.fill_betweenx(y, lo, m + sd, color=col[n], alpha=0.16,
                             linewidth=0)
        drawn += 1
    for lab, vals, style in (extra or ()):
        if not _is3d(vals):
            continue
        v = np.asarray(vals, dtype='f8')
        y, _ = depth_axis(data, grid, len(v))
        ax.plot(v, y, label=lab, **style)
        drawn += 1
    # Idempotent: the spread panels call this twice on one axis (prior, then
    # posterior), and a second invert_yaxis() would put the surface at the
    # bottom again.
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    P.depth_limit(ax, cfg)
    P.tidy(ax, xgrid=True)
    return drawn


def _vars3d(data, cfg, names, block, varlist=None, key='mean'):
    """Config-ordered variables that this cached block holds as profiles."""
    order = varlist or cfg['state_vars']['ocean']
    return [v for v in order
            if any(_is3d((P.get(data['state'], n, 'ocean', block, 'global', v,
                                default=None) or {}).get(key))
                   for n in names)]


def fig_regional_profiles(data, cfg, grid, block='incr_region',
                          label='RMS increment', fname='state_increment_regions.png',
                          band=True, floor=None, varlist=None):
    """Profiles by region, one PNG per region: rows are variables.

    One file per region rather than one big grid with a column per region --
    with several basins configured that grid squeezed every region into a
    sliver too narrow to read, the same problem fig_verif_series (in
    plot_timeseries.py) had. build_report.py embeds all of them behind a
    region picker; it derives the same region list independently by
    checking which <base>_region_*.png files this wrote.

    The shaded band is the area-weighted spread across columns inside the
    region -- narrow means the region is being updated uniformly, wide means
    the regional mean is hiding structure.
    """
    names = P.exp_names(data)
    col = P.color_map(names)
    regions = region_names(data, cfg, block)
    if not regions:
        return []
    ref = data.get('reference')
    vars3d = _vars3d(data, cfg, names, block, varlist)
    if not vars3d:
        return []

    base = fname[:-4] if fname.endswith('.png') else fname
    written = []
    for reg in regions:
        fig, axes = plt.subplots(len(vars3d), 1,
                                 figsize=(4.6, 3.6 * len(vars3d)), squeeze=False)
        drawn_any = False
        for r, v in enumerate(vars3d):
            ax = axes[r][0]
            if not profile_panel(ax, data, grid, names, col, block, v, reg,
                                 band=band, ref=ref, floor=floor, cfg=cfg):
                ax.set_visible(False)
                continue
            drawn_any = True
            ax.set_ylabel('%s\ndepth (m)' % v)
            ax.set_xlabel('%s (%s)' % (label, v))
        if not drawn_any:
            plt.close(fig)
            continue
        P.maybe_legend(axes[0][0], fontsize=8)
        fig.suptitle('%s: %s by region, shaded band = spatial spread within '
                     'region - %s' % (reg.replace('_', ' '), label,
                                      data['cycle']),
                     y=1.01, fontsize=11.5, color=P.INK)
        fig.tight_layout()
        written.append(_save(fig, cfg, '%s_region_%s.png' % (base, P.slug(reg))))
    return written


def fig_spread_regions(data, cfg, grid):
    """Ensemble spread and spread reduction against depth, by region.

    Localization and inflation do not act uniformly in latitude -- a global
    spread profile can look healthy while the tropics are collapsed and the
    Southern Ocean is untouched. Two rows per variable.

    The top row carries the three stages of the spread -- prior, analysis, and
    analysis after inflation -- so the analysis's compression and inflation's
    restoration are read off one axis. The bottom row carries the two ratios
    against the prior: what the update removed, and what actually survives to
    propagate. The gap between those two curves is the work inflation did.

    The post-inflation stage is drawn only where it exists. It is absent for a
    variable the post-inflation file does not carry (there is no `ave_ssh` in
    it) and for every variable in a cache written before that file existed, in
    which case this is the two-stage figure it has always been.
    """
    names = P.exp_names(data)
    col = P.color_map(names)
    block = 'spread_region'
    regions = region_names(data, cfg, block)
    if not regions:
        return None
    have = [n for n in names
            if P.get(data['state'], n, 'ocean', block, default=None)]
    vars3d = _vars3d(data, cfg, have, block, key='prior')
    if not have or not vars3d:
        return None

    nrow = 2 * len(vars3d)
    fig, axes = plt.subplots(nrow, len(regions),
                             figsize=(3.3 * len(regions), 4.2 * nrow),
                             squeeze=False)
    for i, v in enumerate(vars3d):
        for c, reg in enumerate(regions):
            axS, axR = axes[2 * i][c], axes[2 * i + 1][c]
            drawn = profile_panel(axS, data, grid, have, col, block, v, reg,
                                  key='prior', sd_key=None, ls='-',
                                  label='%s prior', cfg=cfg)
            profile_panel(axS, data, grid, have, col, block, v, reg,
                          key='post', sd_key=None, ls=(0, (3, 2)),
                          label='%s analysis', cfg=cfg)
            infl = profile_panel(axS, data, grid, have, col, block, v, reg,
                                 key='an', sd_key=None, ls=(0, (1, 1.4)),
                                 label='%s inflated', cfg=cfg)
            profile_panel(axR, data, grid, have, col, block, v, reg,
                          key='ratio', sd_key=None, ls='-',
                          label=r'%s $\sigma_a/\sigma_b$', cfg=cfg)
            profile_panel(axR, data, grid, have, col, block, v, reg,
                          key='ratio_net', sd_key=None, ls=(0, (1, 1.4)),
                          label=r'%s $\sigma_{an}/\sigma_b$', cfg=cfg)
            if not drawn:
                axS.set_visible(False)
                axR.set_visible(False)
                continue
            axR.set_xlim(0, 1)
            if 2 * i == 0:
                axS.set_title(reg.replace('_', ' '), fontsize=10, color=P.INK)
            if c == 0:
                axS.set_ylabel('%s\ndepth (m)' % v)
                axR.set_ylabel('%s\ndepth (m)' % v)
            axS.set_xlabel('ensemble spread (%s)' % v)
            axR.set_xlabel(('ratio to prior (%s)' if infl
                            else r'$\sigma_a/\sigma_b$ (%s)') % v)
    P.maybe_legend(axes[0][0], fontsize=7.5)
    P.maybe_legend(axes[1][0], fontsize=7.5)
    stages = ('solid prior / dashed analysis / dotted inflated'
              if any(P.get(data['state'], n, 'ocean', 'spread_an', default=None)
                     for n in have)
              else 'solid prior / dashed analysis')
    fig.suptitle('Ensemble spread against depth by region, %s - %s'
                 % (stages, data['cycle']), y=1.01,
                 fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return _save(fig, cfg, 'state_spread_regions.png')


def fig_increment_profiles(data, cfg, grid):
    """RMS increment against depth -- where each system is putting its update."""
    names = P.exp_names(data)
    col = P.color_map(names)
    ocn = {n: P.get(data['state'], n, 'ocean', 'incr_rms', default={})
           for n in names}
    vars3d = [v for v in cfg['state_vars']['ocean']
              if any(_is3d(ocn[n].get(v)) for n in names)]
    vars2d = [v for v in cfg['state_vars']['ocean'] if v not in vars3d]
    ice = {n: P.get(data['state'], n, 'ice', 'incr_rms', default={})
           for n in names}
    ivars = [v for v in cfg.get('state_vars', {}).get('ice', [])
             if any(ice[n].get(v) for n in names)]
    flat = [(v, 'ocean') for v in vars2d] + [(v, 'ice') for v in ivars]

    ncol = len(vars3d) + len(flat)
    fig, axes = plt.subplots(
        1, ncol, squeeze=False,
        figsize=(3.6 * len(vars3d) + 2.1 * len(flat), 4.6),
        gridspec_kw=dict(width_ratios=[1.0] * len(vars3d) + [0.58] * len(flat)))
    axes = axes[0]

    for c, v in enumerate(vars3d):
        ax = axes[c]
        for n in names:
            p = ocn[n].get(v)
            if not _is3d(p):
                continue
            y, dsrc = depth_axis(data, grid, len(p))
            ax.plot(np.array(p, dtype='f8'), y, color=col[n], label=n)
        ax.invert_yaxis()
        P.depth_limit(ax, cfg)
        ax.set_xlabel('RMS increment (%s)' % v)
        if c == 0:
            ax.set_ylabel('depth (m)')
        ax.set_title(v)
        P.tidy(ax, xgrid=True)
    P.maybe_legend(axes[0], loc='lower right', fontsize=8.5)

    # One linear axis per 2-D field. Sharing a single axis across fields whose
    # units differ by orders of magnitude forced a log scale, which turns a
    # magnitude comparison into a comparison of exponents; a panel per field
    # keeps the scale linear and the bar heights meaningful.
    x = np.arange(len(names))
    for c, (v, realm) in enumerate(flat):
        ax = axes[len(vars3d) + c]
        vals = [np.asarray((ocn[n] if realm == 'ocean' else ice[n])
                           .get(v, [np.nan]), dtype='f8')[0] for n in names]
        ax.bar(x, vals, 0.62, color=[col[n] for n in names],
               edgecolor=P.SURFACE, linewidth=1.0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_title('%s\n%s' % (v, realm), fontsize=9)
        ax.set_ylim(bottom=0)
        P.tidy(ax)
    if flat:
        axes[len(vars3d)].set_ylabel('RMS increment (field units)')
    fig.suptitle('Increment magnitude, area-weighted over the wet grid - %s'
                 % data['cycle'], y=1.02, fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return _save(fig, cfg, 'state_increment_profiles.png')


def fig_spread_profiles(data, cfg, grid):
    """Prior and posterior ensemble spread, and how far the analysis cut it."""
    names = [n for n in P.exp_names(data)
             if P.get(data['state'], n, 'ocean', 'spread_prior', default=None)]
    if not names:
        return None
    col = P.color_map(P.exp_names(data))
    vars3d = [v for v in cfg['state_vars']['ocean']
              if _is3d(P.get(data['state'], names[0], 'ocean', 'spread_prior',
                             v, default=None))]
    fig, axes = plt.subplots(1, 2 * len(vars3d),
                             figsize=(3.5 * 2 * len(vars3d), 4.6), squeeze=False)
    axes = axes[0]
    for i, v in enumerate(vars3d):
        axL, axR = axes[2 * i], axes[2 * i + 1]
        for n in names:
            pr = P.get(data['state'], n, 'ocean', 'spread_prior', v, default=None)
            po = P.get(data['state'], n, 'ocean', 'spread_post', v, default=None)
            rt = P.get(data['state'], n, 'ocean', 'spread_ratio', v, default=None)
            if pr is None:
                continue
            y, _ = depth_axis(data, grid, len(pr))
            axL.plot(np.array(pr, dtype='f8'), y, color=col[n],
                     label='%s prior' % n)
            axL.plot(np.array(po, dtype='f8'), y, color=col[n], ls=(0, (3, 2)),
                     label='%s posterior' % n)
            if rt is not None:
                axR.plot(np.array(rt, dtype='f8'), y, color=col[n], label=n)
        for ax in (axL, axR):
            ax.invert_yaxis()
            P.depth_limit(ax, cfg)
            P.tidy(ax, xgrid=True)
        axL.set_xlabel('ensemble spread (%s)' % v)
        axL.set_title('%s spread' % v)
        axR.set_xlim(0, 1)
        axR.set_xlabel(r'$\sigma_a/\sigma_b$')
        axR.set_title('%s spread reduction' % v)
        axL.legend(fontsize=7.5, loc='lower right')
        if i == 0:
            axL.set_ylabel('depth (m)')
    fig.suptitle('Ensemble spread against depth - %s' % data['cycle'],
                 y=1.02, fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return _save(fig, cfg, 'state_spread_profiles.png')


def woa_curve(data, block, region, var, key='mean'):
    """The WOA profile from whichever experiment cached it, or None."""
    for n in P.exp_names(data):
        rec = P.get(data['state'], n, 'ocean', block, region, var, default=None)
        if rec and _is3d(rec.get(key)):
            return rec[key]
    return None


def woa_extra(data, block, region, var):
    """`extra=` for profile_panel: the climatology curve, or nothing."""
    w = woa_curve(data, block, region, var)
    return [(WOA_NAME, w, WOA_STYLE)] if w is not None else []


def fig_woa_bias_profiles(data, cfg, grid):
    """Background against WOA23, and their difference, over depth.

    The most quantitative of the climatology views: left the two states
    overlaid, right what separates them. WOA is a 1955-2022 decadal mean, so
    the difference carries the ocean's real interannual anomaly as well as any
    model error -- it is read for structure, not as a score.
    """
    names = P.exp_names(data)
    have = [n for n in names
            if P.get(data['state'], n, 'ocean', 'woa_bias_mean', default=None)]
    if not have:
        return None
    col = P.color_map(names)
    vars3d = [v for v in cfg.get('background_vars', {}).get('ocean', [])
              if any(_is3d(P.get(data['state'], n, 'ocean', 'woa_bias_mean', v,
                                 default=None)) for n in have)]
    if not vars3d:
        return None
    fig, axes = plt.subplots(1, 2 * len(vars3d),
                             figsize=(3.5 * 2 * len(vars3d), 4.8),
                             squeeze=False)
    axes = axes[0]
    for i, v in enumerate(vars3d):
        axL, axR = axes[2 * i], axes[2 * i + 1]
        for n in have:
            p = P.get(data['state'], n, 'ocean', 'bkg_mean', v, default=None)
            if _is3d(p):
                p = np.asarray(p, dtype='f8')
                axL.plot(p, depth_axis(data, grid, len(p))[0], color=col[n],
                         label=n)
            b = P.get(data['state'], n, 'ocean', 'woa_bias_mean', v,
                      default=None)
            if _is3d(b):
                b = np.asarray(b, dtype='f8')
                axR.plot(b, depth_axis(data, grid, len(b))[0], color=col[n],
                         label=n)
        w = woa_curve(data, 'woa_region', 'global', v)
        if w is not None:
            w = np.asarray(w, dtype='f8')
            axL.plot(w, depth_axis(data, grid, len(w))[0], label=WOA_NAME,
                     **WOA_STYLE)
        axR.axvline(0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)))
        for ax in (axL, axR):
            ax.invert_yaxis()
            P.depth_limit(ax, cfg)
            P.tidy(ax, xgrid=True)
        axL.set_xlabel('mean %s' % v)
        axL.set_title('%s, background and climatology' % v)
        axR.set_xlabel('background $-$ WOA23 (%s)' % v)
        axR.set_title('%s departure from WOA23' % v)
        if i == 0:
            axL.set_ylabel('depth (m)')
        P.maybe_legend(axL, fontsize=8)
        P.maybe_legend(axR, fontsize=8)
    fig.suptitle('Background against the WOA23 climatology - %s   '
                 '(a climatology, not an analysis: the difference carries the '
                 'real anomaly too)' % data['cycle'],
                 y=1.02, fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return _save(fig, cfg, 'woa_bias_profiles.png')


def fig_woa_bias_maps(data, cfg, grid, maps):
    """Background minus WOA23 at each mapped level.

    Rows are field and level, columns experiments -- the shape fig_maps uses
    for the increment, and for the same reason: a signed difference wants a
    symmetric diverging scale, and at global scale this is the only place a
    few tenths of a degree is legible against a 0-30 degC state.
    """
    if maps is None:
        return None
    names = P.exp_names(data)
    rows = _rows_for(maps, names, '/ocean/woa_bias/')
    keep = [i for i, _n in enumerate(names)
            if any(f[i] is not None for _k, f in rows)]
    names = [names[i] for i in keep]
    rows = [(k, [f[i] for i in keep]) for k, f in rows]
    if not rows or not names:
        return None
    limits = {}
    for key, fields in rows:
        fixed = _fixed_incr_limits(cfg, 'ocean', key, block='woa_bias')
        if fixed is None:
            lim = _robust(fields, 99.0)
            fixed = (-lim, lim, P.DIVERGING)
        limits[key] = fixed
    out = None
    for view in views_for('ocean'):
        out = map_grid(cfg, grid, rows, names,
                       'Background $-$ WOA23 climatology - ocean %s'
                       % data['cycle'],
                       'woa_bias_maps_ocean.png', view, limits=limits,
                       cb_label='background $-$ WOA23',
                       row_label=lambda k: _map_row_label(data, k),
                       dpi=VERIF_MAP_DPI) or out
    return out


def fig_background_profiles(data, cfg, grid):
    """Mean background T and S against depth, and the difference between runs.

    Surface maps hide systematic drift at depth; this is where it shows.
    """
    names = P.exp_names(data)
    have = [n for n in names
            if P.get(data['state'], n, 'ocean', 'bkg_mean', default=None)]
    if not have:
        return None
    col = P.color_map(names)
    ref = data.get('reference') if data.get('reference') in have else have[0]
    vars3d = [v for v in cfg.get('background_vars', {}).get('ocean', [])
              if any(_is3d(P.get(data['state'], n, 'ocean', 'bkg_mean', v,
                                 default=None)) for n in have)]
    if not vars3d:
        return None

    fig, axes = plt.subplots(1, 2 * len(vars3d),
                             figsize=(3.5 * 2 * len(vars3d), 4.8),
                             squeeze=False)
    axes = axes[0]
    for i, v in enumerate(vars3d):
        axL, axR = axes[2 * i], axes[2 * i + 1]
        base = np.asarray(P.get(data['state'], ref, 'ocean', 'bkg_mean', v,
                                default=[]), dtype='f8')
        for n in have:
            p = P.get(data['state'], n, 'ocean', 'bkg_mean', v, default=None)
            if not _is3d(p):
                continue
            p = np.asarray(p, dtype='f8')
            y, dsrc = depth_axis(data, grid, len(p))
            # The two mean states can coincide to plotting precision; dashing
            # the non-reference runs stops that reading as a missing series.
            axL.plot(p, y, color=col[n], label=n,
                     ls='-' if n == ref else (0, (4, 2)),
                     lw=2.4 if n == ref else 1.6)
            if n != ref and base.size == p.size:
                axR.plot(p - base, y, color=col[n],
                         label='%s - %s' % (n, ref))
        # Left panel only: axR already means "difference from the reference
        # experiment", and it cannot also mean "difference from WOA". That
        # comparison has its own figure.
        w = woa_curve(data, 'woa_region', 'global', v)
        if w is not None:
            w = np.asarray(w, dtype='f8')
            axL.plot(w, depth_axis(data, grid, len(w))[0], label=WOA_NAME,
                     **WOA_STYLE)
        axR.axvline(0, color=P.MUTED, lw=1.2, ls=(0, (4, 3)))
        for ax in (axL, axR):
            ax.invert_yaxis()
            P.depth_limit(ax, cfg)
            P.tidy(ax, xgrid=True)
        axL.set_xlabel('mean %s' % v)
        axL.set_title('%s, area-weighted mean' % v)
        axR.set_xlabel('difference in %s' % v)
        axR.set_title('%s difference from %s' % (v, ref))
        if i == 0:
            axL.set_ylabel('depth (m)')
        P.maybe_legend(axL, fontsize=8)
        P.maybe_legend(axR, fontsize=8)
    _, dsrc = depth_axis(data, grid)
    fig.suptitle('Background mean state against depth - %s   (depth from %s)'
                 % (data['cycle'], dsrc), y=1.02, fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return _save(fig, cfg, 'bkg_profiles.png')


# Sea ice only exists near the poles, where a global lat/lon map wastes most of
# its area and distorts what is left, so every ice figure is drawn on a polar
# stereographic projection, one figure per hemisphere.
HEMIS = {
    'nh': ('Arctic', ccrs.NorthPolarStereo(central_longitude=-45),
           (-180, 180, 45, 90), 40.0),
    'sh': ('Antarctic', ccrs.SouthPolarStereo(central_longitude=0),
           (-180, 180, -90, -45), -40.0),
}


def views_for(realm):
    """(suffix, label, projection, extent, lat cut) per figure for a realm."""
    if realm == 'ice':
        return [(h,) + HEMIS[h] for h in ('nh', 'sh')]
    return [('', None, global_proj(), None, None)]


def map_grid(cfg, grid, rows, names, title, fname, view,
             cmap=None, limits=None, note=None, cb_label=None, dpi=None,
             row_label=None):
    """Draw one grid of maps: rows are fields, columns are experiments.

    The single map-figure builder for the suite. ``rows`` is a list of
    (row label, [field per experiment]); ``limits`` maps a row label to
    (vmin, vmax, cmap), defaulting to a robust symmetric range.
    """
    suffix, hemi_label, proj, extent, latcut = view
    stride = int(cfg.get('map_stride', 2))
    lon, lat = grid.lon[::stride, ::stride], grid.lat[::stride, ::stride]

    sl = slice(None)
    if latcut is not None:
        sel = lat > latcut if latcut > 0 else lat < latcut
        jj = np.where(sel.any(axis=1))[0]
        if not jj.size:
            return None
        sl = slice(jj.min(), jj.max() + 1)
    lon, lat = lon[sl], lat[sl]

    rows = [(k, f) for k, f in rows if any(x is not None for x in f)]
    if not rows or not names:
        return None

    polar = latcut is not None
    w, h_ = (3.7, 4.1) if polar else (5.2, 2.6)
    fig, axes = plt.subplots(len(rows), len(names),
                             figsize=(w * len(names), h_ * len(rows)),
                             squeeze=False,
                             subplot_kw=dict(projection=proj))
    titled = set()
    for r, (key, fields) in enumerate(rows):
        fields = [f if f is None else f[sl] for f in fields]
        good = [f for f in fields if f is not None and np.any(np.isfinite(f))]
        if not good:
            for ax in axes[r]:
                ax.set_visible(False)
            continue
        vmin, vmax, cm = (limits or {}).get(
            key, (None, None, cmap or P.DIVERGING))
        if vmin is None:
            lim = _robust(fields, 99.0)
            vmin, vmax = -lim, lim
        handle = None
        # The row label goes on the first column that actually has a panel.
        # It used to be pinned to column 0, which vanishes with its hidden
        # axis when the leading experiment lacks the field -- only the LETKF
        # writes u/v increments, and that left those rows unlabelled.
        first_vis = next(c for c, f in enumerate(fields) if f is not None)
        for c, n in enumerate(names):
            ax = axes[r][c]
            if fields[c] is None:
                ax.set_visible(False)
                continue
            if extent is not None:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                _circular_boundary(ax)
            handle = _map_panel(ax, lon, lat, fields[c], cm, vmin, vmax,
                                polar=polar)
            if note:
                note(ax, key, fields[c])
            # Title the first row where this column actually has a panel: the
            # top row may be a field one experiment lacks (MLD is 3DVar-only),
            # and an untitled column reads as an unlabelled experiment.
            if n not in titled:
                ax.set_title(n, fontsize=10, color=P.INK)
                titled.add(n)
            if c == first_vis:
                # GeoAxes ignores set_ylabel, so row labels are drawn by hand
                ax.text(-0.06 if polar else -0.025, 0.5,
                        (row_label(key) if row_label
                         else key.replace('_k', '\nlevel ')),
                        transform=ax.transAxes,
                        rotation=90, fontsize=9, ha='right', va='center',
                        color=P.INK2)
        if handle is not None:
            cb = fig.colorbar(handle, ax=list(axes[r]),
                              fraction=0.024 if polar else 0.016, pad=0.01)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7.5)
            if cb_label:
                cb.set_label(cb_label, fontsize=8.5)
    if hemi_label:
        title = '%s - %s' % (title, hemi_label)
    fig.suptitle(title, y=1.0 - 0.006 * len(rows), fontsize=11.5, color=P.INK)
    stem, ext = os.path.splitext(fname)
    return _save(fig, cfg, '%s%s%s' % (stem, '_' + suffix if suffix else '',
                                       ext), dpi=dpi or MAP_DPI)


def _rows_for(maps, names, prefix, drop=()):
    """Collect (row key, [field per experiment]) for cache keys under prefix."""
    keys = sorted({k.split(prefix, 1)[1] for k in maps.files if prefix in k},
                  key=lambda s: (s.rsplit('_k', 1)[0], int(s.rsplit('_k', 1)[1])))
    keys = [k for k in keys if k.rsplit('_k', 1)[0] not in drop]
    return [(k, [maps['%s%s%s' % (n, prefix, k)]
                 if '%s%s%s' % (n, prefix, k) in maps.files else None
                 for n in names]) for k in keys]


def fig_background_maps(data, cfg, grid, maps, realm='ocean'):
    """The background state itself: what the increments are correcting."""
    if maps is None:
        return None
    names = P.exp_names(data)
    rows = _rows_for(maps, names, '/%s/bkg/' % realm)
    if not rows:
        return None
    if realm == 'ocean':
        # The climatology leads, so a row reads "what the ocean is on average,
        # then what each system produced" -- the order fig_verif_maps uses for
        # its product. WOA has no analogue for ave_ssh, MLD or speed; those
        # rows get None and map_grid hides just that panel, keeping the row
        # label on the first column that does have one.
        woa = dict(_rows_for(maps, names, '/ocean/woa/'))
        if woa:
            names = [WOA_NAME] + names
            rows = [(k, [_first(woa.get(k))] + list(f)) for k, f in rows]
    # States, not anomalies: one scale per field, shared across experiments so
    # the columns are comparable. cfg['map_limits'] is checked first so the
    # scale is also fixed across cycles and across experiments that never
    # share a figure to pool a percentile from (see _fixed_limits); anything
    # not declared there falls back to a percentile computed here, pooled
    # across every experiment. The signed fields are the exception -- a
    # sequential ramp over a velocity component puts zero at an arbitrary
    # mid-colour and hides the direction, which is the only reason to plot
    # the components apart from the speed.
    out = None
    for view in views_for(realm):
        hemi = view[0] or None
        limits = {}
        for key, fields in rows:
            fixed = _fixed_limits(cfg, realm, key, hemi)
            if fixed is not None:
                limits[key] = fixed
                continue
            good = [f for f in fields if f is not None and np.any(np.isfinite(f))]
            if not good:
                continue
            allv = np.concatenate([f[np.isfinite(f)].ravel() for f in good])
            if key.rsplit('_k', 1)[0] in SIGNED_FIELDS:
                lim = float(np.percentile(np.abs(allv), 99)) or 1.0
                limits[key] = (-lim, lim, P.DIVERGING)
            else:
                limits[key] = (float(np.percentile(allv, 1)),
                               float(np.percentile(allv, 99)), _field_cmap(key))
        out = map_grid(cfg, grid, rows, names,
                       'Background state - %s %s' % (realm, data['cycle']),
                       'bkg_maps_%s.png' % realm, view, limits=limits,
                       row_label=lambda k: _map_row_label(data, k)) or out
    return out


def _zero_note(ax, key, f):
    """An all-zero field is a real result, not missing data."""
    finite = f[np.isfinite(f)]
    if finite.size and np.all(finite == 0):
        ax.text(0.5, 0.5, 'field not updated\n(increment identically zero)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color=P.INK2,
                bbox=dict(boxstyle='round,pad=0.4', fc=P.SURFACE, ec=P.GRID))


def fig_maps(data, cfg, grid, maps, kind='incr', realm='ocean'):
    """Increment (diverging), spread-reduction or inflation (sequential) maps.

    The inflation factor spans 1 to ~40 with a median near 1.5 and a hard floor
    at exactly 1 wherever no observation reached the column, so it is drawn on
    a log2 axis: a linear range wide enough for the tail leaves the bulk of the
    field indistinguishable from no inflation at all.
    """
    if maps is None:
        return None
    names = P.exp_names(data)
    # `h` is a layer-thickness increment, which is a masking quantity rather
    # than a field worth a map. `u`/`v` are shown when the config asks for
    # them; only the LETKF writes them, and the empty-column filter below
    # drops any experiment that has nothing for the row.
    rows = _rows_for(maps, names, '/%s/%s/' % (realm, kind), drop=('h',))
    # Drop experiments with nothing to show here (a 3DVar has no ensemble
    # spread), so the figure does not carry an empty column.
    keep = [i for i, _n in enumerate(names)
            if any(f[i] is not None for _k, f in rows)]
    names = [names[i] for i in keep]
    rows = [(k, [f[i] for i in keep]) for k, f in rows]
    if not rows or not names:
        return None
    cb_label = None
    if kind == 'infl':
        # log2 of the multiplier: 0 is no inflation, 1 is a doubling of the
        # spread. The upper end is a robust percentile so the long tail cannot
        # flatten the rest of the field.
        rows = [(k, [None if f is None else np.log2(np.maximum(f, 1e-6))
                     for f in fields]) for k, fields in rows]
        ub = max(1.0, _robust([f for _k, fs in rows for f in fs if f is not None],
                              98.0))
        limits = {k: (0.0, ub, P.SEQUENTIAL) for k, _ in rows}
        # keep this short: it sets vertically beside the colourbar ticks
        cb_label = r'$\log_2(\sigma_{an}/\sigma_a)$'
    elif kind != 'incr':
        limits = {k: (0.0, 1.0, P.SEQUENTIAL) for k, _ in rows}
    else:
        limits = None
    title = {'incr': 'Analysis increment (blue negative / orange positive)',
             'sprred': r'Ensemble spread reduction  $1-\sigma_a/\sigma_b$',
             'infl': r'Applied inflation  $\sigma_{an}/\sigma_a$  '
                     r'($\log_2$: 0 = none, 1 = 2x, 2 = 4x)'}[kind]
    fname = 'state_maps_%s_%s.png' % (
        realm, {'incr': 'increment', 'sprred': 'spread_reduction',
                'infl': 'inflation'}[kind])
    out = None
    for view in views_for(realm):
        row_limits = limits
        if kind == 'incr':
            # cfg['map_limits']['increment'] first, same reasoning as the
            # background maps: a fixed scale holds across cycles and across
            # experiments that never share a figure. A row without a fixed
            # entry keeps the per-row percentile map_grid computes itself,
            # already pooled across experiments and sliced to this hemisphere.
            row_limits = {}
            for key, _fields in rows:
                fixed = _fixed_incr_limits(cfg, realm, key, view[0] or None)
                if fixed is not None:
                    row_limits[key] = fixed
        out = map_grid(cfg, grid, rows, names,
                       '%s - %s %s' % (title, realm, data['cycle']),
                       fname, view, limits=row_limits, cb_label=cb_label,
                       note=_zero_note if kind == 'incr' else None,
                       row_label=lambda k: _map_row_label(data, k)) or out
    return out


# --------------------------------------------------------------------------
# vertical sections
# --------------------------------------------------------------------------

def _line_label(key):
    """'lat+030' -> '30N'; 'lon-140' -> '140W'; 'lat+000' -> 'equator'."""
    axis, val = key[:3], int(key[3:])
    if axis == 'lat':
        return 'equator' if val == 0 else '%d%s' % (abs(val),
                                                    'N' if val > 0 else 'S')
    if val == 0:
        return 'prime meridian'
    return '%d%s' % (abs(val), 'E' if val > 0 else 'W')


def _lon_fmt(v, _pos):
    """Tick label for the raw gridspec longitude, wrapped into -180..180.

    The gridspec longitude runs -300..60, which is monotonic along i and so
    makes a continuous x-axis with no seam -- but '-260' is not a longitude
    anyone reads. Positions stay raw; only the labels are wrapped.
    """
    w = ((v + 180.0) % 360.0) - 180.0
    if abs(w) < 0.5 or abs(abs(w) - 180.0) < 0.5:
        return '%.0f' % abs(w)
    return '%.0f%s' % (abs(w), 'E' if w > 0 else 'W')


def _lat_fmt(v, _pos):
    return '0' if abs(v) < 0.5 else '%.0f%s' % (abs(v), 'N' if v > 0 else 'S')


def _tripolar_note(ax, key, _f):
    """Mark a panel whose transect is north of the tripolar seam."""
    if key[:3] != 'lat' or int(key[3:]) <= TRIPOLAR_LAT:
        return
    ax.text(0.5, 0.04,
            'not a latitude circle: grid rows bend north of %d°N'
            % TRIPOLAR_LAT,
            transform=ax.transAxes, ha='center', va='bottom', fontsize=7.5,
            color=P.ALERT, zorder=6,
            bbox=dict(boxstyle='round,pad=0.3', fc=P.SURFACE, ec=P.ALERT,
                      lw=0.8))


def section_depth_views(cfg):
    """One (cut, filename suffix, label) per section figure set to render.

    A section is the one figure where the global `depth_max:` is the wrong
    instrument: cutting at 1000 m shows the thermocline and the bulk of the
    update, but a LETKF increment genuinely reaches the sea floor in places,
    and cropping would hide precisely the thing worth seeing. So the depth
    axis is a per-figure setting here -- `sections: depth_views:` lists the
    views to emit, and each variable gets one figure per view.

    `full` (or no entry at all) is the uncut column. Defaults to full only,
    so a config predating this key renders exactly what it did before.
    """
    views = (cfg.get('sections') or {}).get('depth_views') or ['full']
    out = []
    for v in views:
        if v is None or (isinstance(v, str)
                         and v.strip().lower() in ('full', 'all', 'bottom')):
            out.append((None, '', 'full column'))
        else:
            d = float(v)
            out.append((d, '_0-%gm' % d, 'top %g m' % d))
    return out


def _section_axes(maps, names):
    """{line key: (x, depth)} from whichever experiment cached them."""
    out = {}
    for n in names:
        pre = '%s/ocean/sec/' % n
        for k in maps.files:
            if not k.startswith(pre):
                continue
            what, _, tag = k[len(pre):].partition('_')
            x, d = out.get(tag, (None, None))
            if what == 'x' and x is None:
                out[tag] = (maps[k], d)
            elif what == 'depth' and d is None:
                out[tag] = (x, maps[k])
    return out


def _section_rows(maps, names, prefix):
    """{var: [(line key, [plane per experiment])]} for cached section keys.

    Lines run zonal south-to-north, then meridional west-to-east.
    """
    tags = {}
    for k in maps.files:
        if prefix not in k:
            continue
        var, _, tag = k.split(prefix, 1)[1].rpartition('_')
        if var and tag[:3] in ('lat', 'lon'):
            tags.setdefault(var, set()).add(tag)
    return {var: [(t, [maps['%s%s%s_%s' % (n, prefix, var, t)]
                       if '%s%s%s_%s' % (n, prefix, var, t) in maps.files
                       else None for n in names])
                  for t in sorted(ts, key=lambda s: (s[:3] == 'lon',
                                                     int(s[3:])))]
            for var, ts in tags.items()}


def _section_limits(cfg, kind, var, rows):
    """(vmin, vmax, cmap): ONE scale across every transect of a variable.

    Deliberately not per-row the way map_grid's limits are. A section figure
    is a single field cut in several places, and the whole point is to compare
    those places -- rescaling each line to its own range would make a quiet
    transect look as active as a busy one.

    cfg['map_limits']['sections'] first, for the same reason the maps take a
    fixed scale: it holds across cycles and across experiments that never
    share a figure. Otherwise a percentile pooled over every line and every
    experiment in this figure.
    """
    sec = (cfg.get('map_limits') or {}).get('sections') or {}
    vals = [f for _k, fs in rows for f in fs
            if f is not None and np.any(np.isfinite(f))]
    if kind != 'bkg':
        # A signed field -- an increment, or the departure from the
        # climatology -- takes a symmetric diverging scale about zero. They
        # get separate config blocks because they sit on different sizes: an
        # increment is one analysis step, a climatology bias is a whole
        # model-minus-ocean departure and is typically several times larger.
        block = {'incr': 'increment', 'woa_bias': 'woa_bias'}[kind]
        entry = ((sec.get(block) or {}).get('ocean') or {}).get(var)
        lim = float(entry) if entry is not None else _robust(vals, 99.0)
        return -lim, lim, P.DIVERGING
    entry = (sec.get('ocean') or {}).get(var)
    if entry is not None:
        lo, hi = float(entry[0]), float(entry[1])
    elif vals:
        allv = np.concatenate([f[np.isfinite(f)].ravel() for f in vals])
        lo = float(np.percentile(allv, 1))
        hi = float(np.percentile(allv, 99))
    else:
        lo, hi = 0.0, 1.0
    return lo, hi, _field_cmap(var)


def _section_panel(ax, X, Y, f, cmap, vmin, vmax):
    fm = np.ma.masked_invalid(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        h = ax.pcolormesh(X, Y, fm, cmap=cmap, vmin=vmin, vmax=vmax,
                          shading='nearest', rasterized=True, zorder=1)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    P.tidy(ax, xgrid=True)
    return h


def section_grid(cfg, rows, names, axes, title, fname, limits,
                 cb_label=None, dpi=None, note=None, depth_cut=None):
    """One grid of vertical sections: rows are transects, columns experiments.

    ``axes`` maps a line key to (x, depth), the shared coordinates written by
    lv_statespace.section_geometry. A line with no cached depth (no
    background, so no honest layer thickness) falls back to the model level
    index rather than a nominal depth in metres -- the same choice
    depth_axis() makes.

    ``limits`` is a single (vmin, vmax, cmap) for the whole figure, so there
    is one colourbar rather than a row of identical ones. See
    _section_limits() for why the scale is shared.
    """
    rows = [(k, f) for k, f in rows if any(x is not None for x in f)]
    if not rows or not names:
        return None
    vmin, vmax, cm = limits
    fig, axs = plt.subplots(len(rows), len(names),
                            figsize=(5.4 * len(names), 2.6 * len(rows)),
                            squeeze=False)
    titled, handle = set(), None
    for r, (key, fields) in enumerate(rows):
        good = [f for f in fields if f is not None and np.any(np.isfinite(f))]
        if not good:
            for ax in axs[r]:
                ax.set_visible(False)
            continue

        nk, nx = good[0].shape
        x, depth = axes.get(key, (None, None))
        # pcolormesh refuses a non-finite coordinate outright, so an axis that
        # is the wrong shape or carries a NaN falls back to the plain index
        # rather than taking the whole figure down. section_geometry() keeps
        # depth finite below the sea floor precisely so this stays unused.
        if x is None or np.size(x) != nx or not np.all(np.isfinite(x)):
            x = np.arange(nx, dtype='f8')
        X = np.broadcast_to(np.asarray(x, dtype='f8'), (nk, nx))
        deep = (depth is not None and np.shape(depth) == (nk, nx)
                and bool(np.all(np.isfinite(depth))))
        Y = (np.asarray(depth, dtype='f8') if deep else
             np.broadcast_to(np.arange(nk, dtype='f8')[:, None], (nk, nx)))

        first_vis = next(c for c, f in enumerate(fields) if f is not None)
        for c, n in enumerate(names):
            ax = axs[r][c]
            if fields[c] is None:
                ax.set_visible(False)
                continue
            handle = _section_panel(ax, X, Y, fields[c], cm, vmin, vmax)
            if deep:
                # The per-figure cut wins over the global `depth_max:`; both
                # are set on every column from one row-level value, so the
                # experiments stay directly comparable either way.
                if depth_cut:
                    ax.set_ylim(float(depth_cut), 0.0)
                else:
                    P.depth_limit(ax, cfg)
            ax.xaxis.set_major_formatter(
                FuncFormatter(_lon_fmt if key[:3] == 'lat' else _lat_fmt))
            if r == len(rows) - 1:
                ax.set_xlabel('longitude' if key[:3] == 'lat' else 'latitude')
            if note:
                note(ax, key, fields[c])
            if n not in titled:
                ax.set_title(n, fontsize=10, color=P.INK)
                titled.add(n)
            if c == first_vis:
                ax.set_ylabel('%s\n%s' % (_line_label(key),
                                          'depth (m)' if deep
                                          else 'model level'))
            else:
                ax.set_yticklabels([])
    if handle is not None:
        cb = fig.colorbar(handle, ax=axs.ravel().tolist(),
                          fraction=0.018, pad=0.015, aspect=45)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5)
        if cb_label:
            cb.set_label(cb_label, fontsize=8.5)
    fig.suptitle(title, y=1.0 - 0.004 * len(rows), fontsize=11.5, color=P.INK)
    return _save(fig, cfg, fname, dpi=dpi or MAP_DPI)


def fig_sections(data, cfg, grid, maps, kind='incr'):
    """Vertical sections of the increment or the background, one PNG per field.

    A horizontal map at a handful of levels cannot show how deep an update
    reaches, or whether it follows the thermocline; a transect can. One file
    per variable rather than one grid carrying every variable and every
    transect -- with a few lines configured that grid squeezes each panel into
    an unreadable sliver, the same reason fig_regional_profiles writes one
    file per region. build_report.py picks them up behind a field picker.
    """
    if maps is None:
        return []
    names = P.exp_names(data)
    prefix = '/ocean/%s_sec/' % kind
    by_var = _section_rows(maps, names, prefix)
    if not by_var:
        return []
    axes = _section_axes(maps, names)
    order = (cfg['state_vars']['ocean'] if kind == 'incr'
             else cfg.get('background_vars', {}).get('ocean', []))
    order = ([v for v in order if v in by_var]
             + [v for v in sorted(by_var) if v not in order])
    what = {'incr': 'Analysis increment', 'bkg': 'Background state',
            'woa_bias': 'Background $-$ WOA23'}[kind]
    woa = _section_rows(maps, names, '/ocean/woa_sec/') if kind == 'bkg' else {}
    views = section_depth_views(cfg)
    written = []
    for var in order:
        rows = by_var[var]
        # Drop experiments with nothing on any line, so the figure carries no
        # empty column (only the LETKF writes u/v increments).
        keep = [i for i, _n in enumerate(names)
                if any(f[i] is not None for _k, f in rows)]
        if not keep:
            continue
        kept = [(k, [f[i] for i in keep]) for k, f in rows]
        cols = [names[i] for i in keep]
        # One scale across every view as well as every line: the shallow cut
        # is a zoom on the same field, so recolouring it would break the
        # comparison with the full-column figure beside it.
        pool = rows
        wsec = woa.get(var)
        if wsec:
            by_line = {t: _first(f) for t, f in wsec}
            kept = [(t, [by_line.get(t)] + f) for t, f in kept]
            cols = [WOA_NAME] + cols
            pool = rows + wsec      # or the climatology would be recoloured
        limits = _section_limits(cfg, kind, var, pool)
        for cut, suffix, label in views:
            p = section_grid(
                cfg, kept, cols, axes,
                '%s along vertical sections - %s (%s) - %s'
                % (what, var, label, data['cycle']),
                'state_sections_%s_%s%s.png' % (kind, var, suffix),
                limits=limits, cb_label=var, note=_tripolar_note,
                depth_cut=cut)
            if p:
                written.append(p)
    return written


def _verif_products(data, names):
    """Products present in the cache, in the order lv_verif declares them."""
    have = set()
    for n in names:
        have |= set(P.get(data['state'], n, 'ocean', 'verif', default={}))
    return [p for p in LV.PRODUCTS if p in have]


def fig_verif_maps(data, cfg, grid, maps, prod):
    """model - product for one field, rows background/analysis."""
    if maps is None:
        return None
    exps = P.exp_names(data)
    found = dict(_rows_for(maps, exps, '/ocean/verif/'))
    spec = LV.PRODUCTS[prod]
    obs = found.get('%s_obs_k0' % prod)
    if obs is None:
        return None
    # Columns are the product and then each experiment, so a row reads as
    # "reference beside what each system produced". The product is the same
    # field in both rows; repeating it keeps each row self-contained.
    names = [spec['label']] + list(exps)
    rows = []
    for st, lbl in (('bkg', 'background'), ('ana', 'analysis')):
        key = '%s_%s_k0' % (prod, st)
        if key not in found:
            continue
        # `obs` is a per-experiment list; any column that has it will do, since
        # every experiment cached the same product field.
        ref = next((f for f in obs if f is not None), None)
        rows.append((lbl, [ref] + list(found[key])))
    if not rows:
        return None
    # One scale across every panel: these are states, so the point is whether
    # the model looks like the product, which a per-panel scale would hide.
    # ADT has had its mean removed, making it an anomaly about zero, so it
    # wants a symmetric diverging scale; the others are absolute states, and
    # use the same cfg['map_limits'] entry as the background-state map of
    # their model_var (e.g. sst -> Temp_k0) so this figure and that one read
    # on an identical scale.
    vals = [f for _k, fs in rows for f in fs if f is not None]
    if spec.get('remove_mean'):
        lim = _robust(vals, 99.0)
        lo, hi, cm = -lim, lim, P.DIVERGING
    else:
        fixed = _fixed_limits(cfg, 'ocean', '%s_k0' % spec['model_var'])
        if fixed is not None:
            lo, hi, cm = fixed
        else:
            allv = np.concatenate([f[np.isfinite(f)].ravel() for f in vals])
            lo, hi, cm = (float(np.percentile(allv, 1)),
                          float(np.percentile(allv, 99)),
                          _field_cmap('%s_k0' % spec['model_var']))
    limits = {k: (lo, hi, cm) for k, _ in rows}
    note = ('  (mean removed)' if spec.get('remove_mean') else '')
    out = None
    for view in views_for('ocean'):
        out = map_grid(cfg, grid, rows, names,
                       '%s and the model state%s - %s'
                       % (spec['label'], note, data['cycle']),
                       'verif_maps_%s.png' % prod, view, limits=limits,
                       cb_label='%s (%s)' % (prod.upper(), spec['units']),
                       dpi=VERIF_MAP_DPI) or out
    return out


def fig_verif_diffs(data, cfg, grid, maps, prod):
    """model - product for one field, rows background/analysis.

    The companion to the field maps: at global scale a 0.45 degC error is
    invisible against a 0..30 degC ramp, so this is where the error actually
    reads. Rows share one symmetric scale, or the analysis would be rescaled
    to look like the background.
    """
    if maps is None:
        return None
    names = P.exp_names(data)
    found = dict(_rows_for(maps, names, '/ocean/verif/'))
    rows = [(lbl, found['%s_%s_diff_k0' % (prod, st)])
            for st, lbl in (('bkg', 'background'), ('ana', 'analysis'))
            if '%s_%s_diff_k0' % (prod, st) in found]
    if not rows:
        return None
    spec = LV.PRODUCTS[prod]
    lim = _robust([f for _k, fs in rows for f in fs if f is not None], 99.0)
    limits = {k: (-lim, lim, P.DIVERGING) for k, _ in rows}
    note = ('  (mean removed from both)' if spec.get('remove_mean') else '')
    out = None
    for view in views_for('ocean'):
        out = map_grid(cfg, grid, rows, names,
                       'Model minus %s%s - %s'
                       % (spec['label'], note, data['cycle']),
                       'verif_diff_%s.png' % prod, view, limits=limits,
                       cb_label='model $-$ product (%s)' % spec['units'],
                       dpi=VERIF_MAP_DPI) or out
    return out


def fig_ocean_regions(cfg, grid):
    """Map of the configured ocean-basin mask, plus any `regions:` boxes.

    Static reference figure -- not tied to a cycle or experiment, so this is
    called once, not from render_cycle(). Answers "what does 'Atlantic'/
    'Pacific'/etc actually cover" for every other figure and table that
    breaks a score down by region.
    """
    mask_path = cfg.get('ocean_basin_mask')
    if not mask_path:
        return None
    from matplotlib.colors import BoundaryNorm, ListedColormap

    basins = basin_regions(mask_path)          # [(code, name), ...]
    codes = basin_at(mask_path, grid.lat, grid.lon180)
    stride = int(cfg.get('map_stride', 2))
    lon, lat = grid.lon[::stride, ::stride], grid.lat[::stride, ::stride]
    field = np.where(codes > 0, codes, np.nan).astype('f8')[::stride, ::stride]

    colors = [P.SERIES[i % len(P.SERIES)] for i in range(len(basins))]
    cmap = ListedColormap(colors)
    edges = [c - 0.5 for c, _ in basins] + [basins[-1][0] + 0.5]
    norm = BoundaryNorm(edges, cmap.N)

    fig, ax = plt.subplots(figsize=(9.5, 4.8),
                           subplot_kw=dict(projection=global_proj()))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), cmap=cmap,
                     norm=norm, shading='nearest', rasterized=True,
                     transform=ccrs.PlateCarree(), zorder=1)
    _decorate(ax)
    ax.set_global()

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    ax.legend(handles, [name for _c, name in basins], loc='upper center',
             bbox_to_anchor=(0.5, -0.05), ncol=len(basins), fontsize=8.5,
             frameon=False)

    # `regions:` boxes (corr_regions and any other named box) drawn as
    # outlines on top, so the figure documents both region mechanisms at
    # once -- a box may straddle the dateline, which is drawn as two pieces.
    for r in cfg.get('regions', []):
        lat0, lat1, lon0, lon1 = region_box(r)
        if lon0 is None:
            continue
        spans = ([(lon0, lon1)] if lon0 <= lon1
                 else [(lon0, 180.0), (-180.0, lon1)])
        for lo, hi in spans:
            ax.plot([lo, hi, hi, lo, lo], [lat0, lat0, lat1, lat1, lat0],
                   transform=ccrs.PlateCarree(), color=P.INK,
                   linewidth=1.1, zorder=5)
        lo0, _hi0 = spans[0]
        ax.text(lo0, lat1, r['name'], transform=ccrs.PlateCarree(),
               fontsize=7.5, ha='left', va='bottom', color=P.INK, zorder=6)

    ax.set_title('Ocean basins (RECCAP2 open_ocean) and named sub-boxes',
                 fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return _save(fig, cfg, 'ocean_regions.png')


def fig_corr_lengths(data, cfg, grid):
    """Increment correlation length -- the footprint the covariance produced."""
    names = P.exp_names(data)
    col = P.color_map(names)
    boxes = list(cfg.get('corr_regions') or [])
    if not boxes:
        return None
    keys = sorted({k for n in names
                   for k in P.get(data['state'], n, 'ocean', 'corr_length',
                                  default={})})
    keys = [k for k in keys if k.startswith(('Temp', 'Salt'))]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 4.0),
                             squeeze=False)
    axes = axes[0]
    x = np.arange(len(boxes))
    w = 0.8 / max(len(names), 1)
    for c, key in enumerate(keys):
        ax = axes[c]
        for j, n in enumerate(names):
            cl = P.get(data['state'], n, 'ocean', 'corr_length', key, default={})
            v = np.array([cl.get(b, np.nan) for b in boxes], dtype='f8')
            ax.bar(x - 0.4 + w * (j + 0.5), v, w * 0.86, color=col[n], label=n,
                   edgecolor=P.SURFACE, linewidth=1.0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(boxes, rotation=30, ha='right')
        ax.set_title(_map_row_label(data, key).replace('\n', '  '),
                     fontsize=10)
        if c == 0:
            ax.set_ylabel('increment 1/e correlation length (km)')
        P.tidy(ax)
    P.maybe_legend(axes[0], fontsize=8.5)
    fig.suptitle('Horizontal structure of the increment - %s' % data['cycle'],
                 y=1.02, fontsize=11.5, color=P.INK)
    fig.tight_layout()
    return _save(fig, cfg, 'state_correlation_lengths.png')


def fig_map_sequence(cycles, cfg, grid, field, realm='ocean', kind='incr',
                     max_rows=8):
    """One field across dates: rows are cycles, columns are experiments.

    This is the view that shows whether the increment pattern is stable from
    cycle to cycle or wandering, which a single date cannot say. Long runs are
    subsampled evenly to keep the figure readable.
    """
    order = sorted(cycles)
    if len(order) > max_rows:
        pick = np.linspace(0, len(order) - 1, max_rows).round().astype(int)
        order = [order[i] for i in sorted(set(pick))]
    names = P.exp_names(cycles[order[-1]])
    key = '%s/%s/%s' % (realm, kind, field)

    rows, stride = [], None
    for c in order:
        m = P.load_maps(cfg, c)
        if m is None:
            continue
        st = _stride(cycles[c], cfg)
        if stride is None:
            stride = st
        elif st != stride:
            print('  ! %s cached with map_stride %d, expected %d -- skipped'
                  % (c, st, stride))
            continue
        fields = [m['%s/%s' % (n, key)] if '%s/%s' % (n, key) in m.files
                  else None for n in names]
        if any(f is not None for f in fields):
            rows.append((c, fields))
    if stride is None or len(rows) < 2:
        return None

    limits = None
    if kind != 'incr':
        limits = {c: (0.0, 1.0, P.SEQUENTIAL) for c, _ in rows}
    what = ('increment' if kind == 'incr'
            else r'spread reduction $1-\sigma_a/\sigma_b$')
    title = ('%s %s across dates - %s (%d of %d cycles shown)'
             % (_map_row_label(cycles[order[-1]], field).replace('\n', ' '),
                what, realm,
                len(rows), len(cycles)))
    out = None
    for view in views_for(realm):
        out = map_grid(cfg, grid, rows, names, title,
                       'seq_%s_%s_%s.png' % (realm, kind, field), view,
                       limits=limits) or out
    return out


def render_cycle(cycle, cycles, cfg, grid, index=None, total=None):
    """Render one date's state figures, reporting progress as it goes.

    Each map figure takes seconds, so a long run needs to say where it is
    rather than sitting silent.
    """
    global TAG
    t0 = time.time()
    data = cycles[cycle]
    maps = P.load_maps(cfg, cycle)
    TAG = '_%s' % cycle if len(cycles) > 1 else ''
    where = ('[%d/%d] ' % (index, total)) if total else ''
    print('%s%s: state-space figures' % (where, cycle), flush=True)

    steps = [('increment profiles', lambda: fig_increment_profiles(data, cfg, grid)),
             ('spread profiles', lambda: fig_spread_profiles(data, cfg, grid))]
    for realm in ('ocean', 'ice'):
        steps.append(('%s increment maps' % realm,
                      lambda r=realm: fig_maps(data, cfg, grid, maps, 'incr', r)))
        steps.append(('%s spread-reduction maps' % realm,
                      lambda r=realm: fig_maps(data, cfg, grid, maps, 'sprred', r)))
        steps.append(('%s inflation maps' % realm,
                      lambda r=realm: fig_maps(data, cfg, grid, maps, 'infl', r)))
    steps.append(('increment sections',
                  lambda: fig_sections(data, cfg, grid, maps, 'incr')))
    steps.append(('background profiles',
                  lambda: fig_background_profiles(data, cfg, grid)))
    steps.append(('increment profiles by region',
                  lambda: fig_regional_profiles(
                      data, cfg, grid, 'incr_region', 'RMS increment',
                      'state_increment_regions.png', floor=0.0)))
    steps.append(('spread profiles by region',
                  lambda: fig_spread_regions(data, cfg, grid)))
    steps.append(('background profiles by region',
                  lambda: fig_regional_profiles(
                      data, cfg, grid, 'bkg_region', 'background mean',
                      'bkg_profiles_regions.png',
                      varlist=cfg.get('background_vars', {}).get('ocean'))))
    for realm in ('ocean', 'ice'):
        steps.append(('%s background maps' % realm,
                      lambda r=realm: fig_background_maps(data, cfg, grid,
                                                          maps, r)))
    steps.append(('background sections',
                  lambda: fig_sections(data, cfg, grid, maps, 'bkg')))
    steps.append(('WOA bias profiles',
                  lambda: fig_woa_bias_profiles(data, cfg, grid)))
    steps.append(('WOA bias profiles by region',
                  lambda: fig_regional_profiles(
                      data, cfg, grid, 'woa_bias_region',
                      'background $-$ WOA23', 'woa_bias_regions.png',
                      varlist=list(lv_woa.VARS))))
    steps.append(('WOA bias maps',
                  lambda: fig_woa_bias_maps(data, cfg, grid, maps)))
    steps.append(('WOA bias sections',
                  lambda: fig_sections(data, cfg, grid, maps, 'woa_bias')))
    for prod in _verif_products(data, P.exp_names(data)):
        steps.append(('%s verification maps' % prod,
                      lambda p=prod: fig_verif_maps(data, cfg, grid, maps, p)))
        steps.append(('%s verification differences' % prod,
                      lambda p=prod: fig_verif_diffs(data, cfg, grid, maps, p)))
    steps.append(('correlation lengths',
                  lambda: fig_corr_lengths(data, cfg, grid)))

    written = 0
    for label, fn in steps:
        t = time.time()
        print('    %-28s' % label, end='', flush=True)
        result = fn()
        # fig_regional_profiles (one PNG per region) returns a list; every
        # other step here still returns a single path or None.
        paths = result if isinstance(result, list) else ([result] if result else [])
        if paths:
            written += len(paths)
            tail = (os.path.basename(paths[0]) if len(paths) == 1 else
                   '%d files (%s...)' % (len(paths), os.path.basename(paths[0])))
            print(' %-46s %5.1fs' % (tail, time.time() - t), flush=True)
        else:
            print(' %-46s %5.1fs' % ('(nothing to plot)', time.time() - t),
                  flush=True)
    print('  %s: %d figures in %.1fs' % (cycle, written, time.time() - t0),
          flush=True)


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
                    help='render only this cycle (repeatable)')
    ap.add_argument('--latest', action='store_true',
                    help='render only the most recent cached cycle')
    ap.add_argument('--no-per-cycle', action='store_true',
                    help='skip the per-date figures, write only the '
                         'across-date sequences')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt
    t_all = time.time()
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    cycles = P.load_cycles(cfg, a.cycle)
    print('loading grid %s' % os.path.basename(cfg['grid']), flush=True)
    grid = Grid(cfg['grid'])
    print('figures -> %s' % cfg['figs'], flush=True)
    # Printed here rather than from render_cycle(), which writes one
    # partial line per figure and would be garbled by an interleaved warning.
    for w in section_warnings(cfg):
        print('  ! %s' % w, flush=True)

    # Static reference figure -- not tied to a cycle, so drawn once here
    # rather than from render_cycle().
    fig_ocean_regions(cfg, grid)

    todo = sorted(cycles)
    if a.latest:
        todo = todo[-1:]
    if not a.no_per_cycle:
        print('rendering %d cycle(s)' % len(todo), flush=True)
        for i, cycle in enumerate(todo, 1):
            render_cycle(cycle, cycles if not a.latest else {cycle: cycles[cycle]},
                         cfg, grid, i, len(todo))

    # across-date views, only meaningful with more than one cycle
    if len(cycles) > 1:
        global TAG
        TAG = ''
        print('across-date sequences (%d cycles)' % len(cycles), flush=True)
        levels = cfg.get('map_levels', [0])
        seq = [('ocean', '%s_k%d' % (v, k))
               for v in cfg.get('sequence_fields', ['Temp', 'ave_ssh'])
               for k in ([0] if v == 'ave_ssh' else levels)]
        seq.append(('ice', 'aice_h_k0'))
        for realm, field in seq:
            t = time.time()
            print('    %-28s' % ('%s %s' % (realm, field)), end='', flush=True)
            path = fig_map_sequence(cycles, cfg, grid, field, realm, 'incr')
            print(' %-46s %5.1fs'
                  % (os.path.basename(path) if path else '(nothing to plot)',
                     time.time() - t), flush=True)
    print('done in %.1fs' % (time.time() - t_all), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
