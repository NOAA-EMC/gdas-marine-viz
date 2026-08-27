"""Shared plotting style for the verification suite.

Palette is the validated categorical set: slots assigned to experiments in
fixed registry order and never cycled, so an experiment keeps its colour no
matter which subset is being plotted.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical palette, light mode (worst adjacent CVD dE 9.1,
# worst adjacent normal-vision dE 22.9).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
          '#e87ba4', '#008300', '#4a3aa7', '#e34948']

INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e4e4e1'
SURFACE = '#fcfcfb'

# Better/worse, taken from the validated series palette. Only ever used to
# reinforce a value that already carries its own sign, so the red/green pair
# is redundant encoding rather than the sole channel.
OK = '#1baf7a'
ALERT = '#e34948'

# Diverging: two hues with a neutral grey midpoint (never a hue at the middle).
DIVERGING = LinearSegmentedColormap.from_list(
    'lv_div', ['#12395f', '#2a78d6', '#a9c9ec', '#e8e8e6',
               '#f5b79b', '#eb6834', '#8a3416'])
# Sequential: one hue, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'lv_seq', ['#f2f6fc', '#a9c9ec', '#5b9ae0', '#2a78d6', '#12395f'])

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID,
    'axes.labelcolor': INK2,
    'axes.titlecolor': INK,
    'axes.titlesize': 11,
    'axes.titleweight': 'semibold',
    'axes.labelsize': 9,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
    'font.size': 9.5,
    # Profiles and bar panels are line art, so this trades little for a lot of
    # embedded size: the report grows with the number of fields and regions,
    # and has a hard 16 MB ceiling for publishing.
    'figure.dpi': 112,
})


def color_map(names):
    """Fixed slot per experiment, in registry order."""
    return {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}


def tidy(ax, xgrid=False):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.grid(axis='both' if xgrid else 'y')


def depth_limit(ax, cfg):
    """Cut a depth axis at `depth_max:` metres, surface at the top.

    Purely a view setting, applied at plot time -- the cache always holds the
    full column, so changing it needs no recompute. Most of the signal lives in
    the top few hundred metres, and on a linear axis a full 6000 m column
    squeezes that into a sliver; this trades the abyss for the part being
    tuned.

    Call it AFTER the axis has been inverted: the explicit limits set the
    direction as well, so it is safe either way, but leaving it earlier would
    let a later invert_yaxis() flip it back.
    """
    dmax = (cfg or {}).get('depth_max')
    if dmax:
        ax.set_ylim(float(dmax), 0.0)


def ref_line(ax, y=1.0, label='ideal = 1'):
    ax.axhline(y, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, xy=(0.005, y), xycoords=('axes fraction', 'data'),
                xytext=(0, -11), textcoords='offset points',
                ha='left', va='top', fontsize=8, color=MUTED)


def maybe_legend(ax, **kw):
    """Legend for two or more series; a single series is named by the title."""
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) >= 2:
        ax.legend(**kw)
    return len(labels)


def short(obstype):
    """Compact obs-type label for axis ticks."""
    return (obstype.replace('insitu_', '').replace('_l3u', '')
            .replace('icec_', 'ice ').replace('profile_', '')
            .replace('surface_', '').replace('sst_', ''))


# Per-experiment obs blocks: independent between runs, so they merge exactly.
# 'common' is deliberately absent -- see merge_cycle().
_OBS_PER_EXP = ('own', 'counts', 'qc', 'nens', 'rank', 'spread_skill')

# Inside 'counts' these two describe the join, not the experiment: a run with a
# single experiment registered "matched" all of its own observations. They are
# meaningless once caches from separate runs are merged.
_JOIN_COUNTS = ('n_matched', 'n_common_pass')


def _joined_set(blk):
    """Experiments this obs block actually joined. Absent key != empty set."""
    js = blk.get('common_experiments')
    if js is None:
        return set((blk.get('common') or {}).keys())
    return set(js)


def merge_cycle(base, extra):
    """Merge one cycle's cache from another source into ``base``.

    Per-experiment results are independent between runs and merge exactly. The
    'common' block is the cross-experiment intersection and CANNOT be: an
    experiment computed on its own has common == own, so merging two of them
    would produce a comparison on unlike samples -- the very thing the common
    sample exists to prevent. It is carried over only when the source actually
    joined every experiment now present, and marked invalid otherwise.
    """
    if base is None:
        # Start from the scalars and let the first source go through exactly
        # the same path as the rest, so its provenance is recorded too.
        base = {k: v for k, v in extra.items()
                if k not in ('obs', 'state', 'experiments')}
        base.update(obs={}, state={}, experiments={})
    for name, rec in extra.get('experiments', {}).items():
        base.setdefault('experiments', {}).setdefault(name, rec)
    for name, rec in extra.get('state', {}).items():
        base.setdefault('state', {}).setdefault(name, rec)
    for t, blk in extra.get('obs', {}).items():
        tgt = base.setdefault('obs', {}).setdefault(
            t, {k: v for k, v in blk.items()
                if k not in _OBS_PER_EXP + ('common',)})
        for key in _OBS_PER_EXP:
            for name, rec in (blk.get(key) or {}).items():
                tgt.setdefault(key, {}).setdefault(name, rec)
        # Record which source supplied each experiment's common record, and
        # what that source actually joined. Intersecting the joined sets
        # instead would let a stale single-experiment cache invalidate a
        # properly rejoined block that already superseded it.
        src_joined = sorted(_joined_set(blk))
        cm = tgt.setdefault('common', {})
        src = tgt.setdefault('common_src', {})
        for n, rec in (blk.get('common') or {}).items():
            if n not in cm:
                cm[n] = rec
                src[n] = src_joined
    return base


def mark_common_validity(data):
    """Flag obs types whose 'common' block does not span every experiment.

    Where it does not, the join-derived counts are blanked too: they described
    a join that no longer matches the experiments on show.
    """
    names = set(exp_names(data))
    ok = True
    for blk in data.get('obs', {}).values():
        src = blk.get('common_src')
        if src is None:                     # unmerged: one source, one join
            valid = bool(names and _joined_set(blk) >= names)
        else:                               # every record must come from a
            valid = bool(names) and all(    # join spanning all experiments
                set(src.get(n, ())) >= names for n in names)
        blk['common_valid'] = valid
        if not blk['common_valid']:
            for rec in (blk.get('counts') or {}).values():
                for k in _JOIN_COUNTS:
                    rec[k] = None
        ok = ok and blk['common_valid']
    data['common_valid'] = ok
    return ok


def load_cycles(cfg, cycles=None):
    """Read cached JSON for the requested cycles, merging every cache source."""
    out = {}
    for c in (cycles or cfg['cycles']):
        merged = None
        for root in cfg.get('caches', [cfg['cache']]):
            p = os.path.join(root, '%s.json' % c)
            if os.path.exists(p):
                with open(p) as f:
                    merged = merge_cycle(merged, json.load(f))
        if merged is not None:
            mark_common_validity(merged)
            out[c] = merged
    if not out:
        raise SystemExit('no cached cycles found in %s -- run compute_cycle.py'
                         % ', '.join(cfg.get('caches', [cfg['cache']])))
    return out


class _Maps(dict):
    """Merged NPZ contents with the np.load interface the plots expect."""

    @property
    def files(self):
        return list(self)


def load_maps(cfg, cycle):
    out = _Maps()
    for root in cfg.get('caches', [cfg['cache']]):
        p = os.path.join(root, '%s_maps.npz' % cycle)
        if os.path.exists(p):
            with np.load(p) as z:
                for k in z.files:
                    out.setdefault(k, z[k])
    return out or None


def exp_names(data):
    """Experiment names in registry order, as stored in the cache."""
    return list(data['experiments'].keys())


def save(fig, cfg, name, dpi=None, quiet=False):
    os.makedirs(cfg['figs'], exist_ok=True)
    p = os.path.join(cfg['figs'], name)
    # Map panels are rasterised pcolormesh; a lower dpi cuts the embedded PNG
    # size substantially at no real cost to legibility.
    fig.savefig(p, bbox_inches='tight', **({'dpi': dpi} if dpi else {}))
    plt.close(fig)
    if not quiet:                      # callers that report progress themselves
        print('  wrote %s' % p, flush=True)
    return p


def get(d, *keys, default=np.nan):
    """Nested lookup that tolerates missing branches and JSON nulls."""
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return default if d is None else d
