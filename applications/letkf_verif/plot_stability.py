#!/usr/bin/env python3
"""SSH cycling-stability diagnostics: numbers instead of watching animations.

The failure mode a tight fit to altimetry has produced before is a slowly
growing mode: the analysis inserts a feature the model does not hold, the
forecast pushes it back, the next analysis re-inserts it larger, and after
enough cycles it is a stripe or a blob nobody wanted. Watching an animation
catches it late and only where one happens to look. This computes, for every
cycle and region, the quantities that move first, fits a trend to each, and
writes a verdict:

  incr_rms, incr_max        size of the SSH increment (area-weighted RMS, max |.|)
  chg_rms                   the 6-h forecast change background(t) - analysis(t-6h)
  persist                   spatial correlation of increment(t) with increment(t-6h):
                            ~0 when each increment corrects something new;
                            sustained positive = the same correction every cycle
  reject                    slope of the forecast change on the previous increment:
                            -1 = the model undoes what the analysis put in;
                            0 = increments are kept
  hp_var_bkg, hp_var_incr   variance of the small-scale part (< ~2 deg) of the
                            background and of the increment
  gs_var_bkg                the same at grid scale (< ~3 cells): mesoscale spin-up
                            raises hp_var_bkg but not this; DA noise raises both
  shock                     RMS(f003-f000) / RMS(f006-f003) of the forecast from this
                            cycle's analysis: initialization adjustment vs the
                            model's own tendency
  n_big, big_lon, big_lat   cells with |increment| > --big (m), and where the
                            largest cluster is
  ontrack                   increment variance in 1-deg bins with ADT obs over bins
                            without (from the cycle's obsbins): along-track striping

Per experiment with an ocean background and increment:

    figs/cycle_ssh_stability_<exp>.png   one panel per metric, one line per region,
                                         dotted = fitted trend
    ssh_stability_<exp>.csv              every metric per cycle and region
    ssh_stability_<exp>.json             the verdict: trend in % of the metric mean
                                         per day, t statistic, adverse flag

Regions: global plus the `corr_regions:` boxes (small, fixed, made for this).
The report shows the figure and the verdict table in section 07.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
from netCDF4 import Dataset  # noqa: E402
from scipy import ndimage  # noqa: E402
import lv_plot as P  # noqa: E402
from lv_common import Fresh, Grid, corr_regions, load_config, region_box  # noqa: E402

HP_CELLS = 9          # boxcar width for the small-scale split (~2 deg at 1/4)
METRICS = ['incr_rms', 'incr_max', 'chg_rms', 'persist', 'reject',
           'hp_var_bkg', 'gs_var_bkg', 'hp_var_incr', 'shock', 'n_big',
           'ontrack']
# which direction of trend is bad, for the verdict
ADVERSE = {'incr_rms': +1, 'incr_max': +1, 'chg_rms': +1, 'persist': +1,
           'reject': -1, 'hp_var_bkg': +1, 'gs_var_bkg': +1, 'hp_var_incr': +1,
           'shock': +1, 'n_big': +1, 'ontrack': +1}
LABELS = {'incr_rms': 'SSH increment RMS (m)', 'incr_max': 'max |increment| (m)',
          'chg_rms': '6-h forecast change RMS (m)',
          'persist': 'increment persistence (corr with previous)',
          'reject': 'rejection: forecast change on previous increment',
          'hp_var_bkg': 'small-scale (<2 deg) background SSH variance (m$^2$)',
          'gs_var_bkg': 'grid-scale background SSH variance (m$^2$)',
          'hp_var_incr': 'small-scale increment variance (m$^2$)',
          'shock': 'initialization shock (first 3 h / next 3 h)',
          'n_big': 'cells with |increment| > threshold',
          'ontrack': 'on-track / off-track increment variance'}


def read_level0(path, var):
    if path is None or not os.path.exists(path):
        return None
    with Dataset(path) as ds:
        if var not in ds.variables:
            return None
        v = ds[var]
        a = v[0, 0] if v.ndim == 4 else (v[0] if v.ndim == 3 else v[:])
        return np.ma.filled(a.astype('f8'), np.nan)


def forecast_file(exp, cycle, fhr):
    """This cycle's own forecast history at hour fhr (f000/f003/f006)."""
    d = exp.dir_for(cycle, exp.bkg.get('stem'), 'ocean') if exp.bkg.get(
        'stem') else os.path.join(exp.dir_for(cycle), 'ocean')
    for cand in ('gdas.t%sz.inst.f%03d.nc' % (cycle[8:10], fhr),):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            return p
    return None


def highpass(f, mask, n=HP_CELLS):
    """f minus its boxcar mean over n x n wet cells (NaN-aware)."""
    x = np.where(mask & np.isfinite(f), f, 0.0)
    w = (mask & np.isfinite(f)).astype('f8')
    num = ndimage.uniform_filter(x, n, mode='nearest')
    den = ndimage.uniform_filter(w, n, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        smooth = np.where(den > 0.3, num / den, np.nan)
    return np.where(mask, f - smooth, np.nan)


def wstats(f, w, sel):
    ok = sel & np.isfinite(f)
    if not ok.any():
        return np.nan, np.nan
    ww = w[ok]
    return (float(np.sqrt(np.sum(ww * f[ok] ** 2) / ww.sum())),
            float(np.nanmax(np.abs(f[ok]))))


def wcorr(a, b, w, sel):
    ok = sel & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    ww = w[ok] / w[ok].sum()
    x, y = a[ok] - np.sum(ww * a[ok]), b[ok] - np.sum(ww * b[ok])
    sxx, syy, sxy = np.sum(ww * x * x), np.sum(ww * y * y), np.sum(ww * x * y)
    return float(sxy / np.sqrt(sxx * syy)) if sxx > 0 and syy > 0 else np.nan


def wslope(y, x, w, sel):
    """Slope of y on x (area-weighted least squares)."""
    ok = sel & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    ww = w[ok] / w[ok].sum()
    xm, ym = np.sum(ww * x[ok]), np.sum(ww * y[ok])
    sxx = np.sum(ww * (x[ok] - xm) ** 2)
    return float(np.sum(ww * (x[ok] - xm) * (y[ok] - ym)) / sxx) if sxx > 0 else np.nan


def big_cluster(f, thr, lat, lon, sel):
    """Count of |f| > thr cells and the centroid of the largest cluster."""
    big = sel & np.isfinite(f) & (np.abs(f) > thr)
    n = int(big.sum())
    if n == 0:
        return 0, np.nan, np.nan
    lab, nlab = ndimage.label(big)
    sizes = ndimage.sum(big, lab, index=np.arange(1, nlab + 1))
    k = int(np.argmax(sizes)) + 1
    m = lab == k
    return n, float(np.nanmean(lon[m])), float(np.nanmean(lat[m]))


def ontrack_index(incr, cfg, cycle, grid, sel):
    """Increment variance on 1-deg bins with ADT obs / bins without."""
    try:
        bins = P.load_obsbins(cfg, cycle)
    except Exception:
        bins = None
    if bins is None:
        return np.nan
    count = None
    for k in bins.files:
        if '/map/n' in k and 'rads_adt' in k:
            count = bins[k] if count is None else count + bins[k]
    if count is None:
        return np.nan
    deg = 180.0 / count.shape[0]
    j = np.clip(((grid.lat + 90) / deg).astype(int), 0, count.shape[0] - 1)
    i = np.clip(((grid.lon180 + 180) / deg).astype(int), 0, count.shape[1] - 1)
    on = count[j, i] > 0
    ok = sel & np.isfinite(incr)
    a, b = ok & on, ok & ~on
    if a.sum() < 100 or b.sum() < 100:
        return np.nan
    return float(np.var(incr[a]) / np.var(incr[b]))


def trend(y, hours_per_step=6.0):
    """(slope in % of mean per day, t statistic) of a linear fit vs time."""
    y = np.asarray(y, dtype='f8')
    ok = np.isfinite(y)
    if ok.sum() < 6:
        return np.nan, np.nan
    t = np.arange(len(y))[ok] * hours_per_step / 24.0
    yy = y[ok]
    A = np.column_stack([t, np.ones_like(t)])
    coef, res, _r, _s = np.linalg.lstsq(A, yy, rcond=None)
    dof = len(yy) - 2
    if dof < 1:
        return np.nan, np.nan
    resid = yy - A @ coef
    s2 = np.sum(resid ** 2) / dof
    var_slope = s2 / np.sum((t - t.mean()) ** 2)
    tstat = coef[0] / np.sqrt(var_slope) if var_slope > 0 else np.nan
    mean = np.mean(yy)
    pct = 100.0 * coef[0] / abs(mean) if mean != 0 else np.nan
    return float(pct), float(tstat)


def analyse(cfg, exp, cycles, grid, regions, field='ave_ssh', big=0.3,
            verbose=True):
    """Every metric for every cycle and region; rows of dicts."""
    w = grid.area
    lat, lon = grid.lat, grid.lon180
    rows = []
    prev_incr = prev_ana = None
    for cycle in cycles:
        bkg = read_level0(exp.background(cycle, 'ocean'), field)
        incr = read_level0(exp.increment(cycle, 'ocean'), field)
        if bkg is None or incr is None:
            prev_incr = prev_ana = None
            continue
        ana = bkg + incr
        chg = None if prev_ana is None else bkg - prev_ana
        hp_b = highpass(bkg, grid.mask)
        gs_b = highpass(bkg, grid.mask, 3)
        hp_i = highpass(incr, grid.mask)
        f0 = read_level0(forecast_file(exp, cycle, 0), field)
        f3 = read_level0(forecast_file(exp, cycle, 3), field)
        f6 = read_level0(forecast_file(exp, cycle, 6), field)
        for rname, sel in regions.items():
            r = {'cycle': cycle, 'region': rname}
            r['incr_rms'], r['incr_max'] = wstats(incr, w, sel)
            r['chg_rms'] = wstats(chg, w, sel)[0] if chg is not None else np.nan
            r['persist'] = (wcorr(incr, prev_incr, w, sel)
                            if prev_incr is not None else np.nan)
            r['reject'] = (wslope(chg, prev_incr, w, sel)
                           if chg is not None and prev_incr is not None
                           else np.nan)
            r['hp_var_bkg'] = wstats(hp_b, w, sel)[0] ** 2
            r['gs_var_bkg'] = wstats(gs_b, w, sel)[0] ** 2
            r['hp_var_incr'] = wstats(hp_i, w, sel)[0] ** 2
            if f0 is not None and f3 is not None and f6 is not None:
                s1 = wstats(f3 - f0, w, sel)[0]
                s2 = wstats(f6 - f3, w, sel)[0]
                r['shock'] = s1 / s2 if s2 > 0 else np.nan
            else:
                r['shock'] = np.nan
            r['n_big'], r['big_lon'], r['big_lat'] = big_cluster(
                incr, big, lat, lon, sel)
            r['ontrack'] = ontrack_index(incr, cfg, cycle, grid, sel)
            rows.append(r)
        if verbose:
            g = rows[-len(regions)]
            print('    %s  incr rms %.4f  chg rms %s  persist %s  reject %s  '
                  'hp_var_bkg %.2e  shock %s'
                  % (cycle, g['incr_rms'],
                     '%.4f' % g['chg_rms'] if np.isfinite(g['chg_rms']) else '-',
                     '%+.2f' % g['persist'] if np.isfinite(g['persist']) else '-',
                     '%+.2f' % g['reject'] if np.isfinite(g['reject']) else '-',
                     g['hp_var_bkg'],
                     '%.2f' % g['shock'] if np.isfinite(g['shock']) else '-'),
                  flush=True)
        prev_incr, prev_ana = incr, ana
    return rows


def verdict(rows, regions):
    """Trend per region and metric: % of the mean per day, t, adverse flag."""
    out = []
    for rname in regions:
        for m in METRICS:
            y = np.array([r[m] for r in rows if r['region'] == rname], dtype='f8')
            pct, tstat = trend(y)
            out.append({'region': rname, 'metric': m, 'pct_per_day': pct,
                        't': tstat, 'mean': float(np.nanmean(y)) if np.any(
                            np.isfinite(y)) else np.nan,
                        'flag': bool(np.isfinite(tstat) and abs(tstat) > 3
                                     and np.sign(pct) == ADVERSE[m])})
    return out


def figure(rows, regions, cycles, exp_name, cfg, fname):
    names = list(regions)
    col = P.color_map(names)
    fig, axes = plt.subplots(6, 2, figsize=(12, 15.5), squeeze=False)
    axes = axes.ravel()
    for k, m in enumerate(METRICS):
        ax = axes[k]
        for rname in names:
            y = np.array([r[m] for r in rows if r['region'] == rname], dtype='f8')
            x = np.arange(len(y))
            ax.plot(x, y, '-', lw=1.2, color=col[rname], label=rname)
            ok = np.isfinite(y)
            if ok.sum() >= 6:
                coef = np.polyfit(x[ok], y[ok], 1)
                ax.plot(x, np.polyval(coef, x), ':', lw=1, color=col[rname])
        ax.set_title(LABELS.get(m, m), fontsize=9.5, color=P.INK)
        ax.tick_params(labelsize=7.5)
        P.tidy(ax)
        if m in ('reject', 'persist'):
            ax.axhline(0, color=P.MUTED, lw=0.8, ls=(0, (4, 3)))
        if m in ('shock', 'ontrack'):
            ax.axhline(1, color=P.MUTED, lw=0.8, ls=(0, (4, 3)))
    drawn = sorted({r['cycle'] for r in rows})
    step = max(1, len(drawn) // 8)
    for ax in axes[:len(METRICS)]:
        ax.set_xticks(np.arange(0, len(drawn), step))
        ax.set_xticklabels([c[4:6] + '-' + c[6:8] + ' ' + c[8:10] + 'Z'
                            for c in drawn[::step]], rotation=30, fontsize=7)
    for ax in axes[len(METRICS):]:
        ax.axis('off')
    axes[0].legend(fontsize=8, frameon=False)

    def when(c):
        return '%s-%s-%s %sZ' % (c[:4], c[4:6], c[6:8], c[8:10])
    fig.suptitle('%s: SSH cycling stability, %s to %s (dotted: fitted trend)'
                 % (exp_name, when(drawn[0]), when(drawn[-1])),
                 fontsize=12, color=P.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return P.save(fig, cfg, fname, dpi=110, quiet=True)


def stability_regions(cfg, grid):
    """global + the corr_regions boxes (small, fixed, made for this)."""
    regions = {'global': grid.mask.copy()}
    for r in corr_regions(cfg):
        lat0, lat1, lon0, lon1 = region_box(r)
        sel = grid.mask & (grid.lat >= lat0) & (grid.lat < lat1)
        if lon0 is not None:
            sel &= ((grid.lon180 >= lon0) & (grid.lon180 < lon1) if lon0 <= lon1
                    else (grid.lon180 >= lon0) | (grid.lon180 < lon1))
        regions[r['name']] = sel
    return regions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config', nargs='?', default=None)
    ap.add_argument('--config', dest='config_opt', default=None)
    ap.add_argument('--root', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--cache', action='append', default=None)
    ap.add_argument('--experiment', action='append', default=None,
                    help='restrict to these experiments (default: every one '
                         'with an ocean background and increment)')
    ap.add_argument('--field', default='ave_ssh')
    ap.add_argument('--big', type=float, default=0.3,
                    help='|increment| threshold (m) for the blow-up count')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt
    t_all = time.time()
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    cycles = [str(c) for c in cfg['cycles']]
    grid = Grid(cfg['grid'])
    regions = stability_regions(cfg, grid)
    os.makedirs(cfg['figs'], exist_ok=True)
    fresh = Fresh(cfg, 'stability', force=a.force, script=__file__)
    exps = [e for e in cfg['experiments']
            if not a.experiment or e.name in a.experiment]
    print('SSH stability: %d cycle(s), regions %s' % (len(cycles), list(regions)),
          flush=True)
    for e in exps:
        have = [c for c in cycles if e.background(c, 'ocean')
                and e.increment(c, 'ocean')]
        if len(have) < 3:
            print('  %s: %d cycle(s) with background+increment -- skipped'
                  % (e.name, len(have)), flush=True)
            continue
        slug = P.slug(e.name)
        # The inputs are a hundred files; the last cycle's stand for them
        # all, with the cycle list as a parameter, so a new cycle or a
        # regenerated last cycle redraws and nothing else does.
        inputs = [e.background(have[-1], 'ocean'), e.increment(have[-1], 'ocean'),
                  __file__]
        params = {'cycles': have, 'field': a.field, 'big': a.big,
                  'regions': sorted(regions)}
        if fresh.ok('exp:%s' % e.name, inputs, params):
            print('  %s: up to date, skipped' % e.name, flush=True)
            continue
        t = time.time()
        print('  %s: %d cycle(s)' % (e.name, len(have)), flush=True)
        rows = analyse(cfg, e, have, grid, regions, a.field, a.big)
        if not rows:
            fresh.record('exp:%s' % e.name, inputs, params, [])
            continue
        csv_path = os.path.join(cfg['outdir'], 'ssh_stability_%s.csv' % slug)
        with open(csv_path, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=['cycle', 'region'] + METRICS
                                + ['big_lon', 'big_lat'])
            wr.writeheader()
            wr.writerows(rows)
        v = verdict(rows, regions)
        json_path = os.path.join(cfg['outdir'], 'ssh_stability_%s.json' % slug)
        clean = [{k: (None if isinstance(x, float) and not np.isfinite(x)
                      else x) for k, x in row.items()} for row in v]
        with open(json_path, 'w') as fh:
            json.dump({'experiment': e.name, 'cycles': have, 'field': a.field,
                       'big': a.big, 'verdict': clean}, fh, indent=1)
        png = figure(rows, regions, have, e.name, cfg,
                     'cycle_ssh_stability_%s.png' % slug)
        fresh.record('exp:%s' % e.name, inputs, params, [png, csv_path, json_path])
        flagged = [x for x in v if x['flag']]
        print('  %s: %d figure, %d flagged trend(s)%s in %.0fs'
              % (e.name, 1, len(flagged),
                 ' (%s)' % ', '.join('%s/%s' % (x['region'], x['metric'])
                                     for x in flagged[:6]) if flagged else '',
                 time.time() - t), flush=True)
    fresh.save()
    print('done in %.0fs' % (time.time() - t_all), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
