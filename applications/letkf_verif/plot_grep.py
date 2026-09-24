#!/usr/bin/env python3
"""Monthly means against the GREP reanalysis ensemble.

Three figure families, all driven by the `grep:` block in experiments.yaml
(remove the block and this stage does nothing):

    content     time series of heat and salt content over the configured
                depth bands, the model at every cycle against each GREP
                member's monthly value, globally and per region
    sections    monthly-mean vertical sections of T, S and the east/north
                velocity, one column per member beside one per experiment
    bias maps   the monthly-mean model minus each member, as the band-mean
                temperature and salinity difference
    sea level   global-mean sea level and its parts -- SSH (the mass term),
                thermosteric and halosteric height -- model at every cycle
                against each member's monthly value (`sealevel: false` to
                turn off)

The three members are drawn separately rather than as an ensemble mean: the
spread between them is the only uncertainty estimate available here, and a
model-minus-GREP difference is only interesting once it is bigger than the
difference between two reanalyses of the same month.

Cost is dominated by reading the model history. The content integrals run at
every cycle, because the time series is the one product that wants density;
the sections subsample to one cycle a day (`section_hour:`), which changes a
monthly mean by far less than the month-to-month signal it is there to show.

The field averaged is the ocean BACKGROUND valid at each cycle, through
Experiment.background() -- the same field every other state-space diagnostic
in this suite scores, and the forecast valid at the centre of the cycle's DA
window.

Run with the gdas-marine-viz venv and OMP_NUM_THREADS=1.

    python3 plot_grep.py experiments.yaml --outdir page
"""

import argparse
import csv
import os
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import lv_grep as G  # noqa: E402
import lv_plot as P  # noqa: E402
import lv_statespace as S  # noqa: E402
import plot_statespace as PS  # noqa: E402
from lv_common import Fresh, Grid, load_config, region_list  # noqa: E402

# Content is reported as the band-mean temperature and salinity as well as
# the integral: 1.74e10 J/m2 is unreadable, 14.2 degC is not, and the two
# carry exactly the same information once the band thickness is fixed.
CONTENT = [('ohc', 'heat', 'degC', G.RHO0 * G.CP, 'band-mean temperature'),
           ('sc', 'salt', 'psu', G.RHO0 * 1e-3, 'band-mean salinity')]

# The members are drawn in neutral tones with their own dash and marker:
# distinguishable from each other, and clearly a different KIND of line from
# the experiments, which keep lv_plot's categorical colours. A member sharing
# an experiment's colour would read as the same series.
MEMBER_STYLE = {'cglo': (':', 'o', '#4f4e4a'),
                'glor': ('--', 's', '#8a8984'),
                'oras': ('-.', '^', '#b9b7b0')}
_FALLBACK = [('--', 'o', '#4f4e4a'), (':', 's', '#8a8984'),
             ('-.', '^', '#b9b7b0')]


def _member_style(member, i):
    return MEMBER_STYLE.get(member, _FALLBACK[i % 3])


def _mean(values):
    """Mean of a list of 2-D arrays, NaN where nothing contributed."""
    acc, cnt = None, None
    for v in values:
        ok = np.isfinite(v)
        a = np.where(ok, v, 0.0)
        acc = a if acc is None else acc + a
        cnt = ok.astype('i4') if cnt is None else cnt + ok
    if acc is None:
        return None
    with np.errstate(invalid='ignore'):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


# --------------------------------------------------------------------------
# the model side, one month
# --------------------------------------------------------------------------

def model_month(cfg, entry, exp, grid, cycles, lines, uv_lines,
                stride, log, floor=None):
    """Monthly reduction of one experiment: content, sections, level maps.

    Returns (data, rows, slrows). ``rows`` is the per-cycle region series
    that feeds the time series and the csv; ``slrows`` the per-cycle global
    sea level, empty without ``floor``; ``data`` holds the monthly means.

    Reduced cycle by cycle and accumulated: a monthly mean of the full 3-D
    field would be 466 MB per variable, and nothing here needs one. Every
    quantity taken is linear in the state, so the mean of the reduction is
    the reduction of the mean.
    """
    bands = entry['bands']
    section_hour = entry.get('section_hour')
    want_sec = bool(entry.get('sections')) and bool(lines)
    regions = _regions(cfg, grid)

    acc = {G.band_key(b): {'ohc': [], 'sc': [], 'deep': None} for b in bands}
    sec, axes, nsec = {}, {}, 0
    rows, slrows = [], []
    for n, cycle in enumerate(cycles):
        path = G.background_file(exp, cycle)
        if path is None:
            continue
        t0 = time.time()
        content = G.model_content(path, bands, floor=floor)
        if 'sealevel' in content:
            slrows.append(_sealevel_row(grid, floor, content['sealevel'],
                                        exp.name, cycle))
        for b in bands:
            k = G.band_key(b)
            acc[k]['ohc'].append(content[k]['ohc'].astype('f4'))
            acc[k]['sc'].append(content[k]['sc'].astype('f4'))
            # A column counts as reaching the band bottom only if it does so
            # at EVERY cycle, so the monthly mean is an average over one
            # fixed volume rather than a varying one.
            d = content[k]['deep']
            acc[k]['deep'] = d if acc[k]['deep'] is None else (acc[k]['deep'] & d)
        rows.extend(_content_rows(grid, regions, content, bands, exp.name,
                                  cycle))

        if want_sec and (not section_hour or cycle[8:10] == str(section_hour)):
            if not axes:
                # The depth axis comes from one cycle's layer thickness. This
                # output is remapped to fixed z levels, so it is the same
                # every cycle; re-deriving it per cycle would cost a second
                # each for an identical answer.
                axes, _wet = S.section_geometry(grid, path, lines, stride)
            for var in entry['variables']:
                use = lines if var not in ('u', 'v') else uv_lines
                if not use:
                    continue
                planes = S.section_planes(grid, path, [var], use,
                                          exp.varmap, stride, wet_mask=True)
                for key, plane in planes.items():
                    sec.setdefault(key, []).append(plane)
            nsec += 1
        if log and (n % 20 == 0 or n == len(cycles) - 1):
            log('      %s (%d/%d) %.0fs' % (cycle, n + 1, len(cycles),
                                            time.time() - t0))

    data = {'n_cycles': len([c for c in cycles
                             if G.background_file(exp, c)]),
            'n_sections': nsec}
    for b in bands:
        k = G.band_key(b)
        if not acc[k]['ohc']:
            continue
        data['content/%s/ohc' % k] = _mean(acc[k]['ohc'])
        data['content/%s/sc' % k] = _mean(acc[k]['sc'])
        data['content/%s/deep' % k] = acc[k]['deep']
    for key, planes in sec.items():
        data['sec/%s' % key] = _mean(planes)
    for key, val in axes.items():
        data['axis/%s' % key] = val
    return data, rows, slrows


def _sealevel_row(grid, floor, parts, who, when):
    """Global means over the common-floor volume, one csv row.

    SSH is averaged over the same columns as the steric parts so the three
    add up over one area.
    """
    keep = grid.mask & (floor > 0)
    row = {'who': who, 'when': when, 'region': 'global'}
    for k in SEALEVEL_FIELDS:
        row[k] = grid.wmean(np.where(keep, parts[k], np.nan))
    return row


def _regions(cfg, grid):
    """{name: mask} in the suite's display order, global first."""
    every = grid.regions(cfg)
    order = ['global'] + [r for r in region_list(cfg) if r in every]
    return [(n, every[n]) for n in order if every[n].any()]


def _content_rows(grid, regions, content, bands, who, when):
    """One csv row per (band, region): the integral and its readable form."""
    out = []
    for b in bands:
        k = G.band_key(b)
        dz = b[1] - b[0]
        for name, mask in regions:
            sel = mask & content[k]['deep']
            if not sel.any():
                continue
            row = {'who': who, 'when': when, 'band': k, 'region': name}
            for field, _lab, _u, scale, _t in CONTENT:
                v = grid.wmean(np.where(sel, content[k][field], np.nan))
                row[field] = v
                row[field + '_mean'] = v / (scale * dz)
            out.append(row)
    return out


# --------------------------------------------------------------------------
# the GREP side, one month
# --------------------------------------------------------------------------

def grep_month(cfg, entry, member, grid, month, lines, uv_lines, stride,
               depth, log, floor=None):
    """The same reduction for one GREP member and month.

    Returns (data, rows, slrows), as model_month does.
    """
    bands = entry['bands']
    path = G.file_for(entry, month[:4])
    it = G.month_index(path, month)
    if it is None:
        return None, [], []
    data = {}
    content = G.grep_content(entry, month, member, grid, bands)
    if content is None:
        return None, [], []
    who = '%s/%s' % (G.LABEL, member)
    slrows = []
    if floor is not None:
        parts = G.grep_sealevel(entry, month, member, grid, floor)
        if parts is not None:
            slrows.append(_sealevel_row(grid, floor, parts, who, month))
    for b in bands:
        k = G.band_key(b)
        data['content/%s/ohc' % k] = content[k]['ohc'].astype('f4')
        data['content/%s/sc' % k] = content[k]['sc'].astype('f4')
        data['content/%s/deep' % k] = content[k]['deep']
    rows = _content_rows(grid, _regions(cfg, grid), content, bands, who,
                         month)
    if entry.get('sections') and lines and depth is not None:
        with G.GrepSource(path, member, it, grid, depth) as src:
            for var in entry['variables']:
                use = lines if var not in ('u', 'v') else uv_lines
                if not use:
                    continue
                planes = S.section_planes(grid, src, [var], use, None,
                                          stride, wet_mask=False)
                for key, plane in planes.items():
                    data['sec/%s' % key] = plane
    if log:
        log('      %s %s: %d field(s)' % (G.LABEL, member, len(data)))
    return data, rows, slrows


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------

def _cache_dir(cfg):
    d = os.path.join(cfg['outdir'], 'grep')
    os.makedirs(d, exist_ok=True)
    return d


def _save_npz(path, data):
    """Atomic: np.savez_compressed appends .npz, so write then rename."""
    tmp = path + '.tmp'
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in data.items()})
    os.replace(tmp + '.npz', path)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_content(cfg, entry, grid, rows, band, region, fname, note=''):
    """Heat and salt content through time: model per cycle, GREP per month."""
    k = G.band_key(band)
    sel = [r for r in rows if r['band'] == k and r['region'] == region]
    if not sel:
        return None
    exps = [e.name for e in cfg['experiments']
            if any(r['who'] == e.name for r in sel)]
    members = [m for m in entry['members']
               if any(r['who'] == '%s/%s' % (G.LABEL, m) for r in sel)]
    if not exps and not members:
        return None
    colours = P.color_map([e.name for e in cfg['experiments']])

    fig, axs = plt.subplots(len(CONTENT), 1, figsize=(9.5, 6.4), sharex=True,
                            squeeze=False)
    for i, (field, label, unit, scale, title) in enumerate(CONTENT):
        ax = axs[i][0]
        for name in exps:
            pts = sorted((r['when'], r[field + '_mean']) for r in sel
                         if r['who'] == name)
            t = [_stamp(w) for w, _v in pts]
            v = np.array([x for _w, x in pts], dtype='f8')
            ok = np.isfinite(v)
            if ok.any():
                ax.plot(np.array(t)[ok], v[ok], '-', lw=1.2,
                        color=colours.get(name, P.INK), label=name)
        for j, m in enumerate(members):
            who = '%s/%s' % (G.LABEL, m)
            pts = sorted((r['when'], r[field + '_mean']) for r in sel
                         if r['who'] == who)
            ls, mk, col = _member_style(m, j)
            for w, val in pts:
                if not np.isfinite(val):
                    continue
                # Drawn as a bar across the whole calendar month, because
                # that is the interval the value is a mean over: a single
                # marker would invite reading it as an instantaneous value
                # comparable with one model cycle.
                lo, hi = _month_span(w)
                ax.plot([lo, hi], [val, val], ls, lw=1.6, color=col,
                        label='%s %s' % (G.LABEL, m) if w == pts[0][0] else None)
                ax.plot([_mid(lo, hi)], [val], mk, ms=4.5, color=col)
        ax.set_ylabel('%s (%s)' % (title, unit))
        P.tidy(ax)
        if i == 0:
            P.maybe_legend(ax, fontsize=8, ncol=2)
    axs[-1][0].set_xlabel('valid time')
    fig.autofmt_xdate()
    fig.suptitle('%s: %g-%g m heat and salt content -- model at every cycle, '
                 '%s members as monthly means%s'
                 % (region, band[0], band[1], G.LABEL, note), fontsize=11)
    fig.tight_layout()
    return P.save(fig, cfg, fname, dpi=110, quiet=True)


SEALEVEL_FIELDS = ['ssh', 'thermo', 'halo']


def fig_sealevel(cfg, entry, slrows, fname, note=''):
    """Global-mean sea level and its parts: model per cycle, GREP per month.

    Four panels, mm:

        total        SSH + thermosteric + halosteric
        SSH          the model's ave_ssh / GREP's zos. MOM6 here and the
                     three NEMO reanalyses are all Boussinesq, where volume
                     does not respond to density, so the global mean of SSH
                     carries NO steric signal: it moves only with the net
                     water crossing the surface -- E-P-R and, in this model,
                     the water taken into sea ice, which a p_surf = 0 ocean
                     does not see as displacement. That makes it the mass
                     (barystatic) term, read directly.
        thermo/halo  diagnosed from T and S, the steric term the Boussinesq
                     SSH leaves out (Greatbatch 1994)

    Adding the steric parts back to SSH assumes GREP's zos, like the model's,
    is the raw Boussinesq field without the Greatbatch correction already
    applied (the CMIP convention: zos and zostoga separate). A member that
    had already added it would count the steric term twice in the total.

    Referencing differs by panel, deliberately. The SSH datum is arbitrary
    and differs by ~1.9 m between the members, so each SSH series is shown
    against its OWN mean over the first month compared. The steric parts are
    integrated with one equation of state over one volume (common_floor), so
    their offsets are real and they share ONE reference: the GREP members'
    mean for that month.
    """
    if not slrows:
        return None
    months = sorted({str(r['when'])[:6] for r in slrows})
    ref_month = months[0]
    exps = [e.name for e in cfg['experiments']
            if any(r['who'] == e.name for r in slrows)]
    members = [m for m in entry['members']
               if any(r['who'] == '%s/%s' % (G.LABEL, m) for r in slrows)]
    if not members:
        return None

    def at_ref(who, field):
        v = [r[field] for r in slrows if r['who'] == who
             and str(r['when'])[:6] == ref_month and np.isfinite(r[field])]
        return float(np.mean(v)) if v else np.nan

    steric_ref = {f: np.nanmean([at_ref('%s/%s' % (G.LABEL, m), f)
                                 for m in members])
                  for f in ('thermo', 'halo')}

    def series(who):
        pts = sorted((str(r['when']), r) for r in slrows if r['who'] == who)
        ssh0 = at_ref(who, 'ssh')
        out = {'when': [w for w, _r in pts]}
        out['ssh'] = np.array([r['ssh'] - ssh0 for _w, r in pts])
        for f in ('thermo', 'halo'):
            out[f] = np.array([r[f] - steric_ref[f] for _w, r in pts])
        out['total'] = out['ssh'] + out['thermo'] + out['halo']
        return {k: (v * 1e3 if k != 'when' else v) for k, v in out.items()}

    panels = [('total', 'sea level\n(SSH + steric)'),
              ('ssh', 'SSH: mass term\n(own first-month mean)'),
              ('thermo', 'thermosteric'),
              ('halo', 'halosteric')]
    colours = P.color_map([e.name for e in cfg['experiments']])
    fig, axs = plt.subplots(len(panels), 1, figsize=(9.5, 9.0), sharex=True,
                            squeeze=False)
    for i, (field, title) in enumerate(panels):
        ax = axs[i][0]
        for name in exps:
            s = series(name)
            v = s[field]
            ok = np.isfinite(v)
            if ok.any():
                t = np.array([_stamp(w) for w in s['when']])
                ax.plot(t[ok], v[ok], '-', lw=1.2,
                        color=colours.get(name, P.INK), label=name)
        for j, m in enumerate(members):
            s = series('%s/%s' % (G.LABEL, m))
            ls, mk, col = _member_style(m, j)
            for n, (w, val) in enumerate(zip(s['when'], s[field])):
                if not np.isfinite(val):
                    continue
                lo, hi = _month_span(w)
                ax.plot([lo, hi], [val, val], ls, lw=1.6, color=col,
                        label='%s %s' % (G.LABEL, m) if n == 0 else None)
                ax.plot([_mid(lo, hi)], [val], mk, ms=4.5, color=col)
        ax.axhline(0.0, color=P.INK, lw=0.5, alpha=0.4)
        ax.set_ylabel('%s (mm)' % title)
        P.tidy(ax)
        if i == 0:
            P.maybe_legend(ax, fontsize=8, ncol=2)
    axs[-1][0].set_xlabel('valid time')
    fig.autofmt_xdate()
    fig.suptitle('global-mean sea level and its parts -- model at every '
                 'cycle, %s members as monthly means\nsteric: common '
                 'reference (%s mean, %s), common floor; SSH: each against '
                 'its own %s mean%s'
                 % (G.LABEL, G.LABEL, ref_month, ref_month, note),
                 fontsize=10)
    fig.tight_layout()
    return P.save(fig, cfg, fname, dpi=110, quiet=True)


def _stamp(when):
    import datetime
    if len(str(when)) == 6:
        return datetime.datetime(int(str(when)[:4]), int(str(when)[4:6]), 15)
    w = str(when)
    return datetime.datetime(int(w[:4]), int(w[4:6]), int(w[6:8]), int(w[8:10]))


def _month_span(month):
    import calendar
    import datetime
    y, m = int(str(month)[:4]), int(str(month)[4:6])
    return (datetime.datetime(y, m, 1),
            datetime.datetime(y, m, calendar.monthrange(y, m)[1], 23))


def _mid(lo, hi):
    return lo + (hi - lo) / 2


def fig_sections(cfg, entry, month, model, grep, tag):
    """Monthly-mean sections: GREP members first, then the experiments."""
    names = (['%s %s' % (G.LABEL, m) for m in sorted(grep)]
             + [n for n in model])
    axes = {}
    src = next((d for d in model.values() if any(kk.startswith('axis/')
                                                 for kk in d)), None)
    if src is not None:
        for kk, v in src.items():
            if not kk.startswith('axis/'):
                continue
            what, _, line = kk[len('axis/'):].partition('_')
            x, d = axes.get(line, (None, None))
            axes[line] = (v if what == 'x' else x, v if what == 'depth' else d)

    out = []
    for var in entry['variables']:
        tags = sorted({kk.split('sec/')[1].rpartition('_')[2]
                       for d in list(model.values()) + list(grep.values())
                       for kk in d if kk.startswith('sec/%s_' % var)},
                      key=_line_order)
        if not tags:
            continue
        rows = []
        for line in tags:
            fields = ([grep[m].get('sec/%s_%s' % (var, line))
                       for m in sorted(grep)]
                      + [model[n].get('sec/%s_%s' % (var, line))
                         for n in model])
            rows.append((line, fields))
        limits = _limits(cfg, rows, var)
        for cut, suffix, label in PS.section_depth_views(cfg):
            fname = 'grep_sections_%s%s_%s.png' % (var, suffix, tag)
            got = PS.section_grid(
                cfg, rows, names, axes,
                '%s monthly mean, %s, %s -- %s members and the experiments'
                % (var, month, label, G.LABEL),
                fname, limits, cb_label=_units(var), depth_cut=cut,
                note=PS._tripolar_note)
            if got:
                out.append(got)
    return out


def _line_order(tag):
    return (0 if tag[:3] == 'lat' else 1, tag)


def _units(var):
    return {'Temp': 'degC', 'Salt': 'psu', 'u': 'm/s', 'v': 'm/s'}.get(var, '')


def _limits(cfg, rows, var):
    """One shared (vmin, vmax, cmap) so the columns stay comparable.

    T and S go through the background sections' own rule
    (plot_statespace._section_limits, kind 'bkg'): the same colormap, and
    the same fixed `map_limits: sections:` range when the config sets one,
    so a GREP section reads on the same scale as the background section of
    the same field. u/v have no background section to match -- the
    background draws speed -- and keep the diverging scale the suite uses
    for every signed field.
    """
    if var in ('u', 'v'):
        vals = np.concatenate([f[np.isfinite(f)].ravel()
                               for _k, fields in rows for f in fields
                               if f is not None and np.any(np.isfinite(f))]
                              or [np.array([0.0])])
        hi = float(np.nanpercentile(np.abs(vals), 99)) or 0.1
        return -hi, hi, P.DIVERGING
    return PS._section_limits(cfg, 'bkg', var, rows)


def fig_bias_maps(cfg, entry, grid, month, model, grep, band, tag):
    """Model minus each member, as the band-mean T and S difference."""
    k = G.band_key(band)
    dz = band[1] - band[0]
    stride = int(cfg.get('map_stride', 2))
    deep = None
    for d in list(model.values()) + list(grep.values()):
        m = d.get('content/%s/deep' % k)
        if m is None:
            return None
        m = np.asarray(m, dtype=bool)
        deep = m if deep is None else (deep & m)
    keep = grid.mask & deep
    out = []
    for field, label, unit, scale, title in CONTENT:
        rows, names = [], []
        for n, d in model.items():
            a = d.get('content/%s/%s' % (k, field))
            if a is None:
                continue
            names.append(n)
            row = []
            for m in sorted(grep):
                b = grep[m].get('content/%s/%s' % (k, field))
                diff = (np.where(keep, np.asarray(a) - np.asarray(b), np.nan)
                        / (scale * dz))
                row.append(diff[::stride, ::stride])
            rows.append((n, row))
        if not rows:
            continue
        cols = ['%s %s' % (G.LABEL, m) for m in sorted(grep)]
        hi = max((float(np.nanpercentile(np.abs(f), 98))
                  for _n, fs in rows for f in fs
                  if f is not None and np.any(np.isfinite(f))), default=1.0)
        hi = hi or 1.0
        got = PS.map_grid(
            cfg, grid, rows, cols,
            'model minus %s, %s-%g m %s, %s'
            % (G.LABEL, band[0], band[1], title, month),
            'grep_bias_%s_%s_%s.png' % (label, k, tag),
            PS.views_for('ocean')[0],
            limits={n: (-hi, hi, P.DIVERGING) for n, _f in rows},
            cb_label='model - %s (%s)' % (G.LABEL, unit))
        if got:
            out.append(got)
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', nargs='?', default=None)
    ap.add_argument('--config', dest='config_opt', default=None)
    ap.add_argument('--root', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--cache', action='append', default=None)
    ap.add_argument('--experiment', action='append', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt

    t_all = time.time()
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    entry = G.configured(cfg)
    if entry is None:
        print('GREP: no `grep:` block in the config -- nothing to do',
              flush=True)
        return 0

    grid = Grid(cfg['grid'])
    S.set_rotation(grid)
    rotated = getattr(grid, 'cos_rot', None) is not None
    os.makedirs(cfg['figs'], exist_ok=True)
    cdir = _cache_dir(cfg)
    fresh = Fresh(cfg, 'grep', force=a.force, script=__file__)
    stride = int(cfg.get('map_stride', 2))
    lines = S.section_lines(grid, cfg, stride) if entry.get('sections') else []
    exps = [e for e in cfg['experiments']
            if not a.experiment or e.name in a.experiment]

    # Without cos_rot/sin_rot the model's u/v stay in the logical grid frame
    # while GREP's uo/vo are eastward/northward. The two frames coincide south
    # of the tripolar fold and diverge north of it, so the velocity sections
    # keep only the transects that stay south of it rather than quietly
    # comparing two different quantities in the Arctic.
    uv_lines = lines
    if not rotated and any(v in ('u', 'v') for v in entry['variables']):
        uv_lines = [ln for ln in lines if not _crosses_fold(ln, grid)]
        print('GREP: %s carries no cos_rot/sin_rot, so the model u/v are in '
              'the logical grid frame and GREP is east/north. Velocity '
              'sections are limited to the %d of %d transect(s) that stay '
              'south of %g N. Regenerate the grid with make_gridfile.py to '
              'compare velocities north of the fold.'
              % (os.path.basename(cfg['grid']), len(uv_lines), len(lines),
                 S.TRIPOLAR_LAT), flush=True)

    months, per_exp = {}, {}
    for e in exps:
        got = G.months_for(cfg, entry, e)
        if not got:
            print('  %s: %s -- skipped'
                  % (e.name, G.skip_reason(cfg, entry, e)), flush=True)
            continue
        per_exp[e.name] = got
        for m, cyc in got:
            months.setdefault(m, set()).update(cyc)
    if not per_exp:
        # Each experiment has already said why on its own line above, and
        # those reasons can differ (a missing archive is not a short month).
        # Repeating a config-only reason here would contradict them.
        print('GREP: nothing to compare', flush=True)
        fresh.save()
        return 0

    print('GREP: %d month(s) %s, %d experiment(s)'
          % (len(months), '/'.join(sorted(months)), len(per_exp)), flush=True)

    # -- the common floor for the sea-level integrals ----------------------
    floor = (_sealevel_floor(cfg, entry, grid, exps, per_exp, cdir, fresh)
             if entry.get('sealevel') else None)
    floor_sig = None if floor is None else round(float(floor.sum()), 1)

    # -- the model side --------------------------------------------------
    rows, slrows = [], []
    model = {}
    for e in exps:
        for month, cycles in per_exp.get(e.name, []):
            npz = os.path.join(cdir, '%s_%s.npz' % (P.slug(e.name), month))
            csvp = _csv_path(cfg, e.name)
            key = 'model:%s:%s' % (e.name, month)
            inputs = [G.background_file(e, cycles[-1]), __file__]
            params = {'cycles': cycles, 'bands': entry['bands'],
                      'variables': entry['variables'],
                      'section_hour': entry.get('section_hour'),
                      'sections': bool(entry.get('sections')),
                      'lines': [ln[0] for ln in lines],
                      'uv_lines': [ln[0] for ln in uv_lines],
                      'floor': floor_sig}
            if fresh.ok(key, inputs, params) and os.path.exists(npz):
                model.setdefault(month, {})[e.name] = _load_npz(npz)
                rows.extend(_rows_from(csvp, e.name, month))
                if floor is not None:
                    slrows.extend(_rows_from(
                        _csv_path(cfg, e.name, 'sealevel'), e.name, month,
                        'sealevel'))
                print('  %s %s: up to date' % (e.name, month), flush=True)
                continue
            t = time.time()
            print('  %s %s: %d background(s), %.0f%% of the month'
                  % (e.name, month, len(cycles),
                     100.0 * G.coverage(month, cycles, entry.get('hour'),
                                        entry.get('step_hours', 6))),
                  flush=True)
            data, mrows, msl = model_month(cfg, entry, e, grid, cycles,
                                           lines, uv_lines, stride,
                                           log=lambda s: print(s, flush=True),
                                           floor=floor)
            if not data:
                continue
            _save_npz(npz, data)
            model.setdefault(month, {})[e.name] = data
            rows.extend(mrows)
            slrows.extend(msl)
            fresh.record(key, inputs, params, [npz])
            print('    %.0fs' % (time.time() - t), flush=True)

    # -- the GREP side ---------------------------------------------------
    depth = None
    for e in exps:
        for month, cycles in per_exp.get(e.name, []):
            p = G.background_file(e, cycles[0])
            if p:
                depth = G.model_depth(p)
                break
        if depth is not None:
            break

    grep = {}
    for month in sorted(model):
        for member in entry['members']:
            npz = os.path.join(cdir, 'GREP_%s_%s.npz' % (member, month))
            key = 'grep:%s:%s' % (member, month)
            inputs = [G.file_for(entry, month[:4]), __file__]
            params = {'bands': entry['bands'], 'variables': entry['variables'],
                      'sections': bool(entry.get('sections')),
                      'lines': [ln[0] for ln in lines],
                      'uv_lines': [ln[0] for ln in uv_lines],
                      'floor': floor_sig}
            if fresh.ok(key, inputs, params) and os.path.exists(npz):
                grep.setdefault(month, {})[member] = _load_npz(npz)
                rows.extend(_rows_from(_csv_path(cfg, G.LABEL),
                                       '%s/%s' % (G.LABEL, member), month))
                if floor is not None:
                    slrows.extend(_rows_from(
                        _csv_path(cfg, G.LABEL, 'sealevel'),
                        '%s/%s' % (G.LABEL, member), month, 'sealevel'))
                continue
            t = time.time()
            print('  %s %s %s' % (G.LABEL, member, month), flush=True)
            data, grows, gsl = grep_month(cfg, entry, member, grid, month,
                                          lines, uv_lines, stride, depth,
                                          log=lambda s: print(s, flush=True),
                                          floor=floor)
            if data is None:
                continue
            _save_npz(npz, data)
            grep.setdefault(month, {})[member] = data
            rows.extend(grows)
            slrows.extend(gsl)
            fresh.record(key, inputs, params, [npz])
            print('    %.0fs' % (time.time() - t), flush=True)

    _write_csv(cfg, rows)
    if slrows:
        _write_csv(cfg, slrows, 'sealevel')

    # -- figures ---------------------------------------------------------
    outs = []
    regions = [n for n, _m in _regions(cfg, grid)]
    # Any month the model does not fully cover is named on every panel: a
    # partial month against GREP's true monthly mean carries the seasonal
    # march of the missing days as if it were model error.
    partial = ['%s %.0f%%' % (m, 100.0 * G.coverage(
        m, sorted({c for e in exps for mm, cc in per_exp.get(e.name, [])
                   if mm == m for c in cc}), entry.get('hour'),
        entry.get('step_hours', 6))) for m in sorted(model)]
    partial = [p for p in partial if float(p.split()[-1][:-1]) < 99.0]
    cover_note = ('\nmodel covers %s of the month' % ', '.join(partial)
                  if partial else '')
    for band in entry['bands']:
        for region in regions:
            f = fig_content(cfg, entry, grid, rows, band, region,
                            'grep_content_%s_region_%s.png'
                            % (G.band_key(band), P.slug(region)),
                            note=cover_note)
            if f:
                outs.append(f)
    f = fig_sealevel(cfg, entry, slrows, 'grep_sealevel_global.png',
                     note=cover_note)
    if f:
        outs.append(f)
    for month in sorted(model):
        if month not in grep:
            continue
        if entry.get('sections'):
            outs.extend(fig_sections(cfg, entry, month, model[month],
                                     grep[month], month) or [])
        for band in entry['bands']:
            outs.extend(fig_bias_maps(cfg, entry, grid, month, model[month],
                                      grep[month], band, month) or [])
    fresh.record('figures', [__file__],
                 {'months': sorted(model), 'regions': regions,
                  'bands': entry['bands']}, outs)
    fresh.save()
    print('GREP: %d figure(s) in %.0fs' % (len(outs), time.time() - t_all),
          flush=True)
    return 0


def _sealevel_floor(cfg, entry, grid, exps, per_exp, cdir, fresh):
    """The common floor (lv_grep.common_floor), cached, or None.

    Built once, from the first compared month and the first experiment's
    background, and shared by every experiment and member, so all the sea
    level series are integrals over one volume. Bathymetry does not change
    between months or experiments on one grid; recomputing it per month
    would only let the volume drift.
    """
    first = None
    for e in exps:
        for month, cycles in per_exp.get(e.name, []):
            if first is None or month < first[0]:
                first = (month, G.background_file(e, cycles[0]))
    if first is None or first[1] is None:
        return None
    month, bg = first
    npz = os.path.join(cdir, 'floor.npz')
    inputs = [cfg['grid'], G.file_for(entry, month[:4])]
    params = {'month': month, 'members': entry['members']}
    if fresh.ok('floor', inputs, params) and os.path.exists(npz):
        return _load_npz(npz)['floor']
    t = time.time()
    floor = G.common_floor(grid, bg, entry, month, entry['members'])
    _save_npz(npz, {'floor': floor})
    fresh.record('floor', inputs, params, [npz])
    print('  common floor from %s: mean %.0f m over %.1f%% of the wet area '
          '(%.0fs)' % (month, grid.wmean(floor),
                       100.0 * grid.wmean((floor > 0).astype('f8')),
                       time.time() - t), flush=True)
    return floor


def _crosses_fold(line, grid):
    _key, axis, idx, _x, warn = line
    return bool(warn) or (axis == 'lon'
                          and np.nanmax(grid.lat[:, idx]) > S.TRIPOLAR_LAT)


FIELDS = ['who', 'when', 'band', 'region', 'ohc', 'ohc_mean', 'sc', 'sc_mean']

# Per-kind csv layout: the columns, and the numeric ones among them.
CSV_KINDS = {
    'content': (FIELDS, ['ohc', 'ohc_mean', 'sc', 'sc_mean']),
    'sealevel': (['who', 'when', 'region'] + SEALEVEL_FIELDS,
                 SEALEVEL_FIELDS),
}


def _csv_path(cfg, who, kind='content'):
    """Where one producer's series lives. Single definition: the cached
    branch reads back exactly what _write_csv wrote."""
    return os.path.join(cfg['outdir'], 'grep_%s_%s.csv'
                        % (kind, P.slug(str(who).split('/')[0])))


def _write_csv(cfg, rows, kind='content'):
    """One csv per producer, so a skipped experiment keeps its old rows."""
    fields = CSV_KINDS[kind][0]
    by = {}
    for r in rows:
        who = r['who'].split('/')[0]
        by.setdefault(G.LABEL if who == G.LABEL else who, []).append(r)
    for who, rs in by.items():
        path = _csv_path(cfg, who, kind)
        seen, keep = set(), []
        for r in sorted(rs, key=lambda x: (x['who'], x.get('band', ''),
                                           x['region'], str(x['when']))):
            sig = (r['who'], r.get('band', ''), r['region'], str(r['when']))
            if sig in seen:
                continue
            seen.add(sig)
            keep.append({k: r.get(k) for k in fields})
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)


def _rows_from(path, who, when, kind='content'):
    """Rows for one producer and month from a previously written csv."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            if r['who'] != who or not str(r['when']).startswith(str(when)[:6]):
                continue
            for k in CSV_KINDS[kind][1]:
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError, KeyError):
                    r[k] = np.nan
            out.append(r)
    return out


if __name__ == '__main__':
    raise SystemExit(main())
