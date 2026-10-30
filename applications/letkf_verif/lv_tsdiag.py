"""Paired temperature and salinity at the profile observations, for T-S
(water-mass) diagrams.

Every other observation-space diagnostic here scores temperature and salinity
apart, as two obs types. A water mass is the pair: a model can fit both
profiles to within their errors one at a time and still put the Mediterranean
outflow on the wrong isopycnal. This pairs the two at each observed level --
the same profile, time, position and depth in the temperature and the salinity
file -- on the common sample, and keeps the observation and every experiment's
background and analysis there.

What is kept is the points themselves, not a reduction: depth range, region
and period are then plot-time choices (`ts_diagram:` in the config) and change
nothing in the cache. For Argo that is ~3e5 levels a cycle, ~9 MB compressed
with three experiments -- written to its own '<cycle>_tsdiag.npz' rather than
the obs bins so that the bins, which every cycle SUMS into, stay a fixed shape
and small.

Temperature is converted to potential temperature (EOS-80) on both sides.
The observations are in-situ, and so is H(x): the operator converts the
model's potential temperature to in-situ at the observation, which is why the
deep O-B is a few hundredths of a degree rather than the 0.3 degC adiabatic
offset at 4000 m it would be otherwise. Converting both with the same
salinity and pressure keeps the difference what the DA saw.

Pairing is on exact scaled-integer metadata, the join's own keys. A level
present in one file and not the other (a QC'd salinity, a temperature-only
float) is simply not a pair.
"""

import numpy as np

from lv_common import _KEY_SCALE, _scaled, basin_at, basin_regions, in_region
from lv_woa import depth_to_pressure, potential_temperature

# Bumped whenever what goes into '<cycle>_tsdiag.npz' changes; a cached file
# with an older (or no) version is rebuilt by the rejoin.
VERSION = 1

TEMP = 'insitu_temp_profile'
SALT = 'insitu_salt_profile'

# The metadata a T level and an S level must share to be one water sample.
PAIR_META = ('dateTime', 'stationID', 'latitude', 'longitude', 'depth')

# Stored per platform: the observation once, the model per experiment.
OBS_FIELDS = ('latitude', 'longitude', 'depth', 'theta', 'salt')
EXP_FIELDS = ('theta_b', 'salt_b', 'theta_a', 'salt_a')

# Depth ranges (m, [top, bottom]) a diagram is drawn for; `ts_diagram:
# {depth_ranges: [...]}` overrides. Below 500 m is where water masses are
# conserved enough to be named; above it the diagram is mostly the seasonal
# mixed layer.
DEFAULT_DEPTHS = [[0, 6000], [0, 500], [500, 2000]]

# Fixed theta-S grid, so one cycle's histogram adds to the next.
THETA_EDGES = np.round(np.arange(-2.5, 32.0 + 1e-9, 0.1), 4)
SALT_EDGES = np.round(np.arange(30.0, 38.0 + 1e-9, 0.02), 4)
# Density classes for the water-mass bias, sigma0 in kg m-3.
SIGMA_EDGES = np.round(np.arange(18.0, 28.5 + 1e-9, 0.05), 4)


def platform(obstype):
    """'insitu_temp_profile_argo' -> 'argo' (and 'profile' for no suffix)."""
    for prefix in (TEMP, SALT):
        if obstype.startswith(prefix):
            return obstype[len(prefix):].lstrip('_') or 'profile'
    return None


def is_pair_type(obstype):
    return obstype.startswith(TEMP) or obstype.startswith(SALT)


def pairs(obstypes):
    """{platform: (temperature type, salinity type)} for every platform that
    has both."""
    have = set(obstypes)
    out = {}
    for t in sorted(have):
        if t.startswith(TEMP):
            s = SALT + t[len(TEMP):]
            if s in have:
                out[platform(t)] = (t, s)
    return out


def depth_ranges(cfg):
    return [list(map(float, r)) for r in
            ((cfg.get('ts_diagram') or {}).get('depth_ranges')
             or DEFAULT_DEPTHS)]


def depth_slug(lo, hi):
    return 'd%g-%gm' % (lo, hi)


def fig_name(kind, plat, region_slug, dslug, period):
    """'ts_<density|bias>_<platform>_<region>_<depth>_<all|YYYYMM>.png' --
    shared by plot_tsdiag.py (writes) and build_report.py (finds)."""
    return 'ts_%s_%s_%s_%s_%s.png' % (kind, plat, region_slug, dslug, period)


# --------------------------------------------------------------------------
# seawater
# --------------------------------------------------------------------------

def sigma0(theta, s):
    """Potential density anomaly at the surface, kg m-3: the UNESCO (1981)
    one-atmosphere equation of state, rho(S, theta, 0) - 1000."""
    t = np.asarray(theta, dtype='f8')
    s = np.asarray(s, dtype='f8')
    rho_w = (999.842594 + 6.793952e-2 * t - 9.095290e-3 * t ** 2
             + 1.001685e-4 * t ** 3 - 1.120083e-6 * t ** 4
             + 6.536332e-9 * t ** 5)
    a = (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t ** 2 - 8.2467e-7 * t ** 3
         + 5.3875e-9 * t ** 4)
    b = -5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t ** 2
    with np.errstate(invalid='ignore'):
        return rho_w + a * s + b * s ** 1.5 + 4.8314e-4 * s ** 2 - 1000.0


# --------------------------------------------------------------------------
# compute side (compute_cycle.observation_block)
# --------------------------------------------------------------------------

def extract(aligned, common_pass):
    """The few arrays the pairing needs from one type's common sample.

    observation_block drops each type's ObsSets as soon as it is scored; this
    is what outlives them until the partner type has been read. The common
    sample is one set of observations, so the metadata and the value come
    from any experiment; only the departures differ.
    """
    names = [n for n, s in aligned.items()
             if s.n and s.ombg is not None and s.oman is not None]
    if not names:
        return None
    first = aligned[names[0]]
    if not all(m in first.meta for m in PAIR_META):
        return None
    return {'meta': {m: first.meta[m] for m in PAIR_META},
            'y': first.y,
            'pass': np.asarray(common_pass, dtype=bool),
            'ombg': {n: aligned[n].ombg for n in names},
            'oman': {n: aligned[n].oman for n in names}}


def _pair_key(part):
    cols = [_scaled(part['meta'][m], _KEY_SCALE[m]) for m in PAIR_META]
    full = np.ascontiguousarray(np.stack(cols, axis=1))
    return full.view(np.dtype((np.void, full.shape[1] * 8))).ravel()


def compute(plat, tpart, spart):
    """npz arrays for one platform from its temperature and salinity
    extracts, under 'tsdiag/<platform>/obs|<experiment>/<field>'."""
    if tpart is None or spart is None:
        return {}
    names = [n for n in tpart['ombg'] if n in spart['ombg']]
    if not names:
        return {}
    _common, it, js = np.intersect1d(_pair_key(tpart), _pair_key(spart),
                                     return_indices=True)
    meta = {m: v[it] for m, v in tpart['meta'].items()}
    t_o, s_o = tpart['y'][it], spart['y'][js]
    ok = tpart['pass'][it] & spart['pass'][js] & np.isfinite(t_o) \
        & np.isfinite(s_o)
    for m in ('latitude', 'longitude', 'depth'):
        ok &= np.isfinite(meta[m])
    # the sample stays common: a level any experiment lacks a model value
    # at is dropped for all of them
    model = {}
    for n in names:
        tb = t_o - tpart['ombg'][n][it]
        sb = s_o - spart['ombg'][n][js]
        ta = t_o - tpart['oman'][n][it]
        sa = s_o - spart['oman'][n][js]
        ok &= np.isfinite(tb) & np.isfinite(sb) & np.isfinite(ta) \
            & np.isfinite(sa)
        model[n] = (tb, sb, ta, sa)
    lat, depth = meta['latitude'][ok], meta['depth'][ok]
    p = depth_to_pressure(depth, lat)
    out = {}
    pre = 'tsdiag/%s/' % plat
    obs = {'latitude': lat, 'longitude': meta['longitude'][ok],
           'depth': depth,
           'theta': potential_temperature(t_o[ok], s_o[ok], p),
           'salt': s_o[ok]}
    for f in OBS_FIELDS:
        out[pre + 'obs/' + f] = np.asarray(obs[f], dtype='f4')
    for n, (tb, sb, ta, sa) in model.items():
        tb, sb, ta, sa = tb[ok], sb[ok], ta[ok], sa[ok]
        vals = {'theta_b': potential_temperature(tb, sb, p), 'salt_b': sb,
                'theta_a': potential_temperature(ta, sa, p), 'salt_a': sa}
        for f in EXP_FIELDS:
            out['%s%s/%s' % (pre, n, f)] = np.asarray(vals[f], dtype='f4')
    return out


# --------------------------------------------------------------------------
# reading back (plot_tsdiag.py)
# --------------------------------------------------------------------------

def per_platform(arrays):
    """{platform: {'obs'|experiment: {field: array}}} from one cycle's npz."""
    out = {}
    for k, v in arrays.items():
        parts = k.split('/')
        if len(parts) != 4 or parts[0] != 'tsdiag':
            continue                       # e.g. the 'tsdiag/version' stamp
        _t, plat, who, field = parts
        out.setdefault(plat, {}).setdefault(who, {})[field] = v
    return out


def region_masks(cfg, lat, lon):
    """[(region, mask)]: 'global', the basins, then the `regions:` boxes --
    the order lv_common.region_list() gives."""
    out = [('global', np.ones(lat.shape, dtype=bool))]
    mask_path = cfg.get('ocean_basin_mask')
    if mask_path:
        codes = basin_at(mask_path, lat, lon)
        out += [(name, codes == code)
                for code, name in basin_regions(mask_path)]
    seen = {name for name, _m in out}
    out += [(r['name'], in_region(r, lat, lon))
            for r in cfg.get('regions', []) if r['name'] not in seen]
    return out


def hist2d(theta, salt):
    """Counts on the fixed theta x S grid; points off the grid are dropped."""
    nt, ns = THETA_EDGES.size - 1, SALT_EDGES.size - 1
    i = np.searchsorted(THETA_EDGES, theta, side='right') - 1
    j = np.searchsorted(SALT_EDGES, salt, side='right') - 1
    ok = (i >= 0) & (i < nt) & (j >= 0) & (j < ns)
    return np.bincount(i[ok] * ns + j[ok],
                       minlength=nt * ns).reshape(nt, ns).astype('i4')


def _class_sums(idx, ok, d):
    nb = SIGMA_EDGES.size - 1
    return (np.bincount(idx[ok], weights=d[ok], minlength=nb),
            np.bincount(idx[ok], weights=d[ok] ** 2, minlength=nb))


def reduce_cycle(arrays, cfg, ranges=None):
    """One cycle's points to sums: {(platform, region, depth slug): {name:
    array}}, which add across cycles (see accumulate).

    Per key: 'obs/hist', '<exp>/hist_b', '<exp>/hist_a', 'n' per sigma0
    class of the observation, and the sums and sums of squares of model
    minus observation, theta and S, background and analysis, per class.
    """
    ranges = ranges or depth_ranges(cfg)
    out = {}
    nb = SIGMA_EDGES.size - 1
    for plat, d in per_platform(arrays).items():
        obs = d.get('obs')
        if not obs or not obs['theta'].size:
            continue
        exps = [n for n in d if n != 'obs']
        lat = obs['latitude'].astype('f8')
        lon = obs['longitude'].astype('f8')
        depth = obs['depth']
        th, sa = obs['theta'], obs['salt']
        cls = np.searchsorted(SIGMA_EDGES, sigma0(th, sa), side='right') - 1
        in_cls = (cls >= 0) & (cls < nb)
        for region, rmask in region_masks(cfg, lat, lon):
            for lo, hi in ranges:
                sel = rmask & (depth >= lo) & (depth < hi)
                if not sel.any():
                    continue
                acc = {'obs/hist': hist2d(th[sel], sa[sel])}
                ok = sel & in_cls
                acc['n'] = np.bincount(cls[ok], minlength=nb)
                for n in exps:
                    m = d[n]
                    for side in ('b', 'a'):
                        mt, ms = m['theta_' + side], m['salt_' + side]
                        acc['%s/hist_%s' % (n, side)] = hist2d(
                            mt[sel], ms[sel])
                        for var, model, ob in (('theta', mt, th),
                                               ('salt', ms, sa)):
                            dd = model.astype('f8') - ob
                            s1, s2 = _class_sums(cls, ok, dd)
                            acc['%s/d%s_%s' % (n, var, side)] = s1
                            acc['%s/d%s2_%s' % (n, var, side)] = s2
                out[(plat, region, depth_slug(lo, hi))] = acc
    return out


def accumulate(total, part):
    """Add one cycle's reduce_cycle() into a running total (in place)."""
    for key, acc in part.items():
        tot = total.setdefault(key, {})
        for k, v in acc.items():
            tot[k] = tot[k] + v if k in tot else v.copy()
    return total


def merge_classes(n, min_n):
    """Contiguous runs of the fixed sigma0 classes, [(i0, i1), ...] with i1
    exclusive, each holding at least ``min_n`` levels.

    The fixed 0.05 kg m-3 classes are right where water is dense in sigma0
    (the deep ocean: thousands of levels a class) and far too fine where it
    is not (the upper ocean spreads a few cycles over 150 classes, and the
    mean of a dozen levels is noise). Pooling until each class holds enough
    levels lets the resolution follow the data. A short remainder at the
    dense end joins the last full class rather than standing alone.
    """
    out, i0, acc = [], None, 0
    for i, c in enumerate(np.asarray(n)):
        if c <= 0 and i0 is None:
            continue
        if i0 is None:
            i0 = i
        acc += c
        if acc >= min_n:
            out.append((i0, i + 1))
            i0, acc = None, 0
    if i0 is not None and acc and out:
        last = np.flatnonzero(np.asarray(n)[i0:] > 0)
        out[-1] = (out[-1][0], i0 + int(last[-1]) + 1)
    return out


def class_edges(groups):
    """Lower and upper sigma0 of each merged class."""
    return (np.array([SIGMA_EDGES[a] for a, _b in groups]),
            np.array([SIGMA_EDGES[b] for _a, b in groups]))


def class_stats(acc, name, var, side, groups):
    """(n, mean, sd) of model minus observation in each merged class."""
    def pooled(a):
        return np.array([a[i0:i1].sum() for i0, i1 in groups], dtype='f8')
    n = pooled(acc['n'])
    s1 = acc.get('%s/d%s_%s' % (name, var, side))
    s2 = acc.get('%s/d%s2_%s' % (name, var, side))
    if s1 is None:
        nan = np.full(n.shape, np.nan)
        return n, nan, nan
    with np.errstate(invalid='ignore', divide='ignore'):
        nn = np.where(n > 0, n, np.nan)
        mean = pooled(s1) / nn
        sd = np.sqrt(np.maximum(pooled(s2) / nn - mean ** 2, 0.0))
    return n, mean, sd
