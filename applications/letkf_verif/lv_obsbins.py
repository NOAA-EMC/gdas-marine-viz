"""Spatially binned observation departures and obs-vs-model regressions.

The scalar obs-space metrics (lv_obsspace) say how well each system fits
its observations; they cannot say WHERE. This reduces every obs type on the
common sample to sums on a fixed latitude/longitude grid -- count, O-B, O-A,
their squares, the assigned and effective observation errors -- and, for
profile types, the same on depth x latitude. Sums rather than means, so the
cycles add: the report shows one date or every date pooled, from the same
files.

Alongside, a 2-D histogram of observation against background and against
analysis on a fixed per-variable range, with the least-squares line through
the points. A fit slope well below 1 with a high correlation is the classic
signature of a system that damps the signal it assimilates; the histogram
carries it in a form that also sums over cycles.

Everything here is written by compute_cycle.py into '<cycle>_obsbins.npz'
next to the cycle's JSON, under keys 'obsbins/<type>/<experiment>/<name>',
and read back by plot_obsbins.py through lv_plot.load_obsbins().
"""

import numpy as np

from lv_obsspace import is_profile

# Bin edge spacing in degrees; `obs_bins: {deg: 1}` in the config overrides.
DEFAULT_DEG = 1.0

# Fixed value ranges for the obs-vs-model histograms, per IODA variable, so
# one cycle's histogram adds to the next. The fallback is deliberately wide.
RANGES = {
    'seaSurfaceTemperature': (-2.0, 35.0),
    'waterTemperature': (-2.0, 35.0),
    'seaSurfaceSalinity': (28.0, 40.0),
    'salinity': (28.0, 40.0),
    'absoluteDynamicTopography': (-2.0, 2.5),
    'seaIceFraction': (0.0, 1.0),
    'seaIceFreeboard': (-0.5, 1.5),
}
NBIN_REG = 100

# The sums kept per bin, in this order; names double as npz keys.
SUMS = ('n', 's_y', 's_ombg', 's_ombg2', 's_oman', 's_oman2',
        's_r', 's_r2', 's_reff', 's_reff2')


def bin_deg(cfg):
    return float((cfg.get('obs_bins') or {}).get('deg', DEFAULT_DEG))


def grid_shape(deg):
    return int(round(180.0 / deg)), int(round(360.0 / deg))


def lat_edges(deg):
    return np.linspace(-90.0, 90.0, grid_shape(deg)[0] + 1)


def lon_edges(deg):
    return np.linspace(-180.0, 180.0, grid_shape(deg)[1] + 1)


def _bin_index(lat, lon, deg):
    ny, nx = grid_shape(deg)
    lon = ((lon + 180.0) % 360.0) - 180.0
    j = np.clip(((lat + 90.0) / deg).astype(int), 0, ny - 1)
    i = np.clip(((lon + 180.0) / deg).astype(int), 0, nx - 1)
    return j * nx + i, ny * nx


def _sums(idx, size, y, ombg, oman, r, reff):
    """Every SUMS array as a flat vector, via bincount (one pass each)."""
    def acc(w):
        if w is None:
            return np.zeros(size, dtype='f4')
        return np.bincount(idx, weights=w, minlength=size).astype('f4')
    return {
        'n': np.bincount(idx, minlength=size).astype('i4'),
        's_y': acc(y), 's_ombg': acc(ombg), 's_ombg2': acc(ombg * ombg),
        's_oman': acc(oman), 's_oman2': acc(oman * oman),
        's_r': acc(r), 's_r2': acc(None if r is None else r * r),
        's_reff': acc(reff), 's_reff2': acc(None if reff is None else reff * reff),
    }


def _fit(x, y):
    """Least-squares y = a x + b with correlation, on finite pairs."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return {'slope': np.nan, 'intercept': np.nan, 'r': np.nan, 'n': n}
    x, y = x[ok], y[ok]
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    sxy = np.sum((x - xm) * (y - ym))
    syy = np.sum((y - ym) ** 2)
    slope = sxy / sxx if sxx > 0 else np.nan
    r = sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else np.nan
    return {'slope': float(slope), 'intercept': float(ym - slope * xm),
            'r': float(r), 'n': n}


def compute(obstype, aligned, common_pass, cfg):
    """Bins and regressions for one obs type on the common sample.

    Returns ``(arrays, scalars)``: ``arrays`` maps
    '<experiment>/<name>' to the array to store, ``scalars`` maps
    experiment to the regression fits (they go into the cycle JSON).
    """
    deg = bin_deg(cfg)
    arrays, scalars = {}, {}
    for name, s in aligned.items():
        lat = s.meta.get('latitude')
        lon = s.meta.get('longitude')
        # An empty file (an obs type with nothing this cycle -- AMSR2 has a
        # few such) carries no ombg/oman groups at all; nothing to bin.
        if (s.n == 0 or s.ombg is None or s.oman is None or s.y is None
                or lat is None or lon is None):
            continue
        sel = common_pass & np.isfinite(s.ombg) & np.isfinite(s.oman)
        sel &= np.isfinite(lat) & np.isfinite(lon)
        y, ombg, oman = s.y[sel], s.ombg[sel], s.oman[sel]
        r = s.R[sel] if s.R is not None else None
        reff = s.Reff[sel] if s.Reff is not None else None
        ny, nx = grid_shape(deg)
        idx, size = _bin_index(lat[sel], lon[sel], deg)
        for k, v in _sums(idx, size, y, ombg, oman, r, reff).items():
            arrays['%s/map/%s' % (name, k)] = v.reshape(ny, nx)

        if is_profile(obstype) and 'depth' in s.meta:
            edges = np.asarray(cfg.get('depth_bins') or [0, 6000], dtype='f8')
            d = s.meta['depth'][sel]
            kd = np.clip(np.searchsorted(edges, d, side='right') - 1,
                         0, len(edges) - 2)
            jl = np.clip(((lat[sel] + 90.0) / deg).astype(int), 0, ny - 1)
            sidx = kd * ny + jl
            for k, v in _sums(sidx, (len(edges) - 1) * ny, y, ombg, oman,
                              r, reff).items():
                arrays['%s/sec/%s' % (name, k)] = v.reshape(len(edges) - 1, ny)

        if not y.size:
            continue
        lo, hi = RANGES.get(s.var, (float(np.nanpercentile(y, 0.5)),
                                    float(np.nanpercentile(y, 99.5))))
        edges = np.linspace(lo, hi, NBIN_REG + 1)
        hb = y - ombg                       # H(x_b): what the model said
        ha = y - oman
        arrays['%s/reg/bkg' % name] = np.histogram2d(y, hb, [edges, edges])[0].astype('i4')
        arrays['%s/reg/ana' % name] = np.histogram2d(y, ha, [edges, edges])[0].astype('i4')
        arrays['%s/reg/edges' % name] = edges.astype('f4')
        scalars[name] = {'bkg': _fit(y, hb), 'ana': _fit(y, ha),
                         'range': [lo, hi], 'deg': deg}
    return arrays, scalars


# --------------------------------------------------------------------------
# reading back
# --------------------------------------------------------------------------

def accumulate(total, arrays):
    """Add one cycle's bins into a running total (in place, returned)."""
    for k, v in arrays.items():
        if k.endswith('/reg/edges'):
            total.setdefault(k, v)
        elif k in total:
            total[k] = total[k] + v
        else:
            total[k] = v.copy()
    return total


def derived(a, prefix):
    """Means and RMS from the sums under '<prefix>/' (map or sec).

    Empty bins are NaN, not zero, so a map reads as missing where there were
    no observations rather than as a perfect fit.
    """
    n = a.get(prefix + '/n')
    if n is None:
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        nn = np.where(n > 0, n, np.nan).astype('f8')

        def mean(k):
            return a[prefix + '/' + k] / nn

        def rms(k):
            return np.sqrt(a[prefix + '/' + k] / nn)
        out = {
            'count': np.where(n > 0, n, np.nan).astype('f8'),
            'ombg_mean': mean('s_ombg'), 'ombg_rms': rms('s_ombg2'),
            'oman_mean': mean('s_oman'), 'oman_rms': rms('s_oman2'),
            'obs_mean': mean('s_y'),
            'r_rms': rms('s_r2'), 'reff_rms': rms('s_reff2'),
        }
        out['ombg_over_r'] = out['ombg_rms'] / out['r_rms']
        out['ombg_over_reff'] = out['ombg_rms'] / out['reff_rms']
    return out
