"""State-space metrics for the LETKF verification suite.

The ocean files are ~4.5 GB each, so every 3-D field is read one level at a
time (12 MB per read) and reduced immediately. Nothing here ever holds a whole
3-D field in memory.
"""

import contextlib
import os

import numpy as np
from netCDF4 import Dataset

import lv_common
import lv_verif
import lv_woa

EARTH_R = 6371.0e3


def _levels(ds, var):
    v = ds[var]
    return v.shape[1] if v.ndim == 4 else 1


def _read_level(ds, var, k):
    """One level as float64 with missing values as NaN.

    np.asarray() on a MaskedArray silently DROPS the mask and hands back the
    raw fill values, so the conversion must happen on the masked array itself.
    The increment and variance files carry no _FillValue and were unaffected,
    but the MOM6 and CICE history files fill land with 1e30.
    """
    v = ds[var]
    a = v[0, k] if v.ndim == 4 else v[0]
    return np.ma.filled(a.astype('f8'), np.nan)


def _sd_from_variance(grid, field, sel=None):
    """Area-weighted RMS standard deviation from a variance field.

    The ensemble files hold variance; the RMS standard deviation is the
    area-weighted sqrt of the mean variance, which is the quantity that pairs
    with an RMS departure.
    """
    m = grid.wmean(field, sel)
    return float(np.sqrt(max(m, 0.0))) if np.isfinite(m) else np.nan


def regional_spread_profiles(grid, prior, post, an, variables, regions):
    """Per-region ensemble spread at each stage of the analysis, and the ratios.

    Returns {region: {var: {'prior', 'post', 'ratio', ...}}}, where each value
    is a list over levels. The 'global' region reproduces the whole-domain
    profiles, so callers take them from here rather than reading the multi-GB
    variance files again.

    Three stages, when the post-inflation file is available:

        prior      sigma_b     background
        post       sigma_a     analysis, before inflation
        an         sigma_an    analysis, after inflation
        ratio      sigma_a  / sigma_b    what the update removed
        ratio_net  sigma_an / sigma_b    what actually propagates
        infl       sigma_an / sigma_a    what inflation put back

    `an`, `ratio_net` and `infl` are absent for a variable the post-inflation
    file does not carry (it holds no `ave_ssh`), and for every variable when
    there is no such file at all -- so an older cache keeps its exact meaning.

    All stages are read level-synchronously. `ratio` is formed only over cells
    where prior and post are both present and the prior variance is positive;
    the two new ratios additionally require the inflated variance and a
    positive posterior, since sigma_a is `infl`'s denominator. That slightly
    different footing is why `ratio_net` is computed rather than derived as
    `ratio * infl` -- the two masks are not identical, and multiplying them
    would quietly mix cell populations.
    """
    out = {r: {} for r in regions}
    if prior is None or post is None:
        return out
    da = Dataset(an) if an is not None else None
    try:
        with Dataset(prior) as dp, Dataset(post) as dq:
            for var in variables:
                if var not in dp.variables or var not in dq.variables:
                    continue
                has_an = da is not None and var in da.variables
                nk = _levels(dp, var)
                keys = ['prior', 'post', 'ratio']
                if has_an:
                    keys += ['an', 'ratio_net', 'infl']
                acc = {r: {k: [] for k in keys} for r in regions}
                for k in range(nk):
                    p = _read_level(dp, var, k)
                    q = _read_level(dq, var, k)
                    ok = np.isfinite(p) & np.isfinite(q) & (p > 0)
                    s = _read_level(da, var, k) if has_an else None
                    if has_an:
                        ok3 = ok & np.isfinite(s) & (s >= 0) & (q > 0)
                    for rname, rmask in regions.items():
                        a = acc[rname]
                        a['prior'].append(_sd_from_variance(grid, p, rmask))
                        a['post'].append(_sd_from_variance(grid, q, rmask))
                        both = rmask & ok
                        sp = _sd_from_variance(grid, p, both)
                        sq = _sd_from_variance(grid, q, both)
                        a['ratio'].append(float(sq / sp) if sp > 0 else np.nan)
                        if not has_an:
                            continue
                        tri = rmask & ok3
                        a['an'].append(_sd_from_variance(grid, s, rmask))
                        sb = _sd_from_variance(grid, p, tri)
                        sa = _sd_from_variance(grid, q, tri)
                        sn = _sd_from_variance(grid, s, tri)
                        a['ratio_net'].append(
                            float(sn / sb) if sb > 0 else np.nan)
                        a['infl'].append(
                            float(sn / sa) if sa > 0 else np.nan)
                for rname in regions:
                    out[rname][var] = acc[rname]
    finally:
        if da is not None:
            da.close()
    return out


# --------------------------------------------------------------------------
# background / first guess
# --------------------------------------------------------------------------

def _faces_to_centre(a, b):
    """Average two cell faces onto the tracer point, ignoring land.

    Land is NaN here (MOM6 writes _FillValue = 0 for velocity), so a coastal
    cell with one wet face keeps that face's value rather than going missing.
    """
    fa, fb = np.isfinite(a), np.isfinite(b)
    n = fa.astype('f8') + fb.astype('f8')
    s = np.where(fa, a, 0.0) + np.where(fb, b, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s / n, np.nan)


def _uv_at_centre(ds, k):
    """The current components brought onto the tracer point.

    Used only by `speed`, which combines the two components and so needs them
    at the same place: MOM6 history writes u on the eastern cell face (xq) and
    v on the northern face (yq), and |(u, v)| taken across that offset is not
    a speed anywhere. soca's ensemble mean, the increments and the ensemble
    variances write both at tracer points already; the two conventions are
    told apart by dimension name rather than by experiment kind.

    Plotting `u` or `v` on their own does NOT come through here -- those are
    read exactly as the file stores them.
    """
    u = _read_level(ds, 'u', k)
    v = _read_level(ds, 'v', k)
    if ds['u'].dimensions[-1] != ds['v'].dimensions[-1]:
        # i is periodic, so the western face of column 0 is the last column;
        # the southern row has no face below it and keeps its own value.
        u = _faces_to_centre(np.roll(u, 1, axis=1), u)
        v = _faces_to_centre(np.vstack([v[:1], v[:-1]]), v)
    return u, v


def _speed_level(ds, k):
    """Current speed |(u, v)| at tracer points."""
    u, v = _uv_at_centre(ds, k)
    return np.hypot(u, v)


# Fields the files do not store directly.
#
# `u` and `v` are deliberately NOT here: asked for by name they are read
# straight off the file, exactly as stored, with no interpolation. That means
# MOM6 history plots them on their native faces (u on xq, v on yq) rather than
# at the tracer point -- a half-cell offset against a collocated file, ~14 km
# at the equator, which is below what a global map resolves. `speed` still
# collocates internally because |(u, v)| is meaningless with the two
# components taken half a cell apart.
#   canonical name -> (variable to take the level count from,
#                      variables that must be present, level reader)
DERIVED = {
    'speed': ('u', ('u', 'v'), _speed_level),
}

# Fallback for the ocean background when no ocean/history was archived at
# all: CICE's own history mirrors the ocean surface state it received from
# the coupler as sst_h (degC) / sss_h (ppt), which stand in for level-0
# Temp/Salt on the same grid. Same (src, divisor) shape as BKG_VARMAP in
# lv_common, so _resolve/_mapped_level need no change to read it. See
# compute()'s background block for where this is actually used -- only
# when the real ocean background is absent, and only for these two vars.
ICE_OCEAN_FALLBACK_VARMAP = {'Temp': ('sst_h', None), 'Salt': ('sss_h', None)}


def _resolve(ds, varmap, var):
    """Variable to size ``var`` from, or None when the file cannot supply it."""
    if var in DERIVED:
        proxy, need, _fn = DERIVED[var]
        return proxy if all(n in ds.variables for n in need) else None
    src = (varmap or {}).get(var, (var, None))[0]
    return src if src in ds.variables else None


def _mapped_level(ds, varmap, var, k):
    """Read one level of ``var``, applying the kind's variable mapping.

    CICE history stores grid-cell-mean thickness; the analysis stores thickness
    per unit ice area. Where the map says so, divide by concentration.
    """
    if var in DERIVED:
        _proxy, need, fn = DERIVED[var]
        return fn(ds, k) if all(n in ds.variables for n in need) else None
    src, div = (varmap or {}).get(var, (var, None))
    if src not in ds.variables:
        return None
    f = _read_level(ds, src, k)
    if div is not None:
        if div not in ds.variables:
            return None
        d = _read_level(ds, div, k)
        with np.errstate(invalid='ignore', divide='ignore'):
            f = np.where(d > 0, f / np.where(d > 0, d, 1.0), np.nan)
    return f


def depth_from_h(grid, path, stride=4):
    """Mid-depth of each model level, from the background layer thickness.

    The increment and variance files carry only a level index, so this is the
    only honest depth axis. Uses the deep-column median of cumsum(h) - h/2;
    a stride-4 subsample is ample for a global profile and avoids reading the
    full 3-D field at full resolution.
    """
    if path is None:
        return None
    with Dataset(path) as ds:
        if 'h' not in ds.variables:
            return None
        nk = _levels(ds, 'h')
        wet = grid.mask[::stride, ::stride]
        prof = np.empty(nk)
        for k in range(nk):
            f = _read_level(ds, 'h', k)[::stride, ::stride]
            f = f[wet & np.isfinite(f) & (f > 0)]
            prof[k] = np.median(f) if f.size else np.nan
    prof = np.where(np.isfinite(prof), prof, 0.0)
    return (np.cumsum(prof) - prof / 2.0).tolist()


H_MIN = 1e-3        # metres; thinner than this is a vanished layer


def _wet_level(ds, k, nk):
    """Cells that hold water at level ``k``, from the layer thickness.

    Below the sea floor MOM6 collapses layers to zero thickness and fills the
    tracers with zeros. Those cells are not ocean: averaging over them drags
    the deep mean salinity toward zero and destroys the colour range on deep
    maps, so every 3-D background field is masked by h.
    """
    if nk <= 1 or 'h' not in ds.variables:
        return None
    h = _read_level(ds, 'h', k)
    return np.isfinite(h) & (h > H_MIN)


# --------------------------------------------------------------------------
# level sources
#
# The reducers below used to take a file path and open it themselves, which is
# why a field held in memory -- the WOA climatology -- could not go through
# them. They take a source instead, and open_source() turns a path back into
# one, so every existing caller is unchanged. The point of the indirection is
# DiffSource: a model-minus-climatology field then needs no reducer of its own.
# --------------------------------------------------------------------------

class LevelSource:
    """A field seen one level at a time, plus a vanished-layer test."""

    def levels(self, var):
        """How many levels ``var`` has here, or None if it has none."""
        raise NotImplementedError

    def level(self, var, k):
        """Level ``k`` of ``var`` as a 2-D array, or None."""
        raise NotImplementedError

    def wet(self, k, nk):
        """Cells holding water at level ``k``, or None when unknowable."""
        return None


class FileSource(LevelSource):
    """One open Dataset, with the experiment kind's variable mapping.

    Exactly what the reducers did inline: _resolve to size a variable,
    _mapped_level to read it, _wet_level for the vanished-layer test.
    """

    def __init__(self, ds, varmap=None):
        self.ds = ds
        self.varmap = varmap

    def levels(self, var):
        src = _resolve(self.ds, self.varmap, var)
        return None if src is None else _levels(self.ds, src)

    def level(self, var, k):
        return _mapped_level(self.ds, self.varmap, var, k)

    def wet(self, k, nk):
        return _wet_level(self.ds, k, nk)


class DiffSource(LevelSource):
    """``a`` minus ``b``, level by level, carrying ``a``'s wetness.

    This is the reason the reducers take a source rather than a path. A
    model-minus-climatology field goes through the very same profile, map and
    section reducers the model itself does, so a bias profile is guaranteed to
    be the difference of the two profiles beside it rather than a separately
    coded near-miss.
    """

    def __init__(self, a, b):
        self.a, self.b = a, b

    def levels(self, var):
        na, nb = self.a.levels(var), self.b.levels(var)
        return None if na is None or nb is None else min(na, nb)

    def level(self, var, k):
        fa = self.a.level(var, k)
        if fa is None:
            return None
        fb = self.b.level(var, k)
        return None if fb is None else fa - fb

    def wet(self, k, nk):
        return self.a.wet(k, nk)


@contextlib.contextmanager
def open_source(src, varmap=None):
    """A path, or an already-built LevelSource, as a LevelSource.

    A path is opened and closed here; a source is passed straight through,
    since its lifetime belongs to whoever built it.

    Tested by shape rather than isinstance so that lv_woa.WoaSource need not
    subclass LevelSource: this module imports lv_woa, so lv_woa importing this
    one back would be a cycle. Nothing else reaching these reducers carries a
    `level` attribute -- a path is a string and an absent field is None.
    """
    if src is None:
        yield None
    elif hasattr(src, 'level'):
        yield src
    else:
        with Dataset(src) as ds:
            yield FileSource(ds, varmap)


def regional_profiles(grid, src, variables, regions, varmap=None,
                      reducer='mean', wet_mask=False):
    """Per-region profiles of a field, with the spatial spread in each region.

    Returns {region: {var: {'mean': [...], 'sd': [...]}}}; the 'global' region
    reproduces the whole-domain profile, so callers take it from here rather
    than reading the file a second time.

    ``src`` is a path or a LevelSource (see open_source).

    ``wet_mask`` drops vanished layers using h and must be set ONLY for
    background files. An increment file's 'h' is an increment of thickness,
    not a thickness, and using it as a wetness test silently masks most of the
    ocean.
    """
    out = {r: {} for r in regions}
    if src is None:
        return out
    with open_source(src, varmap) as s:
        for var in variables:
            nk = s.levels(var)
            if nk is None:
                continue
            acc = {r: {'mean': [], 'sd': []} for r in regions}
            for k in range(nk):
                f = s.level(var, k)
                wet = s.wet(k, nk) if wet_mask else None
                base = np.isfinite(f) if f is not None else None
                if base is not None and wet is not None:
                    base = base & wet
                for rname, rmask in regions.items():
                    if f is None:
                        acc[rname]['mean'].append(np.nan)
                        acc[rname]['sd'].append(np.nan)
                        continue
                    sel = rmask if base is None else (rmask & base)
                    if reducer == 'rms':
                        # An increment field has a near-zero mean, so its own
                        # standard deviation just restates the RMS. The useful
                        # spread is in the magnitude: how much the size of the
                        # increment varies from column to column.
                        acc[rname]['mean'].append(grid.wrms(f, sel))
                        acc[rname]['sd'].append(grid.wsd(np.abs(f), sel))
                    else:
                        acc[rname]['mean'].append(grid.wmean(f, sel))
                        acc[rname]['sd'].append(grid.wsd(f, sel))
            for rname in regions:
                out[rname][var] = acc[rname]
    return out


def background_maps(grid, src, variables, levels, varmap, stride=2):
    """Subsampled background slices for plotting, keyed '<var>_k<level>'."""
    out = {}
    if src is None:
        return out
    with open_source(src, varmap) as s:
        for var in variables:
            nk = s.levels(var)
            if nk is None:
                continue
            ks = [k for k in levels if k < nk] if nk > 1 else [0]
            for k in ks:
                f = s.level(var, k)
                if f is None:
                    continue
                wet = s.wet(k, nk)
                keep = grid.mask if wet is None else (grid.mask & wet)
                f = np.where(keep, f, np.nan)
                out['%s_k%d' % (var, k)] = f[::stride, ::stride].astype('f4')
    return out


def ice_totals(grid, path, varmap, thresh=0.15):
    """Sea-ice extent, area and volume per hemisphere, from the background.

    extent = area where concentration exceeds ``thresh``; area = area-weighted
    concentration; volume = area x concentration x per-ice thickness. Reported
    in 1e6 km^2 and 1e3 km^3, the conventional units.
    """
    out = {}
    if path is None:
        return out
    with Dataset(path) as ds:
        a = _mapped_level(ds, varmap, 'aice_h', 0)
        if a is None:
            return out
        hi = _mapped_level(ds, varmap, 'hi_div_aice_h', 0)
        for hemi, sel in (('nh', grid.lat > 0), ('sh', grid.lat < 0)):
            w = grid.mask & sel & np.isfinite(a)
            ar = grid.area
            out['extent_%s' % hemi] = float(
                np.sum(ar[w & (a > thresh)]) / 1e12)
            out['area_%s' % hemi] = float(np.sum(ar[w] * a[w]) / 1e12)
            if hi is not None:
                v = w & np.isfinite(hi)
                out['volume_%s' % hemi] = float(
                    np.sum(ar[v] * a[v] * hi[v]) / 1e12)
    return out


# --------------------------------------------------------------------------
# maps
# --------------------------------------------------------------------------

def map_fields(grid, src, variables, levels, stride=2):
    """Subsampled 2-D slices for plotting, keyed '<var>_k<level>'."""
    out = {}
    if src is None:
        return out
    with open_source(src) as s:
        for var in variables:
            nk = s.levels(var)
            if nk is None:
                continue
            ks = [k for k in levels if k < nk] if nk > 1 else [0]
            for k in ks:
                f = s.level(var, k)
                if f is None:
                    continue
                f = np.where(grid.mask, f, np.nan)
                out['%s_k%d' % (var, k)] = f[::stride, ::stride].astype('f4')
    return out


def spread_reduction_maps(grid, prior, post, variables, levels, stride=2):
    """1 - sigma_a/sigma_b: where, and how far, the analysis reduced spread.

    A localization that is too broad shows here as reduction reaching far from
    any observation; one that is too tight shows as isolated dimples.
    """
    out = {}
    if prior is None or post is None:
        return out
    with Dataset(prior) as dp, Dataset(post) as dq:
        for var in variables:
            if var not in dp.variables or var not in dq.variables:
                continue
            nk = _levels(dp, var)
            ks = [k for k in levels if k < nk] if nk > 1 else [0]
            for k in ks:
                p = _read_level(dp, var, k)
                q = _read_level(dq, var, k)
                with np.errstate(invalid='ignore', divide='ignore'):
                    r = 1.0 - np.sqrt(q / p)
                r = np.where(grid.mask & (p > 0) & np.isfinite(r), r, np.nan)
                out['%s_k%d' % (var, k)] = r[::stride, ::stride].astype('f4')
    return out


def inflation_maps(grid, post, an, variables, levels, stride=2):
    """sigma_an/sigma_a: where, and how hard, inflation restored the spread.

    The plain ratio is cached rather than its logarithm, so the stored field
    stays a physical multiplier and the plot layer owns the choice of scale.
    Under RTPS the ratio is 1 wherever no observation reached the column --
    the analysis left the spread alone, so there is nothing to relax back --
    which makes this map a picture of observational reach as much as of the
    inflation itself.
    """
    out = {}
    if post is None or an is None:
        return out
    with Dataset(post) as dq, Dataset(an) as da:
        for var in variables:
            if var not in dq.variables or var not in da.variables:
                continue
            nk = _levels(dq, var)
            ks = [k for k in levels if k < nk] if nk > 1 else [0]
            for k in ks:
                q = _read_level(dq, var, k)
                s = _read_level(da, var, k)
                with np.errstate(invalid='ignore', divide='ignore'):
                    r = np.sqrt(s / q)
                r = np.where(grid.mask & (q > 0) & np.isfinite(r), r, np.nan)
                out['%s_k%d' % (var, k)] = r[::stride, ::stride].astype('f4')
    return out


# --------------------------------------------------------------------------
# vertical sections
# --------------------------------------------------------------------------

# A zonal transect is a single grid row, which is an honest constant-latitude
# line only where the tripolar rows are themselves constant latitude. Measured
# on the 1440x1080 gridspec: the latitude spread across a row is exactly zero
# from 78.6S to 64.4N, then grows fast as the rows bend around the two northern
# poles -- 0.6 deg at 64.5N, 6.5 deg by 68.8N, 13 deg by 73.5N. A "zonal
# section at 70N" is therefore not a latitude circle at all. Such a line is
# still extracted and drawn, but flagged by preflight.py, by the plot layer,
# and on the figure panel itself.
TRIPOLAR_LAT = 65.0


def section_warnings(cfg):
    """Configured zonal transects the tripolar grid cannot honour.

    Lives here rather than in the plot layer so preflight.py can report it
    before a long precompute, without importing matplotlib and cartopy.
    """
    out = []
    for t in ((cfg.get('sections') or {}).get('zonal') or []):
        if float(t) > TRIPOLAR_LAT:
            out.append(
                'zonal section at %g N is north of the tripolar seam (%g N): '
                'the grid rows bend there, so this transect is not a latitude '
                'circle. It is still drawn, and the panel says so.'
                % (float(t), TRIPOLAR_LAT))
    return out


def _cut(f, axis, idx, stride):
    """One grid row (zonal) or column (meridional), subsampled along it."""
    return f[idx, ::stride] if axis == 'lat' else f[::stride, idx]


def section_lines(grid, cfg, stride=2):
    """Resolve `sections:` to grid rows and columns.

    Returns [(key, axis, index, x, warn)] in config order. ``axis`` is 'lat'
    for a zonal transect (a grid row; x is longitude) or 'lon' for a
    meridional one (a grid column; x is latitude). ``warn`` marks a zonal
    line north of TRIPOLAR_LAT.

    The row/column is picked from the row-median latitude and column-median
    longitude rather than an argmin over the whole 2-D coordinate, which on a
    curvilinear grid can land on a neighbouring row entirely.
    """
    spec = cfg.get('sections') or {}
    rowlat = np.median(grid.lat, axis=1)
    collon = np.median(grid.lon, axis=0)
    out = []
    for target in (spec.get('zonal') or []):
        t = float(target)
        j = int(np.argmin(np.abs(rowlat - t)))
        out.append(('lat%+04d' % round(t), 'lat', j,
                    np.asarray(grid.lon[j, ::stride], dtype='f4'),
                    t > TRIPOLAR_LAT))
    for target in (spec.get('meridional') or []):
        t = float(target)
        # The gridspec longitude runs -300..60 and is monotonic along i, so a
        # target given in the conventional -180..180 may need a turn added or
        # removed to land inside it. Picking the nearest of the three
        # representations rather than the first one INSIDE the range matters
        # at the wrap boundary: 60E is the grid's own last column, and a
        # strict range test rejected both 60 and -300 (the column medians stop
        # just short of either end), silently dropping the line.
        cand = min((t, t - 360.0, t + 360.0),
                   key=lambda c: float(np.min(np.abs(collon - c))))
        i = int(np.argmin(np.abs(collon - cand)))
        out.append(('lon%+04d' % round(t), 'lon', i,
                    np.asarray(grid.lat[::stride, i], dtype='f4'), False))
    return out


def section_geometry(grid, hpath, lines, stride=2):
    """(axes, wet) for each transect, from the BACKGROUND layer thickness.

    ``axes`` is cached: {'x_<line>': 1-D, 'depth_<line>': (nk, nx)} -- the
    mid-depth of each cell along the transect. ``wet`` is not cached; it is
    the wetness mask the caller applies to an increment's own planes.

    Depth always comes from the background: an increment file's 'h' is an
    increment of thickness, not a thickness, so the increment sections borrow
    the background's depth axis. The two are the same grid at the same
    validity time.

    The depth plane is finite EVERYWHERE, including below the sea floor, where
    it advances by H_MIN per level instead of the vanished layer's zero. Two
    reasons: matplotlib refuses non-finite pcolormesh coordinates outright,
    and a coordinate that stops advancing produces degenerate quads. The
    bathymetry silhouette is drawn by the field planes being masked over those
    same cells, not by holes in the coordinate -- so the sub-floor rows
    collapse into a hair's breadth at the true floor depth and never show.
    """
    axes, wet = {}, {}
    if not lines:
        return axes, wet
    for key, _axis, _idx, x, _warn in lines:
        axes['x_%s' % key] = x
    if hpath is None:
        return axes, wet
    with Dataset(hpath) as ds:
        if 'h' not in ds.variables:
            return axes, wet
        nk = _levels(ds, 'h')
        if nk <= 1:
            return axes, wet
        planes, masks, run = {}, {}, {}
        for key, _a, _i, x, _w in lines:
            planes[key] = np.full((nk, x.size), np.nan, 'f4')
            masks[key] = np.zeros((nk, x.size), bool)
            run[key] = np.zeros(x.size)
        for k in range(nk):
            h = _read_level(ds, 'h', k)
            for key, axis, idx, _x, _w in lines:
                hl = _cut(h, axis, idx, stride)
                ok = np.isfinite(hl) & (hl > H_MIN)
                thick = np.where(ok, hl, H_MIN)
                planes[key][k] = run[key] + thick / 2.0
                run[key] = run[key] + thick
                masks[key][k] = ok
        for key in planes:
            axes['depth_%s' % key] = planes[key]
            wet[key] = masks[key]
    return axes, wet


def section_planes(grid, src, variables, lines, varmap=None, stride=2,
                   wet_mask=False):
    """Depth-vs-distance planes along each transect, keyed '<var>_<line>'.

    2-D fields (ave_ssh, MLD, every ice field) have no vertical section and
    are skipped. ``wet_mask`` drops vanished layers using h and must be set
    ONLY for background files, for the same reason regional_profiles() gives.
    """
    out = {}
    if src is None or not lines:
        return out
    with open_source(src, varmap) as s:
        for var in variables:
            nk = s.levels(var)
            if nk is None or nk <= 1:
                continue
            planes = {key: np.full((nk, x.size), np.nan, 'f4')
                      for key, _a, _i, x, _w in lines}
            for k in range(nk):
                f = s.level(var, k)
                if f is None:
                    break
                keep = grid.mask
                if wet_mask:
                    wet = s.wet(k, nk)
                    if wet is not None:
                        keep = keep & wet
                f = np.where(keep, f, np.nan)
                for key, axis, idx, _x, _w in lines:
                    planes[key][k] = _cut(f, axis, idx, stride)
            for key, plane in planes.items():
                out['%s_%s' % (var, key)] = plane
    return out


# --------------------------------------------------------------------------
# horizontal correlation length of the increment
# --------------------------------------------------------------------------

def _box_corr_length(field, wet, dx_km, dy_km, maxlag=60):
    """1/e length of the isotropic autocorrelation of ``field`` inside a box.

    Land is handled by normalising the FFT autocorrelation of the masked,
    demeaned field by the autocorrelation of the mask itself, which is the
    standard missing-data correction.
    """
    m = wet & np.isfinite(field)
    if np.count_nonzero(m) < 0.3 * m.size:
        return np.nan
    f = np.where(m, field - field[m].mean(), 0.0)
    w = m.astype('f8')

    ny, nx = f.shape
    sy, sx = 2 * ny, 2 * nx
    F = np.fft.rfft2(f, s=(sy, sx))
    W = np.fft.rfft2(w, s=(sy, sx))
    num = np.fft.irfft2(F * np.conj(F), s=(sy, sx))
    den = np.fft.irfft2(W * np.conj(W), s=(sy, sx))
    with np.errstate(invalid='ignore', divide='ignore'):
        c = np.where(den > 0.5, num / den, np.nan)
    if not np.isfinite(c[0, 0]) or c[0, 0] <= 0:
        return np.nan
    rho = c / c[0, 0]

    ly = np.minimum(maxlag, ny - 1)
    lx = np.minimum(maxlag, nx - 1)
    jj, ii = np.meshgrid(np.arange(ly + 1), np.arange(lx + 1), indexing='ij')
    d = np.hypot(jj * dy_km, ii * dx_km).ravel()
    r = rho[:ly + 1, :lx + 1].ravel()
    ok = np.isfinite(r)
    d, r = d[ok], r[ok]
    if d.size < 20:
        return np.nan

    order = np.argsort(d)
    d, r = d[order], r[order]
    nb = 40
    edges = np.linspace(0, d.max(), nb + 1)
    idx = np.clip(np.searchsorted(edges, d, side='right') - 1, 0, nb - 1)
    prof_d, prof_r = [], []
    for b in range(nb):
        s = idx == b
        if np.count_nonzero(s) < 3:
            continue
        prof_d.append(d[s].mean())
        prof_r.append(r[s].mean())
    prof_d, prof_r = np.array(prof_d), np.array(prof_r)
    below = np.where(prof_r < np.exp(-1.0))[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    r0, r1 = prof_r[i - 1], prof_r[i]
    d0, d1 = prof_d[i - 1], prof_d[i]
    if r0 == r1:
        return float(d0)
    return float(d0 + (np.exp(-1.0) - r0) * (d1 - d0) / (r1 - r0))


def correlation_lengths(grid, path, variables, levels, boxes):
    """Fit an increment correlation length in each box, per variable and level."""
    out = {}
    if path is None:
        return out
    with Dataset(path) as ds:
        for var in variables:
            if var not in ds.variables:
                continue
            nk = _levels(ds, var)
            ks = [k for k in levels if k < nk] if nk > 1 else [0]
            for k in ks:
                f = _read_level(ds, var, k)
                for box in boxes:
                    sel = ((grid.lat >= box['lat0']) & (grid.lat < box['lat1'])
                           & (grid.lon180 >= box['lon0'])
                           & (grid.lon180 < box['lon1']))
                    if np.count_nonzero(sel) < 400:
                        continue
                    jj, ii = np.where(sel)
                    j0, j1, i0, i1 = jj.min(), jj.max() + 1, ii.min(), ii.max() + 1
                    sub = f[j0:j1, i0:i1]
                    wet = grid.mask[j0:j1, i0:i1]
                    # local grid spacing, in km, from the geographic coordinates
                    latm = np.deg2rad(np.nanmean(grid.lat[j0:j1, i0:i1]))
                    dlon = np.abs(np.nanmean(np.diff(grid.lon[j0:j1, i0:i1], axis=1)))
                    dlat = np.abs(np.nanmean(np.diff(grid.lat[j0:j1, i0:i1], axis=0)))
                    dx = np.deg2rad(dlon) * EARTH_R * np.cos(latm) / 1000.0
                    dy = np.deg2rad(dlat) * EARTH_R / 1000.0
                    if not (np.isfinite(dx) and np.isfinite(dy)) or dx <= 0 or dy <= 0:
                        continue
                    L = _box_corr_length(sub, wet, dx, dy)
                    out.setdefault('%s_k%d' % (var, k), {})[box['name']] = L
    return out


def surface_state(bkg, incr):
    """Reader for one surface field of the background and of the analysis.

    Returns a callable (variable, 'bkg'|'ana') -> 2-D field or None. The
    analysis is reconstructed as background + increment: the DA writes an
    increment but no analysis state, and the two are the same shape, at tracer
    points, and valid at the same time (`BKG_DEFAULTS` resolves the 3DVar
    background to the previous cycle's f006 file, which is valid at the
    analysis time -- `lv_common --selftest` checks that).

    This assumes the analysis is exactly background + increment, with no
    balance or post-processing step in between.
    """
    cache = {}

    def read(path, var):
        if path is None:
            return None
        key = (path, var)
        if key not in cache:
            with Dataset(path) as ds:
                cache[key] = (_read_level(ds, var, 0)
                              if var in ds.variables else None)
        return cache[key]

    def get(var, state):
        b = read(bkg, var)
        if b is None or state == 'bkg':
            return b
        i = read(incr, var)
        return None if i is None else b + i

    return get


# --------------------------------------------------------------------------

def ice_area_increment(grid, path, var='aice_h'):
    """Area-integrated sea-ice-area increment, NH and SH, in 1e6 km^2.

    The analysis counterpart to the background ice totals: how much ice area
    the analysis step adds or removes in each hemisphere.
    """
    if path is None:
        return {}
    with Dataset(path) as ds:
        if var not in ds.variables:
            return {}
        f = _read_level(ds, var, 0)
    out = {}
    for name, sel in (('NH', grid.lat > 0), ('SH', grid.lat <= 0)):
        w = grid.mask & sel & np.isfinite(f)
        out[name] = float(np.sum(f[w] * grid.area[w]) / 1e12)
    return out


def woa_block(grid, cfg, cycle, bkg, exp, model_depth, regions, blevels,
              lines, stride, maps):
    """WOA climatology and the model-minus-WOA bias for the ocean background.

    Returns the JSON fragment and writes its maps and section planes straight
    into ``maps``. Every quantity here goes through the SAME reducers the model
    itself goes through -- see DiffSource -- so the bias profile is the
    difference of the two profiles beside it by construction, not by a second
    implementation that happens to agree.

    Needs the model depth axis, so it is called after depth_from_h. An
    experiment with no ocean background (or only the ice/history stand-in) has
    no axis and no 3-D field, and is skipped.
    """
    entry = lv_woa.configured(cfg)
    if entry is None or not model_depth:
        return {}
    wvars = [v for v in (cfg.get('background_vars') or {}).get('ocean', [])
             if v in lv_woa.VARS]
    if not wvars:
        return {}
    missing = lv_woa.missing_files(entry, wvars)
    if missing:
        print('\n  ! woa: %d expected file(s) absent, first %s -- skipped'
              % (len(missing), missing[0]), flush=True)
        return {}

    fields, wlat, wlon, info = lv_woa.build(entry, cycle, model_depth, wvars)
    out = {'has_woa': True, 'woa': info}
    with Dataset(bkg) as ds:
        bsrc = FileSource(ds, exp.varmap)
        # wetness always from the MODEL's own layer thickness: below the sea
        # floor MOM6 writes zeros while WOA still has a climatological value,
        # and differencing those fabricates a bias under the bathymetry.
        wsrc = lv_woa.WoaSource(fields, wlat, wlon, grid, wet_from=bsrc)
        dsrc = DiffSource(bsrc, wsrc)

        out['woa_region'] = regional_profiles(grid, wsrc, wvars, regions,
                                              wet_mask=True)
        out['woa_mean'] = {v: d['mean'] for v, d
                           in out['woa_region'].get('global', {}).items()}
        out['woa_bias_region'] = regional_profiles(grid, dsrc, wvars, regions,
                                                   wet_mask=True)
        out['woa_bias_mean'] = {
            v: d['mean']
            for v, d in out['woa_bias_region'].get('global', {}).items()}

        for pre, src in (('woa', wsrc), ('woa_bias', dsrc)):
            for k, v in background_maps(grid, src, wvars, blevels, None,
                                        stride).items():
                maps['ocean/%s/%s' % (pre, k)] = v
        if lines and entry.get('sections'):
            for pre, src in (('woa_sec', wsrc), ('woa_bias_sec', dsrc)):
                for k, v in section_planes(grid, src, wvars, lines,
                                           stride=stride,
                                           wet_mask=True).items():
                    maps['ocean/%s/%s' % (pre, k)] = v
    return out


def compute(grid, exp, cycle, cfg):
    """All state-space diagnostics for one experiment at one cycle."""
    svars = cfg.get('state_vars', {})
    levels = cfg.get('map_levels', [0])
    boxes = lv_common.corr_regions(cfg)
    stride = int(cfg.get('map_stride', 2))

    bvars = cfg.get('background_vars', {})
    blevels = cfg.get('background_levels', levels)
    regions = grid.regions(cfg)
    # Transects are ocean-only (an ice field has no vertical section) and are
    # resolved once: every configured line is cut out of the same level read.
    lines = section_lines(grid, cfg, stride)

    res, maps = {'map_stride': stride}, {}
    for realm in ('ocean', 'ice'):
        variables = svars.get(realm, [])
        if not variables:
            continue
        incr = exp.increment(cycle, realm)
        prior = exp.ensvar(cycle, realm, 'prior')
        post = exp.ensvar(cycle, realm, 'post')
        an = exp.ensvar(cycle, realm, 'an')
        bkg = exp.background(cycle, realm)

        r = {'has_increment': incr is not None,
             'has_ensvar': prior is not None and post is not None,
             # its own flag: 'has_ensvar' keeps meaning what it meant, so a
             # cache written before the post-inflation file existed still
             # reads correctly
             'has_an_ensvar': an is not None,
             'has_background': bkg is not None}

        # -- background --------------------------------------------------
        bv = bvars.get(realm, [])
        if bkg is not None and bv:
            # Hoisted above the reducers: the WOA climatology is placed on
            # this axis, so it has to exist before anything below runs. It is
            # a pure read of h and depends on nothing here.
            if realm == 'ocean':
                d = depth_from_h(grid, bkg)
                if d is not None:
                    res['depth'] = d
                    res['depth_source'] = 'background h (%s)' % (
                        os.path.basename(bkg))
            r['bkg_region'] = regional_profiles(grid, bkg, bv, regions,
                                                exp.varmap, wet_mask=True)
            r['bkg_mean'] = {v: d['mean']
                             for v, d in r['bkg_region'].get('global', {}).items()}
            r['bkg_file'] = os.path.basename(bkg)
            r['bkg_cycle'] = exp.background_cycle(cycle)
            if realm == 'ice':
                r['ice_totals'] = ice_totals(grid, bkg, exp.varmap)
            for k, v in background_maps(grid, bkg, bv, blevels, exp.varmap,
                                        stride).items():
                maps['%s/bkg/%s' % (realm, k)] = v
            if realm == 'ocean':
                r.update(woa_block(grid, cfg, cycle, bkg, exp, res.get('depth'),
                                   regions, blevels, lines, stride, maps))
        elif realm == 'ocean' and bv:
            # No ocean/history to read Temp/Salt from at all (e.g. 3dvar-rt,
            # which only archives ice/history) -- CICE's coupled history
            # mirrors the ocean surface state it received from the coupler
            # as sst_h/sss_h, a usable stand-in for level-0 Temp/Salt only.
            # Never used when a real ocean background exists above; MLD,
            # speed and every level below the surface still need the real
            # file and stay absent.
            fb_vars = [v for v in bv if v in ICE_OCEAN_FALLBACK_VARMAP]
            ice_bkg = exp.background(cycle, 'ice') if fb_vars else None
            if ice_bkg is not None:
                r['bkg_region'] = regional_profiles(
                    grid, ice_bkg, fb_vars, regions,
                    ICE_OCEAN_FALLBACK_VARMAP, wet_mask=False)
                r['bkg_mean'] = {v: d['mean'] for v, d in
                                r['bkg_region'].get('global', {}).items()}
                r['bkg_file'] = os.path.basename(ice_bkg)
                r['bkg_cycle'] = exp.background_cycle(cycle)
                r['bkg_fallback'] = 'ice_history_sst_sss'
                for k, v in background_maps(grid, ice_bkg, fb_vars, [0],
                                            ICE_OCEAN_FALLBACK_VARMAP,
                                            stride).items():
                    maps['%s/bkg/%s' % (realm, k)] = v
        r['incr_region'] = regional_profiles(grid, incr, variables, regions,
                                             reducer='rms')
        r['incr_rms'] = {v: d['mean']
                         for v, d in r['incr_region'].get('global', {}).items()}
        if prior is not None:
            # One pass over the variance files serves every region; the global
            # profiles are read straight back out of it.
            r['spread_region'] = regional_spread_profiles(grid, prior, post, an,
                                                          variables, regions)
            g = r['spread_region'].get('global', {})
            r['spread_prior'] = {v: d['prior'] for v, d in g.items()}
            r['spread_post'] = {v: d['post'] for v, d in g.items()}
            r['spread_ratio'] = {v: d['ratio'] for v, d in g.items()}
            # keyed only on the variables the post-inflation file carries
            for key, dest in (('an', 'spread_an'),
                              ('ratio_net', 'spread_ratio_net'),
                              ('infl', 'spread_infl')):
                got = {v: d[key] for v, d in g.items() if key in d}
                if got:
                    r[dest] = got
        if boxes:
            r['corr_length'] = correlation_lengths(grid, incr, variables,
                                                   levels, boxes)
        if realm == 'ice':
            r['ice_area_increment'] = ice_area_increment(grid, incr)
        res[realm] = r

        for k, v in map_fields(grid, incr, variables, levels, stride).items():
            maps['%s/incr/%s' % (realm, k)] = v

        # -- vertical sections ---------------------------------------------
        # Ocean only, and only for the 3-D fields; the depth axis for BOTH the
        # increment and the background sections comes from the background h
        # (see section_geometry). With no background there is no honest depth
        # axis, so only the x coordinate is written and the plot layer falls
        # back to the level index.
        if realm == 'ocean' and lines:
            axes, wet = section_geometry(grid, bkg, lines, stride)
            for k, v in axes.items():
                maps['ocean/sec/%s' % k] = v
            # The increment carries no usable thickness of its own, so it is
            # masked with the background's wetness -- without this the file's
            # zeros below the sea floor paint straight over the bathymetry.
            for k, v in section_planes(grid, incr, variables, lines,
                                       stride=stride).items():
                m = wet.get(k.rpartition('_')[2])
                maps['ocean/incr_sec/%s' % k] = (
                    v if m is None else np.where(m, v, np.nan).astype('f4'))
            if bkg is not None and bv:
                for k, v in section_planes(grid, bkg, bv, lines, exp.varmap,
                                           stride, wet_mask=True).items():
                    maps['ocean/bkg_sec/%s' % k] = v
        for k, v in spread_reduction_maps(grid, prior, post, variables,
                                          levels, stride).items():
            maps['%s/sprred/%s' % (realm, k)] = v
        for k, v in inflation_maps(grid, post, an, variables,
                                   levels, stride).items():
            maps['%s/infl/%s' % (realm, k)] = v

        # -- against independent gridded analyses --------------------------
        if realm == 'ocean' and (cfg.get('verification') or {}).get('products'):
            scores, vmaps = lv_verif.verify(
                grid, exp, cycle, cfg, regions, surface_state(bkg, incr))
            if scores:
                r['verif'] = scores
            for k, v in vmaps.items():
                maps['%s/verif/%s' % (realm, k)] = v[::stride, ::stride].astype('f4')
    return res, maps
