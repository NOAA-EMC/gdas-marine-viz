"""Verification of surface state fields against independent gridded analyses.

The rest of the suite scores the DA in observation space and against itself.
This scores it against three daily L4 products that sit outside the system:

    adt   CMEMS SEALEVEL NRT L4      sea surface height
    sss   CMEMS SSS daily            sea surface salinity
    sst   OSTIA L4 SSTfnd            sea surface temperature

Two of them are genuinely independent of this DA: the assimilated observation
list holds ice concentration, Argo profiles, drifter SST and AVHRR/VIIRS L3U
SST -- no altimetry and no satellite salinity. OSTIA is built from the same
AVHRR and VIIRS radiances the system assimilates, so SST is a consistency
check rather than independent validation. The number is reported the same way;
read it knowing that.

OSTIA's analysed_sst is a FOUNDATION temperature -- below the diurnal warm
layer, effectively a settled value rather than one tied to a specific hour,
which this module treats as valid at 12Z for the day. The model side of the
comparison is built to match: instead of the single background/analysis
instant at the cycle, it is the per-cell mean of Temp level 0 over that
day's four synoptic cycles (00/06/12/18Z) -- a 24h window centred on 12Z,
using the cycle-frequency snapshots this suite has rather than reading
separate sub-daily history output. PRODUCTS['sst']['only_hour'] then
restricts the comparison itself to the 12Z cycle, since OSTIA is a once-daily
product and the other three cycles would just repeat the same day's score.
See _daily_mean() below.

Every product is a regular lat/lon grid on -180..180, which is what
`Grid.lon180` already provides, so a model point is placed in a product by
index arithmetic rather than by interpolation: no KD-tree and no scipy (which
is optional here). The result lands on the model grid, so the existing
area-weighted reducers, region masks and map builders all apply unchanged.
"""

import glob
import os

import numpy as np
from netCDF4 import Dataset

# Per product: what to read and how it differs from the model. The base
# directory is NOT here -- the three products come from unrelated archives, so
# each one's path is given in the config (see `product_path`).
#
#   pattern       date-templated glob BELOW the configured path; the filenames
#                 carry a production date that cannot be predicted, hence a
#                 glob rather than a format string. Overridable per product.
#   idx           leading index into the variable (time, or time+depth)
#   offset        added to the product to reach the model's units
#   ice/ice_max   product's own ice field, and the value above which a point
#                 is dropped
#   remove_mean   subtract each field's mean over the SHARED valid points --
#                 ADT is referenced to a mean dynamic topography and the model
#                 to its own geoid, so the raw difference is dominated by a
#                 constant offset that says nothing about the ocean
PRODUCTS = {
    'adt': dict(
        pattern='{Y}/{m}/nrt_global_allsat_phy_l4_{Ymd}_*.nc',
        var='adt', lat='latitude', lon='longitude', idx=(0,),
        model_var='ave_ssh', units='m', label='ADT (CMEMS)',
        ice='flag_ice', ice_max=0.5, err='err_sla', remove_mean=True),
    'sss': dict(
        pattern='{Y}/{m}/dataset-sss-ssd-nrt-daily_{Ymd}T*.nc',
        var='sos', lat='lat', lon='lon', idx=(0, 0),
        model_var='Salt', units='psu', label='SSS (CMEMS)',
        ice='sea_ice_fraction', ice_max=15.0, err='sos_error'),
    'sst': dict(
        pattern='{Y}/{m}/{Ymd}*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-*.nc',
        var='analysed_sst', lat='lat', lon='lon', idx=(0,),
        model_var='Temp', units='degC', label='SST (OSTIA)',
        offset=-273.15, mask='mask', err='analysis_error',
        # foundation temperature: the model side is a 24h-centred daily mean
        # (see _daily_mean and the module docstring), scored once a day at
        # the 12Z cycle rather than at every cycle.
        daily_mean=True, only_hour='12'),
}

STATES = ('bkg', 'ana')


def configured(cfg):
    """{name: {path, pattern}} for every product the config asks for.

    `products:` maps a product name to the directory holding it -- everything
    above the YYYY/MM/file layout -- because the three come from unrelated
    archives and share no common root. A product may instead give a mapping
    with `path` and an optional `pattern`, for an archive that names its files
    differently from the default in PRODUCTS.
    """
    out = {}
    for name, entry in ((cfg.get('verification') or {}).get('products')
                        or {}).items():
        spec = PRODUCTS.get(name)
        if spec is None:
            continue
        if isinstance(entry, str):
            entry = {'path': entry}
        path = (entry or {}).get('path')
        if not path:
            continue
        out[name] = {'path': path,
                     'pattern': entry.get('pattern') or spec['pattern']}
    return out


def product_path(cfg, name, cycle):
    """The product file for this cycle, or None."""
    e = configured(cfg).get(name)
    if e is None:
        return None
    pat = e['pattern'].format(Y=cycle[:4], m=cycle[4:6], Ymd=cycle[:8])
    hits = sorted(glob.glob(os.path.join(e['path'], pat)))
    return hits[0] if hits else None


def _axis(ds, name):
    return np.asarray(ds[name][:], dtype='f8')


def _spacing(a):
    """Grid spacing from the endpoints.

    Not from ``a[1] - a[0]``, and never from an equality test: OSTIA's axes are
    float32 and drift up to 1.2e-5 deg from a perfect 0.05 deg grid, which is
    1.4 m at the equator against a 5.5 km cell. Endpoint-derived spacing is
    exact for a uniform axis and robust to that rounding.
    """
    return (a[-1] - a[0]) / (a.size - 1)


def sample_to_grid(grid, arr, lat, lon, half=0):
    """Place a regular lat/lon field on the model grid by index arithmetic.

    ``half`` averages a (2*half+1)^2 window, for products finer than the model
    cell: sampling a 0.05 deg product at a 0.25 deg model point otherwise
    carries the product's sub-grid noise into the score. Longitude is periodic;
    latitudes outside the product's range come back NaN rather than clamped to
    the edge row, which would silently fabricate polar values.
    """
    dla, dlo = _spacing(lat), _spacing(lon)
    j = np.rint((grid.lat - lat[0]) / dla).astype(int)
    i = np.rint((grid.lon180 - lon[0]) / dlo).astype(int)
    inside = (j >= 0) & (j < lat.size)
    j = np.clip(j, 0, lat.size - 1)
    i = np.mod(i, lon.size)
    if half <= 0:
        out = arr[j, i].astype('f8', copy=True)
    else:
        acc = np.zeros(grid.shape)
        cnt = np.zeros(grid.shape)
        for dj in range(-half, half + 1):
            jj = np.clip(j + dj, 0, lat.size - 1)
            for di in range(-half, half + 1):
                v = arr[jj, np.mod(i + di, lon.size)]
                ok = np.isfinite(v)
                acc[ok] += v[ok]
                cnt[ok] += 1
        with np.errstate(invalid='ignore'):
            out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    out[~inside] = np.nan
    return out


def _window(grid, lon):
    """Half-width that averages a product cell up to the model cell size.

    Takes the model cell as 360/nx degrees of longitude. That holds for the
    tripolar grid away from the Arctic bipolar cap, where the columns converge
    and the real cell is smaller; there the window is a little wide, which
    smooths slightly more than necessary rather than aliasing.
    """
    model_dlon = 360.0 / grid.shape[1]
    return max(0, int(model_dlon / abs(_spacing(lon)) / 2))


def _read(ds, name, idx):
    a = ds[name][idx + (slice(None), slice(None))]
    return np.ma.filled(a.astype('f8'), np.nan)


def _level0(path, var):
    """Level 0 (surface) of ``var`` from ``path``, or None.

    Deliberately not lv_statespace.surface_state: lv_statespace imports this
    module (compute() calls verify()), so importing it back here would be
    circular. This is the same read _read_level does there, 2-D case only,
    which is all _daily_mean needs.
    """
    if path is None:
        return None
    with Dataset(path) as ds:
        if var not in ds.variables:
            return None
        v = ds[var]
        a = v[0, 0] if v.ndim == 4 else v[0]
        return np.ma.filled(a.astype('f8'), np.nan)


def _daily_mean(exp, day, var, state):
    """Per-cell mean of ``var`` level 0 over one calendar day's synoptic
    cycles (00/06/12/18Z) -- a 24h window centred on the 12Z cycle.

    Stands in for reading sub-daily history output, which this suite does
    not keep: the day's own cycle-frequency snapshots are the finest time
    resolution available. A cell is left out of a cycle's contribution only
    where that cycle's field is itself NaN there (or the cycle has no
    background/increment at all, e.g. a gap in the archive); it is NaN in
    the result only where every cycle in the window was.

    ``state`` is 'bkg' (the background alone) or 'ana' (background +
    increment, matching surface_state's reconstruction of the analysis --
    see its docstring for why that assumption is fine here too).
    """
    acc = cnt = None
    for hh in ('00', '06', '12', '18'):
        cycle = day + hh
        fld = _level0(exp.background(cycle, 'ocean'), var)
        if fld is not None and state == 'ana':
            incr = _level0(exp.increment(cycle, 'ocean'), var)
            fld = None if incr is None else fld + incr
        if fld is None:
            continue
        if acc is None:
            acc = np.zeros_like(fld)
            cnt = np.zeros_like(fld)
        ok = np.isfinite(fld)
        acc[ok] += fld[ok]
        cnt[ok] += 1
    if acc is None or not np.any(cnt > 0):
        return None
    with np.errstate(invalid='ignore'):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def load_product(grid, path, spec):
    """Product value, its error estimate and its valid mask, on the model grid."""
    with Dataset(path) as ds:
        lat, lon = _axis(ds, spec['lat']), _axis(ds, spec['lon'])
        half = _window(grid, lon)
        val = sample_to_grid(grid, _read(ds, spec['var'], spec['idx']),
                             lat, lon, half)
        err = (sample_to_grid(grid, _read(ds, spec['err'], spec['idx']),
                              lat, lon, half)
               if spec.get('err') in ds.variables else None)
        keep = np.isfinite(val)
        if spec.get('ice') in ds.variables:
            # nearest, not averaged: a fractional ice value straddling the
            # threshold should not admit the cell
            ice = sample_to_grid(grid, _read(ds, spec['ice'], spec['idx']),
                                 lat, lon, 0)
            # A fill value means "no ice information", not "ice". Requiring a
            # finite flag dropped 1941 of 967789 wet cells that carried a
            # perfectly good ADT, for no reason.
            keep &= ~(np.isfinite(ice) & (ice > spec['ice_max']))
        if spec.get('mask') in ds.variables:
            # GHRSST: bit 1 water, 2 land, 4 lake, 8 ice
            m = sample_to_grid(grid, _read(ds, spec['mask'], spec['idx']),
                               lat, lon, 0)
            mi = np.where(np.isfinite(m), m, 0).astype(int)
            keep &= ((mi & 1) > 0) & ((mi & 8) == 0)
    return val + spec.get('offset', 0.0), err, keep


def _stats(grid, diff, sel, err=None):
    n = int(np.count_nonzero(sel))
    if not n:
        return None
    out = {'n': n,
           'bias': grid.wmean(diff, sel),
           'rms': grid.wrms(diff, sel),
           # RMS is a poor summary where a handful of cells dominate: for SSS
           # the median |difference| is 0.15 psu while the RMS is 0.69, carried
           # by river plumes a quarter-degree model cannot resolve. The
           # percentile says which of the two is being read -- area-weighted,
           # like the two above it, or it would not be comparable with them.
           'p90': grid.wpercentile(np.abs(diff), 90, sel)}
    if err is not None:
        m = grid.wmean(err, sel & np.isfinite(err))
        if np.isfinite(m):
            out['obs_err'] = m
    return out


def verify(grid, exp, cycle, cfg, regions, model_level):
    """Score this experiment's background and analysis against each product.

    ``model_level`` is a callable (variable, state) -> 2-D field or None, so
    this module does not need to know how a state is assembled.

    Returns {product: {state: {region: {...}}}} and the maps.

    Three map kinds per product, all carrying the same treatment the score
    does -- masked to the shared valid points, and for ADT with the mean
    removed:

        <product>_obs_k0          the product
        <product>_<state>_k0      the model field
        <product>_<state>_diff_k0 model minus product

    Both the fields and their difference are kept because they answer different
    questions: the fields show whether the model reproduces the product at all,
    and at global scale a 0.45 degC error is invisible against a 0..30 degC
    ramp, so the difference is the only place the error is legible.
    """
    scores, maps = {}, {}
    for name, entry in configured(cfg).items():
        spec = PRODUCTS.get(name)
        if spec is not None and spec.get('only_hour') \
                and cycle[8:10] != spec['only_hour']:
            # Restricted to one cycle hour by design (see PRODUCTS), not a
            # missing file -- routine at every other cycle, so quiet rather
            # than reported like an actual miss below.
            continue
        path = product_path(cfg, name, cycle)
        if spec is None or path is None:
            # Say so. A configured product that resolves to nothing used to be
            # skipped in silence, so the report simply came out without the
            # section and the mistake -- usually a path that does not exist --
            # was only discovered by noticing an absent figure.
            print('  ! verification %s: no file for %s under %s (pattern %s)'
                  % (name, cycle, entry['path'], entry['pattern']))
            continue
        obs, err, keep = load_product(grid, path, spec)
        per_state = {}
        for state in STATES:
            if spec.get('daily_mean'):
                fld = _daily_mean(exp, cycle[:8], spec['model_var'], state)
            else:
                fld = model_level(spec['model_var'], state)
            if fld is None:
                continue
            sel = grid.mask & keep & np.isfinite(fld)
            if not np.any(sel):
                continue
            if spec.get('remove_mean'):
                # Both means over the SHARED points. ADT is only ~66% finite,
                # and its mean over its own domain (+0.300 m here) differs from
                # its mean over the points the model also has (+0.569 m) by
                # more than half the signal being measured.
                fld = fld - grid.wmean(fld, sel)
                obs_s = obs - grid.wmean(obs, sel)
            else:
                obs_s = obs
            diff = fld - obs_s
            maps['%s_%s_k0' % (name, state)] = np.where(sel, fld, np.nan)
            maps['%s_%s_diff_k0' % (name, state)] = np.where(sel, diff, np.nan)
            # The product is one field, not one per state. Keep the LAST
            # state's version rather than setdefault's first: the two differ
            # only where one state has a value the other lacks, and for ADT in
            # the mean removed, and taking whichever happened to run first
            # made the panel silently experiment-dependent. Either way it is
            # the product masked to the points actually compared, which is
            # what the figure caption says.
            maps['%s_obs_k0' % name] = np.where(sel, obs_s, np.nan)
            per_state[state] = {r: s for r, s in
                                ((r, _stats(grid, diff, sel & m, err))
                                 for r, m in regions.items()) if s}
        if per_state:
            scores[name] = per_state
            scores[name]['file'] = os.path.basename(path)
    return scores, maps
