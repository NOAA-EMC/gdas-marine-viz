"""GREP monthly reanalysis ensemble, placed on the model grid.

GREP is the Copernicus multi-reanalysis ensemble: MONTHLY MEANS from three
independent ocean reanalyses (cglo, glor, oras) on a 1/4 deg regular grid,
one file per year, 2020-2024 only.

That makes it a different kind of reference from everything else in this
suite. The L4 products in lv_verif are same-day analyses of one variable at
the surface; WOA is a multi-decadal climatology. GREP is neither: it is three
contemporaneous, independent estimates of the full three-dimensional state,
averaged over the same month the model is averaged over. So a
model-minus-GREP difference can be read against the spread BETWEEN the
reanalyses -- a difference smaller than that spread is not evidence of a model
error, and one much larger is. No single-product comparison here offers that.

Two transformations, and unlike WOA neither is a conversion: GREP's `thetao`
is already potential temperature and `so` already practical salinity, so the
EOS-80 machinery lv_woa needs has no counterpart here.

    horizontal   nearest gather onto the tripolar grid, lv_verif.sample_to_grid
    vertical     onto the model's own level depths, lv_woa.level_weights

The heat and salt content integrals deliberately do NOT go through the
vertical interpolation: each side is integrated on its own native axis with
its own layer thicknesses (see content()), which is both cheaper and more
honest than integrating an interpolated column.

This module does not import lv_statespace -- that module may import this one,
the same arrangement lv_woa has -- and it does not import matplotlib, so
preflight.py can use it before a long run.
"""

import os
import re

import numpy as np
from netCDF4 import Dataset

import lv_verif
from lv_woa import level_weights

# The staged archive covers these years and no others. A month outside the
# range is skipped silently: there is no climatological fallback, because a
# 2020-2024 mean of one calendar month is a climatology and would quietly
# turn "how does this month compare with the other reanalyses" into a
# different question with a much larger error bar.
YEARS = (2020, 2024)

DEFAULT_PATTERN = 'GREP_{year}.nc'
DEFAULT_MEMBERS = ('cglo', 'glor', 'oras')
DEFAULT_BANDS = ((0.0, 300.0), (300.0, 1000.0))
DEFAULT_VARIABLES = ('Temp', 'Salt', 'u', 'v')
DEFAULT_MIN_CYCLES = 20
DEFAULT_MIN_COVERAGE = 0.5   # of the calendar month's cycles

# model variable -> GREP variable stem (the member is a suffix on the stem).
# GREP's uo/vo are eastward/northward at tracer points already; the model side
# reaches east/north through lv_statespace's DERIVED 'u'/'v', which centre the
# C-grid faces and apply the grid rotation. See the memory note on the
# tripolar fold: comparing a raw model `u` against `uo` would be wrong north
# of ~65N and subtly wrong elsewhere.
VARS = {'Temp': 'thetao', 'Salt': 'so', 'u': 'uo', 'v': 'vo'}

LABEL = 'GREP'

# Seawater constants for the content integrals. Boussinesq reference density
# and the heat capacity MOM6 itself uses, so the model-side integral is the
# model's own heat content rather than a differently constituted one.
RHO0 = 1026.0           # kg m-3
CP = 3996.0             # J kg-1 K-1

H_MIN = 1e-3            # metres; thinner than this is a vanished layer

# Steric sea level. Both sides go through ONE equation of state -- MOM6's own
# WRIGHT_FULL, the one this model runs with -- so a model-minus-GREP steric
# difference is a difference in T and S, not in thermodynamics. RHO_0 is the
# model's Boussinesq reference density (MOM_parameter_doc: RHO_0 = 1035), not
# RHO0 above; the two differ by 1%, which is immaterial for anomalies.
RHO_EOS = 1035.0        # kg m-3
GRAV = 9.8              # m s-2, for p = rho0 g z
T_REF, S_REF = 0.0, 35.0

# Wright (1997) full-range fit, coefficients copied from MOM6's
# src/equation_of_state/MOM_EOS_Wright_full.F90.
_WA = (7.133718e-4, 2.724670e-7, -1.646582e-7)
_WB = (5.613770e8, 3.600337e6, -3.727194e4, 1.660557e2, 6.844158e5,
       -8.389457e3)
_WC = (1.609893e5, 8.427815e2, -6.931554, 3.869318e-2, -1.664201e2,
       -2.765195)

_CYCLE_RE = re.compile(r'^\d{10}$')


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def configured(cfg):
    """The normalised `grep:` block, or None when the config does not ask.

    Absence of the block, or `enabled: false`, turns the whole comparison off
    -- the stage no-ops and the report section is omitted. A bare string is
    taken as the path, the same latitude lv_woa.configured allows.
    """
    entry = (cfg or {}).get('grep')
    if not entry:
        return None
    if isinstance(entry, str):
        entry = {'path': entry}
    entry = dict(entry)
    if not entry.get('path') or entry.get('enabled') is False:
        return None
    entry.setdefault('pattern', DEFAULT_PATTERN)
    entry.setdefault('members', list(DEFAULT_MEMBERS))
    entry.setdefault('years', list(YEARS))
    entry.setdefault('hour', None)
    entry.setdefault('min_cycles', DEFAULT_MIN_CYCLES)
    entry.setdefault('min_coverage', DEFAULT_MIN_COVERAGE)
    entry.setdefault('step_hours', 6)
    entry.setdefault('bands', [list(b) for b in DEFAULT_BANDS])
    entry.setdefault('variables', list(DEFAULT_VARIABLES))
    entry.setdefault('levels', None)
    entry.setdefault('sections', True)
    entry.setdefault('sealevel', True)
    entry['members'] = [str(m) for m in entry['members']]
    entry['variables'] = [v for v in entry['variables'] if v in VARS]
    entry['bands'] = [(float(b[0]), float(b[1])) for b in entry['bands']]
    return entry


def band_key(band):
    """'0-300' from (0.0, 300.0) -- the figure-name and csv form."""
    return '%g-%g' % (band[0], band[1])


def map_levels(cfg, entry):
    """Model levels to draw maps at: the block's own, else `map_levels:`."""
    lev = entry.get('levels')
    if lev is None:
        lev = cfg.get('map_levels') or [0]
    return [int(k) for k in lev]


def file_for(entry, year):
    return os.path.join(entry['path'],
                        entry['pattern'].format(year='%04d' % int(year)))


def in_range(entry, year):
    lo, hi = entry.get('years') or YEARS
    return int(lo) <= int(year) <= int(hi)


def months_for(cfg, entry, exp=None):
    """Months of `cycles:` that GREP can actually be compared against.

    Returns [(month, [cycles])] in order, keeping only months whose year is in
    range, whose GREP file is on disk, and which hold at least `min_cycles`
    cycles and enough of the calendar month -- a handful of cycles is not a
    monthly mean, and silently scoring one against a true monthly mean would
    be the easiest way to manufacture a bias. When ``exp`` is given, only
    cycles with a background on disk count.
    """
    hour = entry.get('hour')
    step = int(entry.get('step_hours') or 6)
    pool = [str(c) for c in (cfg.get('cycles') or []) if _CYCLE_RE.match(str(c))]
    want = {}
    for c in pool:
        if hour and c[8:10] != str(hour):
            continue
        if not in_range(entry, c[:4]):
            continue
        if exp is not None and background_file(exp, c) is None:
            continue
        want.setdefault(c[:6], []).append(c)
    out = []
    for month in sorted(want):
        if not os.path.exists(file_for(entry, month[:4])):
            continue
        have = sorted(want[month])
        if len(have) < int(entry['min_cycles']):
            continue
        if coverage(month, have, hour, step) < float(entry['min_coverage']):
            continue
        out.append((month, have))
    return out


def coverage(month, cycles, hour=None, step=6):
    """Fraction of the calendar month's cycles that are present.

    A partial month is not a monthly mean: eleven days of June differenced
    against GREP's true June carries the seasonal march of those missing
    nineteen days as if it were model error. The stage still reports the
    fraction on every panel, so a month that only just clears the bar is
    visibly marked rather than quietly averaged.
    """
    import calendar
    ndays = calendar.monthrange(int(month[:4]), int(month[4:6]))[1]
    per_day = 1 if hour else max(1, 24 // int(step))
    return len(cycles) / float(ndays * per_day)


def missing_files(cfg, entry):
    """GREP files a month of `cycles:` wants but that are not on disk."""
    out = []
    for c in cfg.get('cycles') or []:
        c = str(c)
        if not _CYCLE_RE.match(c) or not in_range(entry, c[:4]):
            continue
        p = file_for(entry, c[:4])
        if not os.path.exists(p) and p not in out:
            out.append(p)
    return out


def skip_reason(cfg, entry, exp=None):
    """Why there is nothing to do, in one line, or None when there is.

    Kept here rather than in the stage so preflight.py says the same thing.
    """
    cycles = [str(c) for c in (cfg.get('cycles') or [])]
    if not cycles:
        return 'no cycles configured'
    if months_for(cfg, entry, exp):
        return None
    years = sorted({c[:4] for c in cycles})
    lo, hi = entry.get('years') or YEARS
    if not any(in_range(entry, y) for y in years):
        return ('cycles cover %s; GREP covers %s-%s'
                % ('/'.join(years), lo, hi))
    miss = missing_files(cfg, entry)
    if miss:
        return 'missing %s' % ', '.join(os.path.basename(p) for p in miss)
    if exp is not None and not any(background_file(exp, c) for c in cycles):
        return 'no ocean background resolves under %s' % exp.root
    return ('no month has %d or more cycles and %.0f%% coverage'
            % (int(entry['min_cycles']), 100.0 * float(entry['min_coverage'])))


# --------------------------------------------------------------------------
# the model side
# --------------------------------------------------------------------------

def background_file(exp, cycle):
    """The ocean background valid at ``cycle``, or None.

    Experiment.background() and nothing else. For `kind: var` that resolves
    to f006 of cycle-6, which is the forecast valid at the CENTRE of this
    cycle's DA window -- the field the rest of the suite scores. Reading the
    cycle's own f003 instead (an earlier version of this module did) gives a
    state valid at cycle+3, the window's trailing edge, and silently ignores
    a per-experiment `background:` override such as 3dvar-rt's, whose history
    lives under a different stem entirely.

    A fixed forecast hour sampled at every 6-h cycle also lands on four
    different times of day, so the monthly mean carries no diurnal alias.
    """
    return exp.background(cycle, 'ocean')


def model_depth(path):
    """The model's level mid-depths, from the file's own z_l where it has one.

    This output is remapped to fixed z levels, so z_l IS the level depth
    everywhere except in partial bottom cells. lv_statespace.depth_from_h
    exists for the increment and variance files, which carry only a level
    index; using it here would substitute a deep-column median for an axis
    the file states exactly.
    """
    with Dataset(path) as ds:
        if 'z_l' in ds.variables:
            return np.asarray(ds['z_l'][:], dtype='f8')
    return None


def _edges(z):
    """Layer interfaces around monotonic level centres ``z``."""
    z = np.asarray(z, dtype='f8')
    mid = 0.5 * (z[:-1] + z[1:])
    first = max(0.0, 2.0 * z[0] - mid[0])
    last = 2.0 * z[-1] - mid[-1]
    return np.concatenate([[first], mid, [last]])


def _band_overlap(top, bot, band):
    """Thickness of each cell that lies inside ``band``. Arrays or scalars."""
    lo, hi = band
    with np.errstate(invalid='ignore'):
        return np.clip(np.minimum(bot, hi) - np.maximum(top, lo), 0.0, None)


def model_content(path, bands, levels=None, floor=None):
    """Heat and salt content of the model column, per band.

    Returns {band_key: {'ohc': 2-D J/m2, 'sc': 2-D kg/m2, 'deep': 2-D bool}}
    plus 'maps' with the requested level fields.

    The layer thickness is the file's own `h`, so partial bottom cells and
    vanished layers are handled by the model's own bookkeeping rather than by
    a nominal z axis: where a layer has collapsed, h is zero, it contributes
    nothing and does not advance the column. ``deep`` marks columns that
    actually reach the bottom of the band -- see common_mask() for why the
    comparison needs it.

    With ``floor`` (common_floor()), the same pass also runs on to the floor
    and adds 'sealevel': {'ssh', 'thermo', 'halo'}, 2-D metres -- see
    steric_parts(). One pass, because the upper ocean is read either way.
    """
    bands = [tuple(b) for b in bands]
    deepest = max(b[1] for b in bands)
    if floor is not None:
        deepest = max(deepest, float(np.nanmax(floor)))
    out = {band_key(b): None for b in bands}
    with Dataset(path) as ds:
        nk = ds['h'].shape[1] if ds['h'].ndim == 4 else ds['h'].shape[0]
        shape = ds['h'].shape[-2:]
        acc = {band_key(b): {'ohc': np.zeros(shape), 'sc': np.zeros(shape)}
               for b in bands}
        thermo, halo = np.zeros(shape), np.zeros(shape)
        top = np.zeros(shape)
        for k in range(nk):
            h = np.ma.filled(ds['h'][0, k].astype('f8'), 0.0)
            h = np.where(np.isfinite(h) & (h > H_MIN), h, 0.0)
            bot = top + h
            if np.all(top >= deepest):
                break
            t = np.ma.filled(ds['Temp'][0, k].astype('f8'), np.nan)
            s = np.ma.filled(ds['Salt'][0, k].astype('f8'), np.nan)
            wet = (h > 0) & np.isfinite(t) & np.isfinite(s)
            for b in bands:
                dz = np.where(wet, _band_overlap(top, bot, b), 0.0)
                if not np.any(dz > 0):
                    continue
                a = acc[band_key(b)]
                a['ohc'] += np.where(wet, t, 0.0) * dz
                a['sc'] += np.where(wet, s, 0.0) * dz
            if floor is not None:
                dz = np.where(wet, _band_overlap(top, bot, (0.0, floor)), 0.0)
                if np.any(dz > 0):
                    th, ha = steric_parts(np.where(wet, t, T_REF),
                                          np.where(wet, s, S_REF),
                                          0.5 * (top + bot))
                    thermo += th * dz
                    halo += ha * dz
            top = bot
        if floor is not None:
            ssh = (np.ma.filled(ds['ave_ssh'][0].astype('f8'), np.nan)
                   if 'ave_ssh' in ds.variables else np.full(shape, np.nan))
            out['sealevel'] = {'ssh': ssh, 'thermo': thermo, 'halo': halo}
    for b in bands:
        a = acc[band_key(b)]
        out[band_key(b)] = {'ohc': RHO0 * CP * a['ohc'],
                            'sc': RHO0 * 1e-3 * a['sc'],
                            'deep': top >= b[1]}
    return out


# --------------------------------------------------------------------------
# sea level
# --------------------------------------------------------------------------

def rho_wright(t, s, p):
    """In-situ density [kg m-3], MOM6 WRIGHT_FULL; t potential degC, s psu,
    p Pa."""
    a0, a1, a2 = _WA
    b0, b1, b2, b3, b4, b5 = _WB
    c0, c1, c2, c3, c4, c5 = _WC
    al0 = a0 + (a1 * t + a2 * s)
    p0 = b0 + (b4 * s + t * (b1 + (t * (b2 + b3 * t) + b5 * s)))
    lam = c0 + (c4 * s + t * (c1 + (t * (c2 + c3 * t) + c5 * s)))
    return (p + p0) / (lam + al0 * (p + p0))


def steric_parts(t, s, z):
    """Thermosteric and halosteric height per metre of water column at depth z.

        thermo = -[rho(T, S_ref, p) - rho(T_ref, S_ref, p)] / rho0
        halo   = -[rho(T, S,     p) - rho(T,     S_ref, p)] / rho0

    The split is EXACT: thermo + halo is the full steric height against the
    fixed (T_ref, S_ref) state, with no cross term left over -- the halosteric
    part is evaluated at the actual temperature, not at T_ref. The reference
    itself is arbitrary and only sets the absolute level, which the figure
    removes; what survives is a difference between two states on the same
    volume, and that does not depend on it.
    """
    p = RHO_EOS * GRAV * np.asarray(z, dtype='f8')
    r_ts = rho_wright(t, s, p)
    r_t = rho_wright(t, S_REF, p)
    r_0 = rho_wright(T_REF, S_REF, p)
    return -(r_t - r_0) / RHO_EOS, -(r_ts - r_t) / RHO_EOS


def model_floor(path):
    """Model column depth [m]: the sum of its own layer thicknesses."""
    with Dataset(path) as ds:
        h = np.ma.filled(ds['h'][0].astype('f8'), 0.0)
    h = np.where(np.isfinite(h) & (h > H_MIN), h, 0.0)
    return h.sum(axis=0)


def grep_floor(entry, month, member, grid):
    """One member's column depth on the model grid, from its own wet mask.

    The bottom interface of the deepest wet level, so the same edges
    grep_sealevel() integrates over. NaN where the member is dry.
    """
    path = file_for(entry, month[:4])
    it = month_index(path, month)
    name = '%s_%s' % (VARS['Temp'], member)
    if it is None:
        return None
    with Dataset(path) as ds:
        if name not in ds.variables:
            return None
        z = np.asarray(ds['depth'][:], dtype='f8')
        lat = np.asarray(ds['latitude'][:], dtype='f8')
        lon = np.asarray(ds['longitude'][:], dtype='f8')
        edges = _edges(z)
        floor = np.zeros((lat.size, lon.size))
        for k in range(z.size):
            wet = ~np.ma.getmaskarray(ds[name][it, k])
            if not wet.any():
                break
            floor = np.where(wet, edges[k + 1], floor)
    return lv_verif.sample_to_grid(grid, np.where(floor > 0, floor, np.nan),
                                   lat, lon, 0)


def common_floor(grid, model_path, entry, month, members):
    """Depth every source is integrated to: the shallowest of the model and
    the members, column by column; 0 where any of them is dry.

    Steric height is an integral over the column, so two sources integrated
    to their own floors differ wherever their bathymetry does -- a few
    hundred metres of 1 degC water is centimetres of steric height, as large
    as the signal. Integrating everyone to the same floor puts every panel
    on one volume; what is dropped is the sliver of abyss one source has and
    another lacks (~70 m of a ~3700 m mean column).
    """
    floor = model_floor(model_path)
    for m in members:
        f = grep_floor(entry, month, m, grid)
        if f is not None:
            floor = np.minimum(floor, np.nan_to_num(f, nan=0.0))
    return np.where(grid.mask, floor, 0.0)


def grep_sealevel(entry, month, member, grid, floor):
    """SSH and steric height of one GREP member, on the model grid, 2-D m.

    Each GREP level is gathered onto the model grid and integrated there with
    GREP's OWN layer edges, cut at the common floor -- the vertical axis is
    still GREP's, only the horizontal placement is the model's, which is what
    lets both sides share one floor.
    """
    path = file_for(entry, month[:4])
    it = month_index(path, month)
    if it is None:
        return None
    tname = '%s_%s' % (VARS['Temp'], member)
    sname = '%s_%s' % (VARS['Salt'], member)
    zname = 'zos_%s' % member
    with Dataset(path) as ds:
        if tname not in ds.variables or sname not in ds.variables:
            return None
        z = np.asarray(ds['depth'][:], dtype='f8')
        lat = np.asarray(ds['latitude'][:], dtype='f8')
        lon = np.asarray(ds['longitude'][:], dtype='f8')
        edges = _edges(z)
        half = lv_verif._window(grid, lon)
        deepest = float(np.nanmax(floor))
        thermo, halo = np.zeros(grid.shape), np.zeros(grid.shape)
        for k in range(z.size):
            if edges[k] >= deepest:
                break
            t = np.ma.filled(ds[tname][it, k].astype('f8'), np.nan)
            s = np.ma.filled(ds[sname][it, k].astype('f8'), np.nan)
            th, ha = steric_parts(t, s, z[k])
            th = lv_verif.sample_to_grid(grid, th, lat, lon, half)
            ha = lv_verif.sample_to_grid(grid, ha, lat, lon, half)
            dz = _band_overlap(edges[k], edges[k + 1], (0.0, floor))
            ok = np.isfinite(th) & np.isfinite(ha) & (dz > 0)
            thermo += np.where(ok, th * dz, 0.0)
            halo += np.where(ok, ha * dz, 0.0)
        ssh = (lv_verif.sample_to_grid(
            grid, np.ma.filled(ds[zname][it].astype('f8'), np.nan),
            lat, lon, half) if zname in ds.variables
            else np.full(grid.shape, np.nan))
    return {'ssh': ssh, 'thermo': thermo, 'halo': halo}


# --------------------------------------------------------------------------
# the GREP side
# --------------------------------------------------------------------------

def month_index(path, month):
    """Index of ``month`` (YYYYMM) in a GREP file's time axis, or None."""
    import datetime
    with Dataset(path) as ds:
        t = ds['time']
        units = getattr(t, 'units', 'seconds since 1950-01-01')
        m = re.search(r'since\s+(\d{4})-(\d{2})-(\d{2})', units)
        base = (datetime.datetime(int(m.group(1)), int(m.group(2)),
                                  int(m.group(3))) if m
                else datetime.datetime(1950, 1, 1))
        scale = 1.0 if units.strip().lower().startswith('sec') else 86400.0
        for i, v in enumerate(np.asarray(t[:], dtype='f8')):
            d = base + datetime.timedelta(seconds=float(v) * scale)
            if '%04d%02d' % (d.year, d.month) == month:
                return i
    return None


def grep_content(entry, month, member, grid, bands):
    """Heat and salt content of one GREP member, on the MODEL grid.

    Integrated on GREP's OWN depth axis with thicknesses from its own level
    interfaces, then gathered horizontally -- the integral never sees the
    vertical interpolation. Interpolating a column onto the model levels and
    integrating that would smear the thermocline through the interpolation
    weights before the integral ever ran, and the answer would depend on the
    model's level spacing, which has nothing to do with GREP.
    """
    path = file_for(entry, month[:4])
    it = month_index(path, month)
    if it is None:
        return None
    bands = [tuple(b) for b in bands]
    out = {}
    with Dataset(path) as ds:
        z = np.asarray(ds['depth'][:], dtype='f8')
        lat = np.asarray(ds['latitude'][:], dtype='f8')
        lon = np.asarray(ds['longitude'][:], dtype='f8')
        edges = _edges(z)
        deepest = max(b[1] for b in bands)
        tname, sname = '%s_%s' % (VARS['Temp'], member), '%s_%s' % (VARS['Salt'], member)
        if tname not in ds.variables or sname not in ds.variables:
            return None
        acc = {band_key(b): {'ohc': np.zeros((lat.size, lon.size)),
                             'sc': np.zeros((lat.size, lon.size))}
               for b in bands}
        floor = np.zeros((lat.size, lon.size))
        for k in range(z.size):
            if edges[k] >= deepest:
                break
            t = np.ma.filled(ds[tname][it, k].astype('f8'), np.nan)
            s = np.ma.filled(ds[sname][it, k].astype('f8'), np.nan)
            wet = np.isfinite(t) & np.isfinite(s)
            floor = np.where(wet, edges[k + 1], floor)
            for b in bands:
                thick = float(_band_overlap(edges[k], edges[k + 1], b))
                if thick <= 0:
                    continue
                a = acc[band_key(b)]
                a['ohc'] += np.where(wet, t, 0.0) * thick
                a['sc'] += np.where(wet, s, 0.0) * thick
        half = lv_verif._window(grid, lon)
        for b in bands:
            a = acc[band_key(b)]
            out[band_key(b)] = {
                'ohc': lv_verif.sample_to_grid(grid, RHO0 * CP * a['ohc'],
                                               lat, lon, half),
                'sc': lv_verif.sample_to_grid(grid, RHO0 * 1e-3 * a['sc'],
                                              lat, lon, half),
                'deep': lv_verif.sample_to_grid(
                    grid, (floor >= b[1]).astype('f8'), lat, lon, half) > 0.5}
    return out


def common_mask(grid, model_deep, grep_deep):
    """Wet columns that reach the band bottom in the model AND in GREP.

    Integrating to a shelf's own floor on one side and to the band's bottom on
    the other differences two integrals over different volumes, which shows up
    as a coherent ring along every shelf edge and swamps the interior signal.
    The mask is built once per band from every member so all the panels cover
    the same water.
    """
    keep = grid.mask & np.asarray(model_deep, dtype=bool)
    for d in grep_deep:
        if d is not None:
            keep = keep & np.asarray(d, dtype=bool)
    return keep


class GrepSource:
    """lv_statespace LevelSource for one member and month, on the model grid.

    Presents GREP on the MODEL's level depths and the model's tripolar grid,
    so section_planes and the map reducers consume it exactly as they consume
    a model file. Levels the vertical interpolation cannot reach -- the model
    column runs to 6375 m and GREP stops at 5902 m -- return None rather than
    GREP's deepest value carried downward, which would score a fabricated
    constant against the abyss (the reason level_weights omits instead of
    clamping).

    ``wet_from`` is the MODEL's own source: wetness never comes from GREP's
    fill, for the reason lv_woa.WoaSource gives.
    """

    def __init__(self, path, member, it, grid, depth, wet_from=None):
        self.path = path
        self.member = member
        self.it = it
        self.grid = grid
        self.wet_from = wet_from
        self.ds = Dataset(path)
        self.z = np.asarray(self.ds['depth'][:], dtype='f8')
        self.lat = np.asarray(self.ds['latitude'][:], dtype='f8')
        self.lon = np.asarray(self.ds['longitude'][:], dtype='f8')
        self.half = lv_verif._window(grid, self.lon)
        self.nk = len(depth)
        self.want = {k: (i, w) for k, i, w in level_weights(self.z, depth)}
        self._src = {}
        self._out = {}

    def close(self):
        try:
            self.ds.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _name(self, var):
        stem = VARS.get(var)
        if stem is None:
            return None
        name = '%s_%s' % (stem, self.member)
        return name if name in self.ds.variables else None

    def _src_level(self, name, i):
        key = (name, i)
        if key not in self._src:
            if len(self._src) > 5:
                self._src.clear()
            a = self.ds[name][self.it, i]
            self._src[key] = np.ma.filled(a.astype('f8'), np.nan)
        return self._src[key]

    def levels(self, var):
        return None if self._name(var) is None else self.nk

    def level(self, var, k):
        name = self._name(var)
        if name is None or k not in self.want:
            return None
        key = (name, k)
        if key not in self._out:
            if len(self._out) > 3:
                self._out.clear()
            i, w = self.want[k]
            a = self._src_level(name, i)
            b = self._src_level(name, i + 1)
            src = (1.0 - w) * a + w * b
            self._out[key] = (
                None if not np.any(np.isfinite(src))
                else lv_verif.sample_to_grid(self.grid, src, self.lat,
                                             self.lon, self.half))
        return self._out[key]

    def wet(self, k, nk):
        return None if self.wet_from is None else self.wet_from.wet(k, nk)
