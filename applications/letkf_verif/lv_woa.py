"""WOA23 climatology, placed on the model grid for comparison with the background.

Unlike the L4 products in lv_verif, this is NOT a same-day analysis: it is the
1955-2022 decadal average. "Model minus WOA" therefore carries the ocean's real
interannual and eddy anomaly as well as any model error, and every figure built
on it says so. It is read for structure -- a thermocline at the wrong depth, a
collapsing halocline, a basin-wide offset -- not as a score.

The field is built to be MODEL-SHAPED: (nk, ny, nx) on the model's own mean
level depths, so every reducer in lv_statespace consumes it unchanged through
the LevelSource protocol.

Three transformations happen here, in this order, and the order matters:

    1. time      linear between the two mid-month climatologies bracketing the
                 cycle's day of year
    2. splice    monthly at and above 1500 m, annual below -- the monthly files
                 simply stop there
    3. depth     onto the model's level depths, AFTER converting in-situ to
                 potential temperature (see potential_temperature)

This module deliberately does not import lv_statespace: that module imports
this one. WoaSource therefore takes the model's own source for its wetness
test rather than reaching for lv_statespace._wet_level.
"""

import calendar
import datetime
import os

import numpy as np
from netCDF4 import Dataset

import lv_verif

# The monthly files stop here; the annual files run to 5500 m. On the standard
# 0.25 deg distribution the monthly depth array is bitwise identical to the
# annual's first 57 entries, and annual[56] is exactly 1500.0, so the splice is
# an index cut rather than an interpolation. _check_axes() enforces that.
SPLICE_DEPTH = 1500.0
N_MONTHLY = 57

DEFAULT_PATTERN = 'woa23_decav_{v}{mm}_{res}.nc'
DEFAULT_RES = '04'

# model variable -> (file letter, field variable, units)
VARS = {'Temp': ('t', 't_an', 'degC'),
        'Salt': ('s', 's_an', 'psu')}

LABEL = 'WOA23'


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def configured(cfg):
    """The normalised `woa:` block, or None when the config does not ask.

    A bare string is taken as the path, mirroring how `verification:`
    products may be written either way.
    """
    entry = (cfg or {}).get('woa')
    if not entry:
        return None
    if isinstance(entry, str):
        entry = {'path': entry}
    entry = dict(entry)
    if not entry.get('path'):
        return None
    entry.setdefault('pattern', DEFAULT_PATTERN)
    entry.setdefault('res', DEFAULT_RES)
    entry.setdefault('splice_depth', SPLICE_DEPTH)
    entry.setdefault('potential_temperature', True)
    entry.setdefault('sections', True)
    return entry


def path_for(entry, var, month):
    """The WOA file for one model variable and one month (0 = annual).

    A plain format rather than lv_verif's glob: WOA filenames are fully
    determined -- no production-date suffix -- so a miss is a real miss and is
    worth reporting rather than silently returning None from an empty glob.
    """
    letter = VARS[var][0]
    name = entry['pattern'].format(v=letter, mm='%02d' % month,
                                   res=entry['res'])
    return os.path.join(entry['path'], name)


def missing_files(entry, variables=('Temp', 'Salt')):
    """Every expected WOA file that is not on disk. Empty when all present."""
    out = []
    for var in variables:
        if var not in VARS:
            continue
        for m in range(0, 13):
            p = path_for(entry, var, m)
            if not os.path.exists(p):
                out.append(p)
    return out


# --------------------------------------------------------------------------
# time
# --------------------------------------------------------------------------

def month_centres(year):
    """Fractional 0-based day-of-year of each monthly climatology's centre.

    The monthly fields are centred on the MIDDLE of their month, not its
    first day -- their own `time` is 395.5 + m months since 1955. Taken in the
    cycle's own calendar so February's centre moves by half a day in a leap
    year; that shift is under a day and immaterial to the seasonal cycle, but
    a nominal 365-day year would leave the December-to-January interval a day
    short and make the wrap arithmetic inconsistent with the rest.
    """
    out = {}
    for m in range(1, 13):
        n = calendar.monthrange(year, m)[1]
        first = datetime.date(year, m, 1).timetuple().tm_yday - 1
        out[m] = first + n / 2.0
    return out


def _days_in(year):
    return 366 if calendar.isleap(year) else 365


def month_weights(cycle):
    """[(month, weight), (month, weight)] for a YYYYMMDDHH cycle.

    Linear in time between the two mid-month centres bracketing the cycle;
    the weights sum to 1.

    The December-to-January wrap is handled by shifting one node onto the
    other's year, NOT by a modulo on the day number. A cycle on 3 January sits
    between 15 December of the previous year and 15 January of this one; a
    `doy % 365` puts it outside every interval, and a nearest-centre match
    then picks January twice, silently dropping the December half of the blend
    and biasing every early-January cycle toward the wrong season.
    """
    y, mo, dd = int(cycle[:4]), int(cycle[4:6]), int(cycle[6:8])
    hh = int(cycle[8:10]) if len(cycle) >= 10 else 0
    t = (datetime.date(y, mo, dd).timetuple().tm_yday - 1) + hh / 24.0
    c = month_centres(y)
    if t < c[1]:                                  # before mid-January
        m0, m1 = 12, 1
        t0, t1 = c[12] - _days_in(y - 1), c[1]
    elif t >= c[12]:                              # on or after mid-December
        m0, m1 = 12, 1
        t0, t1 = c[12], c[1] + _days_in(y)
    else:
        m0 = max(m for m in range(1, 13) if c[m] <= t)
        m1 = m0 + 1
        t0, t1 = c[m0], c[m1]
    w = (t - t0) / (t1 - t0)
    return [(m0, 1.0 - w), (m1, w)]


# --------------------------------------------------------------------------
# seawater: in-situ -> potential temperature
# --------------------------------------------------------------------------

def depth_to_pressure(z, lat):
    """Depth (m) to pressure (dbar), Saunders (1981)."""
    c1 = 5.92e-3 + 5.25e-3 * np.sin(np.deg2rad(np.abs(lat))) ** 2
    return ((1.0 - c1) - np.sqrt((1.0 - c1) ** 2 - 8.84e-6 * z)) / 4.42e-6


def _lapse(t, s, p):
    """Adiabatic lapse rate in degC/dbar (Bryden 1973)."""
    return (3.5803e-5 + 8.5258e-6 * t - 6.836e-8 * t * t + 6.6228e-10 * t ** 3
            + (1.8932e-6 - 4.2393e-8 * t) * (s - 35.0)
            + (1.8741e-8 - 6.7795e-10 * t + 8.733e-12 * t * t
               - 5.4481e-14 * t ** 3) * p
            + (-1.1351e-10 + 2.7759e-12 * t) * (s - 35.0) * p)


def potential_temperature(t, s, p, pref=0.0):
    """In-situ to potential temperature, EOS-80 (Fofonoff 1977 RK4).

    WOA distributes IN-SITU temperature; MOM6 here runs WRIGHT_FULL and writes
    POTENTIAL temperature. The two differ by the adiabatic compression term --
    nothing at the surface, but 0.06 degC at 1000 m, 0.35 at 4000 m and 0.63 at
    6000 m. That is negligible against a thermocline bias and larger than a
    real abyssal one, so differencing the two conventions directly would make
    the deep temperature bias mostly a units artefact.

    Salinity needs no such conversion: WOA's is practical (PSS-78) and so is
    the model's.
    """
    h = pref - p
    xk = h * _lapse(t, s, p)
    t = t + 0.5 * xk
    q = xk
    p = p + 0.5 * h
    xk = h * _lapse(t, s, p)
    t = t + 0.29289322 * (xk - q)
    q = 0.58578644 * xk + 0.121320344 * q
    xk = h * _lapse(t, s, p)
    t = t + 1.707106781 * (xk - q)
    q = 3.414213562 * xk - 4.121320344 * q
    p = p + 0.5 * h
    xk = h * _lapse(t, s, p)
    return t + (xk - 2.0 * q) / 6.0


# --------------------------------------------------------------------------
# depth
# --------------------------------------------------------------------------

def level_weights(zsrc, zdst):
    """[(k, i, w)] with target level k = (1-w)*zsrc[i] + w*zsrc[i+1].

    Targets OUTSIDE [zsrc[0], zsrc[-1]] are OMITTED, never clamped. np.interp
    clamps, which on this grid would carry WOA's 5500 m value down the four
    model levels below it (the column reaches 6137 m) and score a fabricated
    constant against the abyss as if it were data -- exactly where nobody
    would look for it.
    """
    zsrc = np.asarray(zsrc, dtype='f8')
    out = []
    for k, z in enumerate(zdst):
        if not np.isfinite(z) or z < zsrc[0] or z > zsrc[-1]:
            continue
        i = int(np.searchsorted(zsrc, z, side='right')) - 1
        i = min(max(i, 0), zsrc.size - 2)
        span = zsrc[i + 1] - zsrc[i]
        out.append((k, i, float((z - zsrc[i]) / span) if span > 0 else 0.0))
    return out


def _check_axes(adepth, mdepth, apath, mpath):
    """The splice is an index cut, so the two axes must agree where they overlap."""
    if mdepth.size != N_MONTHLY or not np.array_equal(adepth[:N_MONTHLY],
                                                      mdepth):
        raise ValueError(
            'WOA depth axes do not line up: %s has %d levels and %s has %d, '
            'and the first %d do not match. The monthly/annual splice assumes '
            'they do; a different WOA release needs this revisited.'
            % (os.path.basename(mpath), mdepth.size, os.path.basename(apath),
               adepth.size, N_MONTHLY))


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _src_level(handles, var, i, adepth, splice):
    """One WOA source level: time-blended monthly, or the annual field."""
    fvar = VARS[var][1]
    if adepth[i] <= splice:
        acc = None
        for m, w in handles['months']:
            a = handles[(var, m)][fvar][0, i]
            a = np.ma.filled(a.astype('f8'), np.nan) * w
            acc = a if acc is None else acc + a
        return acc
    a = handles[(var, 0)][fvar][0, i]
    return np.ma.filled(a.astype('f8'), np.nan)


def build(entry, cycle, model_depth, variables=('Temp', 'Salt'), log=None):
    """WOA on its NATIVE grid, at the MODEL's level depths.

    Returns (fields, lat, lon, info) where fields is {var: (nk, 720, 1440) f4}.

    Deliberately stops short of the horizontal sample: placing every level on
    the tripolar grid up front would cost 466 MB per variable, and WoaSource
    does it one level at a time instead, which is an index gather.

    Source levels are visited in increasing depth with a two-level rolling
    buffer, so each file is read through once per variable.
    """
    variables = [v for v in variables if v in VARS]
    if not variables:
        return {}, None, None, {}
    months = month_weights(cycle)
    handles, opened = {'months': months}, []
    try:
        for var in variables:
            for m in [0] + [mm for mm, _w in months]:
                p = path_for(entry, var, m)
                ds = Dataset(p)
                opened.append(ds)
                handles[(var, m)] = ds
                handles[('path', var, m)] = p

        v0 = variables[0]
        adepth = np.asarray(handles[(v0, 0)]['depth'][:], dtype='f8')
        mdepth = np.asarray(handles[(v0, months[0][0])]['depth'][:],
                            dtype='f8')
        _check_axes(adepth, mdepth, handles[('path', v0, 0)],
                    handles[('path', v0, months[0][0])])
        lat = np.asarray(handles[(v0, 0)]['lat'][:], dtype='f8')
        lon = np.asarray(handles[(v0, 0)]['lon'][:], dtype='f8')

        splice = float(entry.get('splice_depth', SPLICE_DEPTH))
        want = level_weights(adepth, model_depth)
        nk = len(model_depth)
        need = sorted({i for _k, i, _w in want} | {i + 1 for _k, i, _w in want})

        # Potential temperature needs salinity at the same point, so the
        # source levels of both variables are held together.
        buf = {}
        p2d = None
        if entry.get('potential_temperature') and 'Temp' in variables:
            p2d = depth_to_pressure(
                np.asarray(adepth)[:, None, None],
                lat[None, :, None])              # (nlev, nlat, 1)

        out = {v: np.full((nk,) + (lat.size, lon.size), np.nan, 'f4')
               for v in variables}
        for i in need:
            buf[i] = {v: _src_level(handles, v, i, adepth, splice)
                      for v in variables}
            if p2d is not None and 'Salt' in buf[i] and 'Temp' in buf[i]:
                buf[i]['Temp'] = potential_temperature(
                    buf[i]['Temp'], buf[i]['Salt'], p2d[i])
        for k, i, w in want:
            for v in variables:
                a, b = buf[i][v], buf[i + 1][v]
                out[v][k] = ((1.0 - w) * a + w * b).astype('f4')

        info = {'months': [[int(m), float(w)] for m, w in months],
                'splice_depth': splice,
                'potential_temperature': bool(p2d is not None),
                'n_src_levels': int(adepth.size),
                'n_model_levels': int(nk),
                'n_below_woa': int(nk - len(want)),
                'files': sorted(os.path.basename(handles[('path', v, m)])
                                for v in variables
                                for m in [0] + [mm for mm, _w in months])}
        if log:
            log('    WOA %s: months %s, %d/%d model levels covered'
                % (cycle, ' '.join('%02d@%.2f' % (m, w) for m, w in months),
                   len(want), nk))
        return out, lat, lon, info
    finally:
        for ds in opened:
            ds.close()


class WoaSource:
    """lv_statespace LevelSource backed by build().

    The horizontal placement happens HERE, one level at a time, through
    lv_verif.sample_to_grid -- the same index arithmetic the L4 products use,
    so a WOA point lands in a model cell exactly the way an OSTIA point does.
    _window() returns 0 for this pair (both grids are 0.25 deg), making it a
    pure nearest-neighbour gather.

    ``wet_from`` is the MODEL's own source. Wetness is never taken from WOA's
    fill: below the sea floor MOM6 writes zeros while WOA has a perfectly good
    climatological value there, and differencing the two fabricates a bias
    under the bathymetry.
    """

    def __init__(self, fields, lat, lon, grid, wet_from=None):
        self.fields = fields
        self.grid = grid
        self.lat, self.lon = lat, lon
        self.half = lv_verif._window(grid, lon) if lon is not None else 0
        self.wet_from = wet_from
        self._cache = {}

    def levels(self, var):
        f = self.fields.get(var)
        return None if f is None else f.shape[0]

    def level(self, var, k):
        f = self.fields.get(var)
        if f is None or k >= f.shape[0]:
            return None
        key = (var, k)
        if key not in self._cache:
            if len(self._cache) > 3:
                self._cache.clear()
            src = f[k]
            self._cache[key] = (
                None if not np.any(np.isfinite(src))
                else lv_verif.sample_to_grid(self.grid, src.astype('f8'),
                                             self.lat, self.lon, self.half))
        return self._cache[key]

    def wet(self, k, nk):
        return None if self.wet_from is None else self.wet_from.wet(k, nk)
