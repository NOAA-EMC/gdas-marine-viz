"""Atmospheric forcing seen by the ocean, from the coupled atmosphere's
surface history.

Each experiment here is a coupled run, so its ocean and ice are driven by
its own atmosphere. A difference between two experiments' backgrounds is
then partly a difference in forcing, not only in the DA -- and a change in
the ocean state feeds back on the winds and fluxes above it. This reduces,
per cycle, the surface fields the ocean and ice actually respond to, from
the atmosphere history valid at the analysis time (the same f006 the ocean
background is), onto the ocean grid:

    wind10   10 m wind speed                          m s-1
    tau      surface wind-stress magnitude            N m-2
    qnet     net surface heat flux into the ocean     W m-2  (SW + LW - LH - SH,
             positive warming the ocean)
    prate    precipitation rate                       mm day-1
    t2m      2 m air temperature                      degC

Maps at the report's stride go to the cycle's npz under 'atmos/<field>_k0';
area-weighted means per region go to the JSON. Everything is restricted to
ocean points. An experiment without an atmosphere history (3dvar-rt keeps
only its restarts) simply has no block.
"""

import os

import numpy as np
from netCDF4 import Dataset

# name -> (label, units, colour scale (vmin, vmax), diverging?)
FIELDS = {
    'wind10': ('10 m wind speed', 'm s$^{-1}$', (0.0, 20.0), False),
    'tau': ('wind stress', 'N m$^{-2}$', (0.0, 0.6), False),
    'qnet': ('net heat flux into ocean', 'W m$^{-2}$', (-300.0, 300.0), True),
    'prate': ('precipitation', 'mm day$^{-1}$', (0.0, 50.0), False),
    't2m': ('2 m air temperature', '$^\\circ$C', (-30.0, 35.0), False),
}


def _read(ds, name):
    return np.ma.filled(ds[name][0].astype('f8'), np.nan)


def derive(ds):
    """The FIELDS from one 'sfc' history file, on the atmosphere grid."""
    out = {}
    # the components are kept too (not drawn as rows) so the wind-speed
    # map can carry direction arrows
    out['u10'] = _read(ds, 'ugrd10m')
    out['v10'] = _read(ds, 'vgrd10m')
    out['wind10'] = np.hypot(out['u10'], out['v10'])
    out['tau'] = np.hypot(_read(ds, 'uflx_ave'), _read(ds, 'vflx_ave'))
    out['qnet'] = (_read(ds, 'dswrf_ave') - _read(ds, 'uswrf_ave')
                   + _read(ds, 'dlwrf_ave') - _read(ds, 'ulwrf_ave')
                   - _read(ds, 'lhtfl_ave') - _read(ds, 'shtfl_ave'))
    out['prate'] = _read(ds, 'prate_ave') * 86400.0
    out['t2m'] = _read(ds, 'tmp2m') - 273.15
    return out


def history_file(exp, cycle):
    """The atmosphere 'sfc' history valid at the analysis time: the f006 of
    the cycle the ocean background comes from, under the same stem."""
    bc = exp.background_cycle(cycle)
    stem = exp.bkg.get('stem')
    d = (exp.dir_for(bc, stem, 'atmos') if stem
         else os.path.join(exp.dir_for(bc), 'atmos'))
    p = os.path.join(d, 'gdas.t%sz.sfc.f006.nc' % bc[8:10])
    return p if os.path.exists(p) else None


def to_ocean_grid(field, lat, lon, grid):
    """Nearest atmosphere cell for every ocean cell.

    The atmosphere grid is Gaussian in latitude, so the index is found by
    search rather than by spacing arithmetic; longitude is periodic.
    """
    asc = lat[0] < lat[-1]
    la = lat if asc else lat[::-1]
    j = np.clip(np.searchsorted(la, grid.lat), 1, la.size - 1)
    j = np.where(np.abs(la[j - 1] - grid.lat) < np.abs(la[j] - grid.lat),
                 j - 1, j)
    if not asc:
        j = la.size - 1 - j
    dlo = (lon[-1] - lon[0]) / (lon.size - 1)
    i = np.mod(np.rint((np.mod(grid.lon180 - lon[0], 360.0)) / dlo).astype(int),
               lon.size)
    return np.where(grid.mask, field[j, i], np.nan)


def compute(grid, exp, cycle, cfg, regions, stride):
    """(scalars, maps) for one experiment and cycle, or (None, {})."""
    path = history_file(exp, cycle)
    if path is None:
        return None, {}
    wanted = list(cfg.get('atmos_vars') or FIELDS)
    if 'wind10' in wanted:
        wanted += ['u10', 'v10']
    with Dataset(path) as ds:
        lat = np.asarray(ds['grid_yt'][:], dtype='f8')
        lon = np.asarray(ds['grid_xt'][:], dtype='f8')
        fields = derive(ds)
    scalars = {'file': os.path.basename(path), 'region': {}}
    maps = {}
    for name in wanted:
        if name not in fields:
            continue
        f = to_ocean_grid(fields[name], lat, lon, grid)
        maps['atmos/%s_k0' % name] = f[::stride, ::stride].astype('f4')
        if name not in FIELDS:
            continue                    # components: maps only
        for rname, sel in regions.items():
            scalars['region'].setdefault(rname, {})[name] = grid.wmean(f, sel)
    return scalars, maps
