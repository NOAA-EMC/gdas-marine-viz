"""The parametric background-error standard deviation, D in B = K D C D K^T.

soca's diagb writes D once per cycle as `*bkgerr_parametric_stddev.nc`, one
file per realm, with one field per control variable. This module finds those
files and presents them to lv_statespace's reducers as a LevelSource on the
model grid, so the maps and sections of D are cut exactly like the increment
and background ones.

Two things differ from every other state file in the suite:

    grid    D lives on the B-matrix geometry, which is the model grid
            subsampled [::2, ::2] (checked point for point against the
            gridspec: identical lat/lon). It is expanded back by repetition,
            so each coarse value covers the 2x2 block it was taken at the
            corner of. Nothing is interpolated.
    depth   like an increment, the file carries only a level index. Sections
            borrow the background's layer thickness, the same arrangement
            lv_statespace.section_geometry documents for the increments.

Zero is not a standard deviation diagb ever assigns to wet water, so D == 0
is masked: it is land, a vanished layer below the sea floor, or (for the ice
fields) no ice. Without that, the sub-floor zeros paint over the bathymetry
and pin every colour scale to zero.

This module does not import matplotlib, so preflight.py can use it.
"""

import os
import glob

import numpy as np
from netCDF4 import Dataset

# Relative to Experiment.dir_for(cycle) -- '.../gdas.YYYYMMDD/HH/analysis'.
# The reanalysis-stream archives keep D under the analysis tree; a COMROOT
# keeps it in a bmatrix/ directory beside it. Tried in order.
PATTERNS = ('{realm}/bmatrix/*bkgerr_parametric_stddev.nc',
            '../bmatrix/{realm}/*bkgerr_parametric_stddev.nc')

REALMS = ('ocean', 'ice')

# Written to the file but not control variables with a parametric sigma:
# `h` carries a 1e-5 placeholder, and ave_ssh is identically ~0 because SSH
# enters through the balance operator K, not through D. Anything else that is
# zero everywhere is dropped at read time as well (see variables()).
NOT_CONTROL = {'h'}
ZERO = 1e-10

# Display order; anything not listed follows alphabetically.
ORDER = ['Temp', 'Salt', 'u', 'v', 'ave_ssh',
         'aice_h', 'hi_h', 'hi_div_aice_h', 'hs_h', 'hs_div_aice_h']

UNITS = {'Temp': 'degC', 'Salt': 'psu', 'u': 'm/s', 'v': 'm/s',
         'ave_ssh': 'm', 'aice_h': 'fraction', 'hi_h': 'm',
         'hi_div_aice_h': 'm', 'hs_h': 'm', 'hs_div_aice_h': 'm'}


def bkgerr_file(exp, cycle, realm):
    """Path to one realm's D for ``cycle``, or None."""
    base = exp.dir_for(cycle)
    for pat in PATTERNS:
        hits = sorted(glob.glob(os.path.normpath(
            os.path.join(base, pat.format(realm=realm)))))
        if hits:
            return hits[0]
    return None


def sort_vars(names):
    rank = {v: i for i, v in enumerate(ORDER)}
    return sorted(names, key=lambda v: (rank.get(v, len(ORDER)), v))


def variables(path):
    """Control variables in a D file: 2-D/3-D fields that are not all zero."""
    out = []
    with Dataset(path) as ds:
        for name, var in ds.variables.items():
            if name in NOT_CONTROL or var.ndim < 3:
                continue
            a = np.ma.filled(var[:].astype('f8'), 0.0)
            if np.nanmax(np.abs(a)) > ZERO:
                out.append(name)
    return sort_vars(out)


class BkgerrSource:
    """One D file as an lv_statespace LevelSource on the model grid.

    Duck-typed rather than subclassing lv_statespace.LevelSource, for the
    reason lv_woa.WoaSource gives: open_source() tests for a `level`
    attribute.
    """

    def __init__(self, path, grid):
        self.ds = Dataset(path)
        self.grid = grid
        ny, nx = self.ds.variables[next(
            n for n, v in self.ds.variables.items() if v.ndim >= 3)].shape[-2:]
        fy, fx = grid.shape[0] // ny, grid.shape[1] // nx
        if (fy * ny, fx * nx) != tuple(grid.shape):
            self.close()
            raise ValueError('%s is %dx%d, not a whole-number subsample of '
                             'the %dx%d model grid'
                             % (path, ny, nx, grid.shape[0], grid.shape[1]))
        self.fy, self.fx = fy, fx

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

    def levels(self, var):
        if var not in self.ds.variables:
            return None
        v = self.ds[var]
        return v.shape[1] if v.ndim == 4 else 1

    def level(self, var, k):
        if var not in self.ds.variables:
            return None
        v = self.ds[var]
        if v.ndim == 4:
            if k >= v.shape[1]:
                return None
            a = v[0, k]
        else:
            a = v[0] if v.ndim == 3 else v[:]
        a = np.ma.filled(a.astype('f8'), np.nan)
        a = np.where(a > ZERO, a, np.nan)
        if self.fy > 1 or self.fx > 1:
            a = np.repeat(np.repeat(a, self.fy, axis=0), self.fx, axis=1)
        return a

    def wet(self, k, nk):
        return None
