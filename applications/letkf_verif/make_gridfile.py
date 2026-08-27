#!/usr/bin/env python3
"""Write a small, portable grid file for the verification suite.

The full soca gridspec is ~190 MB, but the suite only reads lon / lat / area /
mask2d from it -- depth comes from each cycle's background. This extracts
exactly that into a single small file, which is what you copy to another
machine instead of the original.

lon / lat / area keep their float64 precision: the gridspec stores them as
float64, and every area-weighted number in the suite is built from `area`.
Rounding to float32 shifts those by ~1e-7 relative -- harmless in itself, but
it means results computed here would not reproduce results computed from the
original bit for bit, and a reproducibility check would then have to
distinguish "different grid file" from "different answer". The mask is written
as float32 because it holds only 0 and 1, which float32 represents exactly.

    python3 make_gridfile.py --grid .../soca_gridspec.nc --out lv_grid.nc

Then point experiments.yaml at it:

    grid: lv_grid.nc
"""

import argparse
import os
import sys

from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lv_common import Grid, expand  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True, help='full soca_gridspec.nc')
    ap.add_argument('--out', default='lv_grid.nc')
    a = ap.parse_args()

    g = Grid(expand(a.grid))
    ny, nx = g.shape

    out = expand(a.out)
    with Dataset(out, 'w', format='NETCDF4') as d:
        d.createDimension('y', ny)
        d.createDimension('x', nx)
        d.setncattr('title', 'slimmed grid for tools/letkf_verif')
        d.setncattr('source_grid', os.path.abspath(expand(a.grid)))
        for name, arr, kind, units in (('lon', g.lon, 'f8', 'degrees_east'),
                                       ('lat', g.lat, 'f8', 'degrees_north'),
                                       ('area', g.area, 'f8', 'm2'),
                                       ('mask2d', g.mask, 'f4', '1')):
            v = d.createVariable(name, kind, ('y', 'x'), zlib=True,
                                 complevel=4)
            v[:] = arr.astype(kind)
            v.units = units
    print('wrote %s (%.1f MB) -- %d x %d'
          % (out, os.path.getsize(out) / 1e6, ny, nx))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
