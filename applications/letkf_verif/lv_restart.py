#!/usr/bin/env python3
"""A MOM6 restart as a stand-in background, remapped to the history z grid.

The history and analysis files are on the fixed z grid (`z_l`, the same h in
every column), but a MOM6 restart is on the model's native hybrid layers:
level 30 of a restart is ~130 m in one column and ~190 m in the next, and in
the interior the layers follow density, not depth. Reading a restart level by
level therefore puts a different depth under every pixel. This module
interpolates it, column by column, onto the z levels of the experiment's own
analysis file, and writes the result once per restart as a small z-level file
that every reader then opens like a history file.

The remap is linear in depth between layer centres; above the first centre
the top layer's value is used, and below the sea floor the cell is dry (h=0,
missing). u and v use the thickness averaged onto their C-grid faces.
"""

import os
import time

import numpy as np
from netCDF4 import Dataset

H_MIN = 1e-3        # metres; thinner than this is a vanished layer
# Written, in this order. MLD is left out on purpose: the restart's is ePBL's
# active mixing layer, not the history's delta-rho = 0.125 mixed layer.
VARS3D = ('Temp', 'Salt', 'u', 'v')
VARS2D = ('ave_ssh',)
LOCK_WAIT_SEC = 1800


def _full(ds, name):
    v = ds[name]
    a = v[0] if v.ndim in (3, 4) else v[:]
    return np.ma.filled(a.astype('f4'), np.nan)


def target_dz(analysis_path):
    """Thickness of each history z level, from a z-grid analysis file's h:
    its deepest column, the one that holds every level."""
    with Dataset(analysis_path) as ds:
        h = _full(ds, 'h')
    total = np.nansum(h, axis=0)
    j, i = np.unravel_index(np.nanargmax(total), total.shape)
    return np.nan_to_num(h[:, j, i]).astype('f8')


def _faces(h, axis):
    """Thickness on the C-grid face east (axis=-1, periodic) or north
    (axis=-2, last row repeated) of each tracer cell."""
    if axis == -1:
        nb = np.roll(h, -1, axis=-1)
    else:
        nb = np.concatenate([h[:, 1:], h[:, -1:]], axis=1)
    return 0.5 * (h + nb)


class _Column:
    """Where each z-level mid-depth falls among one grid's layer centres."""

    def __init__(self, h, zt):
        nk = h.shape[0]
        self.zc = np.cumsum(h, axis=0) - 0.5 * h
        self.depth = np.sum(h, axis=0)
        self.wet = h > H_MIN
        self.zt = zt
        # idx[k] = number of layer centres shallower than zt[k], built from
        # one searchsorted per source layer (centres are monotonic in each
        # column) instead of a (nt, nk, ny, nx) comparison
        nt = zt.size
        cnt = np.zeros((nt + 1,) + h.shape[1:], dtype='i2')
        flat = cnt.reshape(nt + 1, -1)
        cols = np.arange(flat.shape[1])
        for s in range(nk):
            pos = np.searchsorted(zt, self.zc[s].ravel(), side='right')
            flat[pos, cols] += 1
        self.idx = np.cumsum(cnt, axis=0)[:nt].astype('i2')
        self.nk = nk

    def remap(self, f):
        """f on the native layers -> f on the z levels (NaN below the floor)."""
        f = f.copy()
        # vanished layers carry the last wet value down, so a centre in the
        # collapsed stack at the floor never pulls in their fill zeros
        for s in range(1, self.nk):
            f[s] = np.where(self.wet[s], f[s], f[s - 1])
        out = np.empty((self.zt.size,) + f.shape[1:], dtype='f4')
        for k, z in enumerate(self.zt):
            i = self.idx[k].astype('i8')
            lo = np.clip(i - 1, 0, self.nk - 1)[None]
            hi = np.clip(i, 0, self.nk - 1)[None]
            z0 = np.take_along_axis(self.zc, lo, 0)[0]
            z1 = np.take_along_axis(self.zc, hi, 0)[0]
            f0 = np.take_along_axis(f, lo, 0)[0]
            f1 = np.take_along_axis(f, hi, 0)[0]
            with np.errstate(invalid='ignore', divide='ignore'):
                w = np.where(z1 > z0, (z - z0) / (z1 - z0), 0.0)
            v = f0 + np.clip(w, 0.0, 1.0) * (f1 - f0)
            out[k] = np.where(z < self.depth, v, np.nan)
        return out


def remap(src_path, dz, out_path):
    """Write ``out_path``: the restart at ``src_path`` on the z levels dz."""
    import lv_statespace as LS         # open_dataset merges MOM.res_N.nc
    ztop = np.concatenate([[0.0], np.cumsum(dz)[:-1]])
    zt = ztop + 0.5 * dz
    tmp = '%s.tmp%d' % (out_path, os.getpid())
    with LS.open_dataset(src_path) as ds, \
            Dataset(tmp, 'w', format='NETCDF4') as out:
        h = _full(ds, 'h')
        ny, nx = h.shape[1:]
        out.createDimension('Time', 1)
        out.createDimension('z_l', zt.size)
        out.createDimension('yh', ny)
        out.createDimension('xh', nx)
        out.source_restart = os.path.basename(src_path)

        def var(name, dims):
            # one chunk per level: the readers go level by level, and the
            # default chunking packs ~40 levels per chunk, so every level
            # read would decompress all of them
            chunks = (1, 1, ny, nx) if len(dims) == 4 else (1, ny, nx)
            return out.createVariable(name, 'f4', dims, zlib=True,
                                      complevel=1, chunksizes=chunks,
                                      fill_value=np.float32(1e20))

        cols = {'t': _Column(h, zt)}
        depth = cols['t'].depth
        hz = np.clip(depth[None] - ztop[:, None, None], 0.0,
                     dz[:, None, None]).astype('f4')
        var('h', ('Time', 'z_l', 'yh', 'xh'))[0] = hz
        z = out.createVariable('z_l', 'f8', ('z_l',))
        z[:] = zt
        for name in VARS3D:
            if name not in ds.variables:
                continue
            grid = {'u': 'u', 'v': 'v'}.get(name, 't')
            if grid not in cols:
                cols[grid] = _Column(_faces(h, -1 if grid == 'u' else -2), zt)
            field = cols[grid].remap(_full(ds, name))
            var(name, ('Time', 'z_l', 'yh', 'xh'))[0] = np.ma.masked_invalid(field)
            del field
        for name in VARS2D:
            if name in ds.variables:
                var(name, ('Time', 'yh', 'xh'))[0] = np.ma.masked_invalid(
                    _full(ds, name))
    os.replace(tmp, out_path)


def remapped(src_path, analysis_path, outdir):
    """Path to the z-level copy of a restart, building it on first use.

    One file per restart, shared by every cycle that falls back on it; a
    lock file stops concurrent precompute workers remapping the same restart
    at once (the others wait for it).
    """
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, os.path.basename(src_path)[:-3] + '.z.nc')
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src_path):
        return out
    lock = out + '.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        t0 = time.time()
        while os.path.exists(lock) and time.time() - t0 < LOCK_WAIT_SEC:
            time.sleep(5)
        return out if os.path.exists(out) else None
    try:
        os.close(fd)
        t0 = time.time()
        remap(src_path, target_dz(analysis_path), out)
        print('\n  restart remapped to z: %s (%.0fs)'
              % (os.path.basename(out), time.time() - t0), flush=True)
    finally:
        os.remove(lock)
    return out
