#!/usr/bin/env python3
"""Regional strong-current diagnostics against Copernicus L4 ADT.

Broad, branching and eddy-active flows do not have one defensible current
axis. This module locates every configured current on ONE analysis cycle from
maps of geostrophic-current speed above one explicit, shared threshold per
region. The threshold is applied unchanged to Copernicus and to every
experiment, so the maps show both placement and whether the model reaches the
observed strength -- a weak model cannot look equivalent by being scored
against its own percentile.

The one-dimensional jet-centroid and cross-front metrics are kept only for
regions explicitly marked ``axis_diagnostic: true``. They are not calculated
for broad or multi-branch currents, where a single axis is an artefact of the
extraction rather than a property of the flow.

Two caveats belong with every number this produces:

* Copernicus L4 is a mapped, smoothed analysis of much of the same altimeter
  data the experiments assimilate. It is a well-posed common reference, not
  independent truth, and it is over-smoothed in data-sparse regions -- which
  is why a model can legitimately exceed 100% of its mesoscale variance.
* Front POSITION converges over weeks of cycling; a few days cannot move it
  far. Front STRUCTURE -- width, strength, footprint -- is set by the
  covariance and appears immediately. Read structure differences as real and
  position differences as provisional.

This is the computation half. `plot_fronts.py` is the report stage that drives
it and draws the figures.
"""

import os
import sys

import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lv_verif as LV  # noqa: E402

G = 9.81
OMEGA = 7.2921e-5
PROFILE_HALFWIDTH_DEG = 3.0
SMOOTH_SIGMA = 2

# Target resolution of the per-region working grid, in degrees. Both the model
# analysis and the L4 product are interpolated onto it so the speed threshold
# means the same thing for every product in the box.
GRID_STEP_DEG = 0.125

# Geostrophy is undefined at the equator and unusable near it: g/f diverges, so
# a vanishing height gradient still yields a huge "current". Only the Global
# box spans this band, and there it was contributing 10-13% of the strong-
# current footprint -- unevenly, 9.8% for Copernicus against 13.5% for the
# weakest experiment, which flattered exactly the run that needed flattering.
EQUATOR_MASK_DEG = 2.0


def _regions(spec):
    """Validate and normalize ``frontal_analysis.regions``."""
    out = {}
    items = spec.items() if isinstance(spec, dict) else [
        (r.get('name'), r) for r in (spec or [])]
    for name, entry in items:
        entry = dict(entry or {})
        name = name or entry.get('name')
        missing = [key for key in ('lon', 'lat', 'strong_speed_mps')
                   if key not in entry]
        if not name or missing:
            raise ValueError('frontal region %r needs name, lon, lat, and %s'
                             % (name, ', '.join(missing)))
        if len(entry['lon']) != 2 or len(entry['lat']) != 2:
            raise ValueError('frontal region %s needs two lon and lat bounds'
                             % name)
        threshold = float(entry['strong_speed_mps'])
        if threshold <= 0:
            raise ValueError('frontal region %s has non-positive threshold'
                             % name)
        entry.update(name=name, lon=tuple(map(float, entry['lon'])),
                     lat=tuple(map(float, entry['lat'])),
                     strong_speed_mps=threshold,
                     axis_diagnostic=bool(entry.get('axis_diagnostic', False)))
        if entry['axis_diagnostic']:
            if entry.get('kind') not in ('zonal', 'merid'):
                raise ValueError('%s: axis diagnostic needs kind zonal/merid'
                                 % name)
            # Sign matters where two branches of opposite direction share a
            # box: the Brazil Current runs south and the Malvinas north, and a
            # sign-blind search jumps between them from one latitude to the
            # next.
            entry['sign'] = int(entry.get('sign', 1))
        out[name] = entry
    if not out:
        raise ValueError('frontal_analysis.regions is empty')
    return out


def runtime(cfg):
    """Enabled frontal settings from a loaded config, or None if switched off.

    Returns ``(frontal_cfg, regions, cycle)``. Raises only on a configuration
    that is switched on but unusable, so the caller can treat a disabled or
    absent block as "nothing to do" without inspecting it.
    """
    frontal_cfg = cfg.get('frontal_analysis') or {}
    if not frontal_cfg.get('enabled'):
        return None
    adt_name = frontal_cfg.get('adt_product', 'adt')
    if adt_name != 'adt' or adt_name not in LV.configured(cfg):
        raise ValueError('frontal analysis needs verification.products.adt')
    cycle = str(frontal_cfg.get('cycle') or '')
    return frontal_cfg, _regions(frontal_cfg.get('regions')), cycle


def load_model_grid(path):
    """The MOM6 grid is separable in the configured frontal latitude range.

    Below the tripolar cap the gridspec has zero longitude spread within a
    column and zero latitude spread within a row, so the 2-D coordinates
    collapse to two 1-D axes and the region interpolation stays a cheap
    regular-grid lookup. Rows are cut at 62N for exactly that reason.
    """
    with nc.Dataset(path) as dataset:
        lon = np.ma.filled(dataset['lon'][:], np.nan)
        lat = np.ma.filled(dataset['lat'][:], np.nan)
        mask = np.ma.filled(dataset['mask2d'][:], 0)
    if lon.ndim == 3:
        lon, lat = lon[0], lat[0]
    if mask.ndim == 3:
        mask = mask[0]
    rows = np.where((lat[:, 0] > -72) & (lat[:, 0] < 62))[0]
    lat1 = lat[rows, 0].astype(float)
    lon1 = ((lon[0].astype(float) + 180) % 360) - 180
    order = np.argsort(lon1)
    return dict(rows=rows, order=order, lat=lat1, lon=lon1[order],
                mask=mask[rows][:, order] > 0)


def model_ssh(path, grid):
    with nc.Dataset(path) as dataset:
        field = np.ma.filled(dataset['ave_ssh'][:], np.nan)
    field = field[0] if field.ndim == 3 else field
    field = field[grid['rows']][:, grid['order']].astype(float)
    field[~grid['mask']] = np.nan
    return field


def copernicus_adt(path):
    with nc.Dataset(path) as dataset:
        return (np.ma.filled(dataset['adt'][0], np.nan).astype(float),
                np.asarray(dataset['latitude'][:], float),
                np.asarray(dataset['longitude'][:], float))


def check_cycle(cfg, cycle):
    """Raise a clear error unless every displayed product resolves at `cycle`."""
    missing = []
    if LV.product_path(cfg, 'adt', cycle) is None:
        missing.append('Copernicus ADT')
    for experiment in cfg['experiments']:
        if experiment.analysis(cycle) is None:
            # Distinguish the two ways this fails: no pattern configured at
            # all, versus a pattern that matches nothing at this cycle. They
            # need different fixes and used to read identically.
            why = ('no analysis_pattern: configured'
                   if not experiment.analysis_pattern
                   else 'analysis_pattern %r matched nothing'
                        % experiment.analysis_pattern)
            missing.append('%s (%s)' % (experiment.name, why))
    if missing:
        raise FileNotFoundError(
            'frontal-analysis cycle %s is unavailable for %s'
            % (cycle, ', '.join(missing)))


def regrid(field, src_lat, src_lon, tgt_lat, tgt_lon):
    fn = RGI((src_lat, src_lon), field, bounds_error=False, fill_value=np.nan)
    lat2, lon2 = np.meshgrid(tgt_lat, tgt_lon, indexing='ij')
    return fn(np.stack([lat2, lon2], axis=-1))


def geostrophic(eta, lat, lon):
    """Geostrophic velocity components from SSH/ADT on a regular grid.

    Returns NaN within EQUATOR_MASK_DEG of the equator rather than the
    divide-by-zero it would otherwise produce there.
    """
    dy = np.gradient(lat)[:, None] * 111e3
    dx = np.gradient(lon)[None, :] * 111e3 * np.cos(np.deg2rad(lat))[:, None]
    coriolis = 2 * OMEGA * np.sin(np.deg2rad(lat))[:, None]
    coriolis = np.where(np.abs(lat)[:, None] < EQUATOR_MASK_DEG, np.nan,
                        coriolis)
    return (-G / coriolis * np.gradient(eta, axis=0) / dy,
            G / coriolis * np.gradient(eta, axis=1) / dx)


def strong_mask(speed, threshold):
    """Finite speed cells meeting a shared absolute current threshold."""
    return np.isfinite(speed) & (speed >= threshold)


def _centroid_axis(field, coord, sign):
    """Smoothed jet-core centroid along axis 0, only for approved narrow jets.

    A plain argmax produces a square-wave axis that steps between grid rows.
    Taking the weighted centroid of the contiguous above-half-maximum run
    instead gives a continuous axis that tracks the jet core sub-gridscale.
    """
    current = gaussian_filter(np.nan_to_num(sign * field, nan=0.0),
                              sigma=(SMOOTH_SIGMA, SMOOTH_SIGMA))
    axis = np.full(current.shape[1], np.nan)
    peak = np.full(current.shape[1], np.nan)
    for index, column in enumerate(current.T):
        maximum = int(np.argmax(column))
        value = column[maximum]
        if not np.isfinite(value) or value <= 0:
            continue
        lower = upper = maximum
        while lower > 0 and column[lower - 1] >= 0.5 * value:
            lower -= 1
        while upper + 1 < len(column) and column[upper + 1] >= 0.5 * value:
            upper += 1
        weight = column[lower:upper + 1] - 0.5 * value
        axis[index] = (np.sum(weight * coord[lower:upper + 1]) / np.sum(weight)
                       if np.sum(weight) > 0 else coord[maximum])
        peak[index] = value
    if not np.any(np.isfinite(axis)):
        return axis, peak
    return gaussian_filter(np.where(np.isfinite(axis), axis, np.nanmean(axis)),
                           sigma=4), peak


def jet_axis(ug, vg, lat, lon, region):
    if region['kind'] == 'zonal':
        return _centroid_axis(ug, lat, region['sign'])
    return _centroid_axis(vg.T, lon, region['sign'])


def _profile(component, cop_axis, offsets, tlat, tlon, region):
    if region['kind'] == 'zonal':
        profile = np.full((len(offsets), component.shape[1]), np.nan)
        for index in range(component.shape[1]):
            profile[:, index] = np.interp(
                cop_axis[index] + offsets, tlat, component[:, index],
                left=np.nan, right=np.nan)
    else:
        component = component.T
        profile = np.full((len(offsets), component.shape[1]), np.nan)
        for index in range(component.shape[1]):
            profile[:, index] = np.interp(
                cop_axis[index] + offsets, tlon, component[:, index],
                left=np.nan, right=np.nan)
        profile *= region['sign']
    return np.nanmean(profile, axis=1)


def run(cfg, regions, cycle):
    """Analyze each region at one cycle and return plot-ready arrays."""
    grid = load_model_grid(cfg['grid'])
    experiments = {experiment.name: experiment
                   for experiment in cfg['experiments']}
    adt, adt_lat, adt_lon = copernicus_adt(LV.product_path(cfg, 'adt', cycle))
    results = {}
    for name, region in regions.items():
        lat0, lat1 = region['lat']
        lon0, lon1 = region['lon']
        tlat = np.arange(lat0, lat1 + 1e-9, GRID_STEP_DEG)
        tlon = np.arange(lon0, lon1 + 1e-9, GRID_STEP_DEG)
        offsets = np.arange(-PROFILE_HALFWIDTH_DEG,
                            PROFILE_HALFWIDTH_DEG + 1e-9, GRID_STEP_DEG)
        axes, peaks, profiles = {}, {}, {}
        fields = {'copernicus': regrid(adt, adt_lat, adt_lon, tlat, tlon)}
        fields.update({key: regrid(model_ssh(experiment.analysis(cycle), grid),
                                   grid['lat'], grid['lon'], tlat, tlon)
                       for key, experiment in experiments.items()})
        speed, strong, components = {}, {}, {}
        for key, eta in fields.items():
            ug, vg = geostrophic(eta, tlat, tlon)
            speed[key] = np.hypot(ug, vg)
            strong[key] = strong_mask(speed[key], region['strong_speed_mps'])
            components[key] = ug if region.get('kind') == 'zonal' else vg
            if region['axis_diagnostic']:
                axis, peak = jet_axis(ug, vg, tlat, tlon, region)
                axes[key] = np.asarray([axis])
                peaks[key] = np.asarray([peak])
        if region['axis_diagnostic']:
            cop_axis = axes['copernicus'][0]
            for key, component in components.items():
                profiles[key] = np.asarray([
                    _profile(component, cop_axis, offsets, tlat, tlon, region)])
        results[name] = dict(
            cfg=region, tlat=tlat, tlon=tlon, offsets=offsets,
            speed=speed, strong_mask=strong, axes=axes, peaks=peaks,
            profiles=profiles)
    return results


def metrics(results, experiment_names):
    """Axis displacement/sharpness metrics only for approved coherent jets."""
    rows = []
    for name, result in results.items():
        region = result['cfg']
        if not region['axis_diagnostic']:
            continue
        cop_axis = result['axes']['copernicus']
        cop_peak = np.nanmean(np.abs(result['peaks']['copernicus']))
        cop_profile = np.nanmean(result['profiles']['copernicus'], axis=0)
        # A meridional axis is a longitude, so its displacement shrinks with
        # the cosine of latitude before it becomes a distance.
        scale = (np.full(cop_axis.shape[1], 111.0) if region['kind'] == 'zonal'
                 else 111.0 * np.cos(np.deg2rad(result['tlat'])))
        for experiment in experiment_names:
            axis = result['axes'][experiment]
            km = ((axis - cop_axis) * scale[None, :]).ravel()
            km = km[np.isfinite(km)]
            profile = np.nanmean(result['profiles'][experiment], axis=0)

            def fwhm(values):
                peak = np.nanmax(values)
                indices = np.where(values >= peak / 2)[0]
                return ((result['offsets'][indices[-1]]
                         - result['offsets'][indices[0]]) * 111
                        if len(indices) > 1 else np.nan)

            own_peak = float(np.nanmean(np.abs(result['peaks'][experiment])))
            rows.append(dict(
                region=name, kind=region['kind'], experiment=experiment,
                median_abs_km=float(np.median(np.abs(km))),
                bias_km=float(np.mean(km)),
                p90_km=float(np.percentile(np.abs(km), 90)),
                own_axis_speed=own_peak, cop_axis_speed=float(cop_peak),
                speed_pct=float(100 * own_peak / cop_peak),
                composite_peak=float(np.nanmax(profile)),
                composite_pct=float(100 * np.nanmax(profile)
                                    / np.nanmax(cop_profile)),
                fwhm_km=float(fwhm(profile)),
                cop_fwhm_km=float(fwhm(cop_profile))))
    return rows


def print_metrics(rows):
    if not rows:
        print('  no axis metrics requested (no region sets axis_diagnostic)')
        return
    print('  %-22s %-24s %9s %7s %7s %5s %6s %7s'
          % ('region', 'experiment', 'med|off|', 'bias', 'p90', 'u%', 'peak%',
             'FWHM'))
    for row in rows:
        print('  %-22s %-24s %8.0fk %+6.0fk %6.0fk %4.0f%% %5.0f%% %6.0fk'
              % (row['region'], row['experiment'], row['median_abs_km'],
                 row['bias_km'], row['p90_km'], row['speed_pct'],
                 row['composite_pct'], row['fwhm_km']))
