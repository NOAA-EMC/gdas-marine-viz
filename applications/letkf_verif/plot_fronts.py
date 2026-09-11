#!/usr/bin/env python3
"""Frontal-current figures. One analysis cycle, not the cache.

This is the odd one out among the plot_ stages: every other one reads only the
per-cycle reductions written by compute_cycle.py, while this reads two full
fields -- each experiment's written ocean analysis and the Copernicus L4 ADT
grid -- for a single cycle. That is why it needs `analysis_pattern:` on every
experiment, and why it is optional.

Writes straight into cfg['figs'], the way the other stages do:

    figs/front_strong_<slug>.png   one per configured current box
    figs/front_profiles.png        cross-front composites, coherent jets only
    front_metrics.csv              axis metrics, when any region asks for them

The method, its assumptions and its caveats live in lv_fronts.py.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import lv_fronts as LF  # noqa: E402
from lv_common import load_config  # noqa: E402
from lv_plot import slug  # noqa: E402

LAND = '#e7e3db'
COAST = '#8d8878'

# Copernicus is drawn in near-black against the experiments' tab10 colours: it
# is the shared reference on every panel, not one series among several.
STYLE = {'copernicus': dict(color='#101010', label='Copernicus L4')}


def _keys(result, labels):
    return ['copernicus'] + [key for key in labels if key in result['speed']]


def _label(key, labels):
    return STYLE.get(key, {}).get('label', labels.get(key, key))


def strong_current_map(name, result, figs, cycle, labels):
    """One multi-panel strong-current footprint figure for a current region.

    Shading and outline both come from one synoptic cycle. Every product uses
    the same region-wide ABSOLUTE threshold, so a weak model cannot look
    equivalent simply by being scored against its own percentile.
    """
    keys = _keys(result, labels)
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    region = result['cfg']
    threshold = region['strong_speed_mps']
    all_speed = np.concatenate(
        [result['speed'][key][np.isfinite(result['speed'][key])]
         for key in keys])
    vmax = max(float(np.nanpercentile(all_speed, 99)), threshold * 1.25)
    figure = plt.figure(figsize=(5.0 * ncol, 4.2 * nrow))
    maps = []
    for index, key in enumerate(keys):
        axis = figure.add_subplot(nrow, ncol, index + 1,
                                  projection=ccrs.PlateCarree())
        speed = result['speed'][key]
        strong = result['strong_mask'][key]
        pm = axis.pcolormesh(result['tlon'], result['tlat'], speed,
                             cmap='Blues', vmin=0, vmax=vmax, shading='auto',
                             transform=ccrs.PlateCarree(), rasterized=True)
        if np.any(strong):
            axis.contour(result['tlon'], result['tlat'], strong.astype(float),
                         levels=[0.5],
                         colors=[STYLE.get(key, {}).get('color', '#b23a2f')],
                         linewidths=2.0, transform=ccrs.PlateCarree(), zorder=5)
        axis.add_feature(cfeature.LAND, facecolor=LAND, zorder=6)
        axis.add_feature(cfeature.COASTLINE, linewidth=.55, edgecolor=COAST,
                         zorder=7)
        axis.set_extent([result['tlon'].min(), result['tlon'].max(),
                         result['tlat'].min(), result['tlat'].max()],
                        crs=ccrs.PlateCarree())
        # Boxes are deliberately elongated (the Agulhas return spans 65 deg of
        # longitude and 13 of latitude); a true aspect would waste most of the
        # panel on whitespace.
        axis.set_aspect('auto')
        grid = axis.gridlines(draw_labels=True, linewidth=.3, color='white',
                              alpha=.45)
        grid.top_labels = grid.right_labels = False
        grid.xlabel_style = {'size': 7, 'color': '#666'}
        grid.ylabel_style = {'size': 7, 'color': '#666'}
        finite = np.isfinite(speed)
        fraction = 100 * np.mean(strong[finite]) if np.any(finite) else 0.0
        axis.set_title(_label(key, labels), fontsize=11, fontweight='bold',
                       loc='left', pad=4)
        axis.set_title('%.1f%% strong footprint' % fraction, fontsize=7.5,
                       color='#666', loc='right', pad=4)
        maps.append((axis, pm))
    for index in range(len(keys), nrow * ncol):
        figure.add_subplot(nrow, ncol, index + 1).axis('off')
    # A dedicated axes keeps the shared horizontal colorbar below the panel
    # grid; Matplotlib's automatic placement can otherwise overlap the lower
    # row of cartopy GeoAxes.
    colorbar = figure.colorbar(
        maps[-1][1], cax=figure.add_axes([.14, .075, .72, .022]),
        orientation='horizontal')
    colorbar.set_label('geostrophic speed  (m s$^{-1}$)', fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle('%s: where the current is locally strong' % name,
                    fontsize=15, fontweight='bold', x=.045, ha='left', y=.985)
    figure.text(.045, .947,
                'Cycle %s. Shared threshold %.2f m s$^{-1}$; outline: speed at '
                'or above threshold.' % (cycle, threshold),
                fontsize=9, color='#555', ha='left')
    figure.subplots_adjust(left=.05, right=.985, bottom=.16, top=.83,
                           wspace=.12, hspace=.20)
    path = os.path.join(figs, 'front_strong_%s.png' % slug(name))
    figure.savefig(path, dpi=125, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return path


def profiles(results, figs, cycle, labels, ncol=2):
    """Cross-front composites only for selected, coherent zonal currents."""
    regions = [name for name, result in results.items()
               if result['cfg']['axis_diagnostic']
               and result['cfg'].get('kind') == 'zonal']
    if not regions:
        return None
    nrow = int(np.ceil(len(regions) / ncol))
    figure, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.0 * nrow),
                                squeeze=False)
    axes = axes.ravel()
    keys = ['copernicus'] + list(labels)
    colors = plt.get_cmap('tab10').colors
    for axis, name in zip(axes, regions):
        result = results[name]
        offsets = result['offsets'] * 111.
        for index, key in enumerate(keys):
            if key not in result['profiles']:
                continue
            values = result['profiles'][key][0]
            axis.plot(offsets, values,
                      color=STYLE.get(key, {}).get(
                          'color', colors[index % len(colors)]),
                      lw=2.8 if key == 'copernicus' else 1.8,
                      label=_label(key, labels) if name == regions[0] else None,
                      solid_capstyle='round')
        axis.axvline(0, color='#bbb', lw=.8, zorder=0)
        axis.axhline(0, color='#bbb', lw=.8, zorder=0)
        axis.set_title(name, fontsize=11, fontweight='bold', loc='left', pad=5)
        axis.set_xlim(offsets.min(), offsets.max())
        axis.tick_params(labelsize=8)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', lw=.4, color='#e8e8e8')
    for axis in axes[len(regions):]:
        axis.axis('off')
    for axis in axes[max(0, len(regions) - ncol):len(regions)]:
        axis.set_xlabel('distance across the Copernicus axis (km)',
                        fontsize=8.5)
    for index in range(0, len(regions), ncol):
        axes[index].set_ylabel(
            'along-current geostrophic velocity  (m s$^{-1}$)', fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc='lower center',
                  ncol=min(len(handles), 5), frameon=False, fontsize=9.5,
                  bbox_to_anchor=(.5, -.01))
    figure.suptitle('Cross-front structure of the selected coherent jets',
                    fontsize=15, fontweight='bold', x=.045, ha='left', y=.99)
    figure.text(.045, .955,
                'Cycle %s. Profiles are retained only where a single current '
                'axis is defensible; each column is aligned on the Copernicus '
                'axis.' % cycle, fontsize=9, color='#555', ha='left')
    figure.tight_layout(rect=[0, .04, 1, .91])
    path = os.path.join(figs, 'front_profiles.png')
    figure.savefig(path, dpi=125, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config', nargs='?', default=None,
                    help='path to the experiments yaml (or use --config)')
    ap.add_argument('--config', dest='config_opt', default=None,
                    help='same as the positional argument')
    ap.add_argument('--root', default=None,
                    help='override every experiment root (or $LETKF_VERIF_ROOT)')
    ap.add_argument('--outdir', default=None,
                    help='send cache/figs/report/scorecard here '
                         '(or $LETKF_VERIF_OUTDIR)')
    ap.add_argument('--cache', action='append', default=None,
                    help='extra cache directory to read and merge (repeatable)')
    ap.add_argument('--cycle', default=None,
                    help='override frontal_analysis.cycle in the config')
    ap.add_argument('--regions', nargs='+', default=None,
                    help='subset of configured current names')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt
    t_all = time.time()
    cfg = load_config(a.config, a.root, a.outdir, a.cache)

    settings = LF.runtime(cfg)
    if settings is None:
        print('frontal_analysis is not enabled -- nothing to draw')
        return 0
    frontal_cfg, regions, cycle = settings
    if a.regions:
        wanted = set(a.regions)
        regions = {name: region for name, region in regions.items()
                   if name in wanted}
        if not regions:
            raise SystemExit('no frontal region matched %s' % a.regions)
    cycle = str(a.cycle or cycle)
    if not cycle:
        raise SystemExit('set frontal_analysis.cycle or pass --cycle YYYYMMDDHH')
    LF.check_cycle(cfg, cycle)

    figs = cfg['figs']
    os.makedirs(figs, exist_ok=True)
    print('frontal analysis: cycle %s, %d region(s)' % (cycle, len(regions)),
          flush=True)
    print('figures -> %s' % figs, flush=True)
    results = LF.run(cfg, regions, cycle)

    names = [experiment.name for experiment in cfg['experiments']]
    labels = {experiment.name: experiment.label
              for experiment in cfg['experiments']}
    rows = LF.metrics(results, names)
    LF.print_metrics(rows)

    written = [strong_current_map(name, result, figs, cycle, labels)
               for name, result in results.items()]
    profile = profiles(results, figs, cycle, labels)
    if profile:
        written.append(profile)
    if rows:
        csv_path = os.path.join(cfg['outdir'], 'front_metrics.csv')
        with open(csv_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('wrote %s' % csv_path)
    print('wrote %d frontal figure(s) in %.1fs'
          % (len(written), time.time() - t_all), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
