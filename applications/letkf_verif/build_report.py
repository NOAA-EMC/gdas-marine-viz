#!/usr/bin/env python3
"""Build the self-contained HTML verification report from the cache and figures.

Figures are embedded as data URIs so each page stands alone. Everything shown
comes from cache/<cycle>.json -- the report never re-reads the model output.

The report is 8 pages, one per section, so no single page carries every
figure in the suite (state maps, verif maps and cycling figures across every
field/level/realm/product add up fast -- see WARN_SIZE_MB below). Every page
repeats the masthead, experiment legend and a nav strip to the other 7
sections, so any page works as an entry point, not just the first.
"""

import argparse
import base64
import glob
import html
import os
import re
import sys
import tarfile
from string import Template

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lv_plot as P  # noqa: E402
import lv_verif as LV  # noqa: E402
import plot_statespace as PS  # noqa: E402
from lv_common import basin_regions, load_config, region_list  # noqa: E402

# Page size at which we say something in the log. ADVISORY ONLY: nothing is
# dropped, downscaled or refused above it, and the exit code is unaffected --
# the report is written in full either way. It exists because publishing a
# page as an artifact caps around here, which is a property of that one
# downstream use, not of the report. Local viewing and the tarball do not
# care, so this must never become a gate.
WARN_SIZE_MB = 16.0

# ---------------------------------------------------------------------------
# html helpers
# ---------------------------------------------------------------------------


def fig_path(figs, base, cycle):
    """Per-cycle figures are tagged '<base>_<cycle>.png' when several cycles
    are rendered; fall back to the untagged name for a single-cycle run."""
    for cand in ('%s_%s.png' % (base, cycle), '%s.png' % base):
        p = os.path.join(figs, cand)
        if os.path.exists(p):
            return p
    return os.path.join(figs, '%s.png' % base)


_MISSING = []

# Figures are LINKED from the pages by default -- '<img src="figs/x.png">' --
# and the tarball carries every PNG a page references. Embedding them as
# base64 made a page grow with every field, region and now every cycle it
# shows; with a date menu over the background and gridded-product views that
# was heading past 100 MB a page. ``EMBED`` (--embed) restores the old
# single-file pages for the odd case where one HTML file has to travel alone.
EMBED = False
FIG_REL = 'figs'      # page -> figs directory, set by main()
_USED = {}            # absolute figure path -> path inside the tarball


def hemi_imgs(figs, base, cycle, alt, optional=False):
    """Both hemispheres of a polar figure, in order."""
    return ''.join(
        img(fig_path(figs, '%s_%s' % (base, h), cycle),
            '%s (%s)' % (alt, label), optional=optional)
        for h, label in (('nh', 'Arctic'), ('sh', 'Antarctic')))


def img(path, alt, optional=False):
    """A figure, linked (or with --embed, inlined). A missing one is
    reported, never silently dropped.

    Silently returning '' let renamed figures vanish from the report while
    every script still reported success. Linked images load lazily: a page
    with a date menu over a dozen map families references hundreds of PNGs,
    and the browser only needs the ones on screen.
    """
    if not os.path.exists(path):
        if not optional:
            _MISSING.append(os.path.basename(path))
        return ''
    if EMBED:
        with open(path, 'rb') as f:
            src = 'data:image/png;base64,%s' % base64.b64encode(f.read()).decode()
        lazy = ''
    else:
        rel = '%s/%s' % (FIG_REL, os.path.basename(path))
        _USED[os.path.abspath(path)] = rel
        src = html.escape(rel)
        lazy = ' loading="lazy"'
    # The link is the zoom: with JS it opens the lightbox in HEAD (fit to the
    # window, click again for 1:1 pixels, click outside or Esc to close);
    # without it, or middle-clicked, it is just the PNG in its own tab.
    return ('<figure class="fig"><div class="fig-scroll">'
            '<a class="fig-zoom" href="%s" onclick="return lvZoom(this)" '
            'title="%s">'
            '<img src="%s" alt="%s"%s></a></div></figure>'
            % (src, html.escape(alt), src, html.escape(alt), lazy))


def picker_widget(group, items, label_fn, panel_fn, style_fn=None,
                  collapsible=None, dropdown=None, default=None):
    """Pure-CSS radio-button tabs, or (``dropdown``) a native <select>: one
    panel visible at a time. At most one of ``collapsible``/``dropdown``.

    The chip form (the default) needs no JavaScript at all: each item gets
    a hidden radio input plus a label styled as a chip, and one generated
    CSS rule per item reveals that item's panel when its radio is checked
    (the first is checked by default). ``group`` scopes the radio group
    name and every id, so more than one picker (region, obs type, ...) can
    live on the same page without their ids or radio groups colliding.

    ``collapsible``, if given (a <summary> label string), tucks the chip row
    itself inside a <details> disclosure, collapsed by default -- for a
    picker with many items where the chip row is the clutter, not just the
    panels. That nests the radios one level deeper than '.picker-panels', so
    the plain ":checked ~ sibling" rule used otherwise can no longer reach
    across to it; this switches to a ":has()" rule instead, which does not
    require a shared parent. Every browser this report has ever needed to
    support already ships :has() (Chrome/Edge and Safari since early 2022,
    Firefox since Dec 2023).

    ``dropdown``, if given (a label used as the <select>'s accessible name),
    renders a native <select> instead of chips -- the one widget in this
    whole report with any JavaScript, because CSS genuinely has no rule for
    this one: nothing lets a <select>'s chosen VALUE affect an unrelated
    sibling's display the way :checked does for a radio. The swap is a few
    lines of inline vanilla JS on the select's "change" event; nothing else
    on the page needs or uses JS, and ``style_fn`` (no reliable cross-browser
    way to colour an <option>) is ignored in this mode.

    ``label_fn``/``panel_fn``/``style_fn`` take one raw item (not its slug)
    and return its chip/option label, its panel's inner HTML, and (chip mode
    only) an optional ` style="..."` string for the chip (e.g. to colour-
    match a companion map) respectively. ``default`` names the item shown
    first (the first in ``items`` when not given) -- a date menu opens on
    the latest cycle while still listing them oldest first.
    """
    tabs, rules, panels, options = [], [], [], []
    first = default if default in items else items[0] if items else None
    for i, item in enumerate(items):
        s = P.slug(item)
        tid, pid = '%s-tab-%s' % (group, s), '%s-panel-%s' % (group, s)
        label = label_fn(item)
        default = item == first
        if dropdown:
            options.append('<option value="%s"%s>%s</option>'
                           % (pid, ' selected' if default else '',
                              html.escape(label)))
        else:
            style = style_fn(item) if style_fn else ''
            tabs.append(
                '<input type="radio" name="%s" id="%s" class="picker-tab"%s>'
                '<label for="%s" class="picker-chip"%s>%s</label>'
                % (group, tid, ' checked' if default else '', tid, style,
                   html.escape(label)))
            if collapsible:
                rules.append(
                    '.picker-widget:has(#%s:checked) .picker-panels #%s'
                    '{display:block}' % (tid, pid))
            else:
                rules.append(
                    '#%s:checked ~ .picker-panels #%s{display:block}'
                    % (tid, pid))
        panels.append(
            '<div id="%s" class="picker-panel%s">%s</div>'
            % (pid, ' default' if (dropdown and default) else '',
               panel_fn(item)))
    if not (tabs or options):
        return ''

    if dropdown:
        chooser = (
            '<label class="picker-row"><span class="picker-label">%s</span>'
            '<select class="picker-select" aria-label="%s" '
            'onchange="lvPickerShow(this)">%s</select></label>'
            # ":scope >" so this only ever touches its OWN panels. A plain
            # ".picker-panel" search also matches the panels of any picker
            # NESTED inside one of them, and inline display:none beats the
            # nested picker's own stylesheet rule -- which silently blanked
            # an inner region picker the first time the outer menu changed.
            '<script>function lvPickerShow(s){'
            'var w=s.closest(".picker-widget");'
            'var p=w.querySelectorAll('
            '":scope > .picker-panels > .picker-panel");'
            'for(var i=0;i<p.length;i++){p[i].style.display="none"}'
            'var t=document.getElementById(s.value);'
            'if(t){t.style.display="block"}}</script>'
            % (html.escape(dropdown), html.escape(dropdown),
               ''.join(options)))
    elif collapsible:
        chooser = (
            '<details class="picker-chooser"><summary>%s</summary>'
            '<div class="picker-chips">%s</div></details>'
            % (html.escape(collapsible), ''.join(tabs)))
    else:
        chooser = ''.join(tabs)

    return ('<style>%s</style>'
            '<div class="picker-widget">%s'
            '<div class="picker-panels">%s</div></div>'
            % (''.join(rules), chooser, ''.join(panels)))


def basin_colors(cfg):
    """{basin name: hex colour}, matching ocean_regions.png's legend order.

    Shared by every region picker so a basin's chip is always the same
    colour as its patch on that map -- same P.SERIES assignment, same basin
    order. Empty when no `ocean_basin_mask:` is configured.
    """
    mask_path = cfg.get('ocean_basin_mask')
    if not mask_path:
        return {}
    return {name: P.SERIES[i % len(P.SERIES)]
            for i, (_code, name) in enumerate(basin_regions(mask_path))}


def _basin_chip_style(basin_color, r):
    """`regions:` boxes and 'global' (which ocean_regions.png draws as
    outlines, not fills) get a neutral chip instead of chasing exact
    polygon geometry for a colour that map never gave them either."""
    c = basin_color.get(r)
    return (' style="border-color:%s;background:%s1f;color:%s"'
            % (c, c, c)) if c else ''


def verif_widget(cycles, cfg, names, figs):
    """Region picker + one panel per region for 'fit to gridded analyses'.

    The product/region presence filter here MUST match plot_timeseries.
    fig_verif_series exactly: that is what decided which verif_region_*.png
    files exist to embed.
    """
    prods = [p for p in LV.PRODUCTS
             if any(P.get(d, 'state', n, 'ocean', 'verif', p, default=None)
                    for d in cycles.values() for n in names)]
    if not prods:
        return ''
    regions = ['global'] + region_list(cfg)
    regions = [r for r in regions
               if any(P.get(d, 'state', n, 'ocean', 'verif', p, 'bkg', r,
                            default=None)
                      for d in cycles.values() for n in names for p in prods)]
    if not regions:
        return ''

    basin_color = basin_colors(cfg)
    return picker_widget(
        'verif', regions,
        label_fn=lambda r: r.replace('_', ' '),
        panel_fn=lambda r: img(
            os.path.join(figs, 'verif_region_%s.png' % P.slug(r)),
            '%s: fit to gridded analyses' % r.replace('_', ' '),
            optional=True),
        style_fn=lambda r: _basin_chip_style(basin_color, r))


def _verif_pair(figs, prod, cycle):
    """Product-beside-model maps then model-minus-product, for one date;
    the ice-realm product comes as a hemisphere pair."""
    hemi = LV.product_realm(prod) == 'ice'
    when = PS.cycle_row_label(cycle)
    up = prod.upper()
    if hemi:
        return (hemi_imgs(figs, 'verif_maps_%s' % prod, cycle,
                          '%s: product beside the model background and '
                          'analysis, %s' % (up, when), optional=True)
                + hemi_imgs(figs, 'verif_diff_%s' % prod, cycle,
                            'Model minus %s, %s' % (up, when), optional=True))
    return (img(fig_path(figs, 'verif_maps_%s' % prod, cycle),
                '%s: product beside the model background and analysis, %s'
                % (up, when), optional=True)
            + img(fig_path(figs, 'verif_diff_%s' % prod, cycle),
                  'Model minus %s, %s' % (up, when), optional=True))


def regional_series_widget(group, cfg, figs, base, label):
    """Region chips over per-region time-series figures ('<base>_<slug>.png'),
    the way verif_widget does for the gridded-analysis scores."""
    regions = ['global'] + region_list(cfg)
    regions = [r for r in regions if os.path.exists(
        os.path.join(figs, '%s_%s.png' % (base, P.slug(r))))]
    if not regions:
        return ''
    basin_color = basin_colors(cfg)
    return picker_widget(
        group, regions,
        label_fn=lambda r: r.replace('_', ' '),
        panel_fn=lambda r: img(
            os.path.join(figs, '%s_%s.png' % (base, P.slug(r))),
            '%s: %s' % (r.replace('_', ' '), label), optional=True),
        style_fn=lambda r: _basin_chip_style(basin_color, r))


def stability_widget(cfg, figs):
    """SSH cycling: tendency maps, the tendency strip, the four series and
    their trend verdict, per experiment, from plot_stability.py's outputs
    (figs/ssh_tendency_<exp>.png, figs/ssh_tendency_strip_<exp>.png,
    figs/cycle_ssh_stability_<exp>.png, ssh_stability_<exp>.json)."""
    entries = []
    for e in cfg['experiments']:
        slug = P.slug(e.name)
        series = os.path.join(figs, 'cycle_ssh_stability_%s.png' % slug)
        jpath = os.path.join(cfg['outdir'], 'ssh_stability_%s.json' % slug)
        if not os.path.exists(series):
            continue
        block = img(os.path.join(figs, 'ssh_tendency_%s.png' % slug),
                    '%s: latest cycle -- background SSH, increment, 6-h '
                    'tendency and its 5-degree low-pass' % e.name, optional=True)
        block += img(os.path.join(figs, 'ssh_tendency_strip_%s.png' % slug),
                     '%s: low-pass 6-h tendency of the last four cycles' % e.name,
                     optional=True)
        block += img(series, '%s: SSH cycling series against cycle, dotted '
                     'lines are the fitted trends' % e.name)
        if os.path.exists(jpath):
            import json
            with open(jpath) as fh:
                v = json.load(fh)
            rows_by_metric = {}
            for row in v.get('verdict', []):
                rows_by_metric.setdefault(row['metric'], {})[row['region']] = row
            regions = list(dict.fromkeys(row['region'] for row in v['verdict']))
            cells = []
            for m, per in rows_by_metric.items():
                line = [html.escape(m)]
                for r in regions:
                    row = per.get(r)
                    if row is None or row.get('pct_per_day') is None:
                        line.append('&mdash;')
                        continue
                    txt = '%+.1f <span class="dim">(t %.1f)</span>' % (
                        row['pct_per_day'], row['t'] or 0.0)
                    line.append('<span class="bad">%s</span>' % txt
                                if row.get('flag') else txt)
                cells.append(line)
            block += (
                '<p class="lede">Trend of each series over the %d cycles, in '
                '%% of its mean per day, with the t statistic; '
                '<span class="bad">red</span> = significant (|t| &gt; 3) in '
                'the direction that means trouble.</p>' % len(v['cycles'])
                + table(['metric'] + [r.replace('_', ' ') for r in regions],
                        cells))
        entries.append((e.name, block))
    if not entries:
        return ''
    lede = (
        '<section class="subsection"><h3>SSH cycling</h3>'
        '<p class="lede">The <b>6-h tendency</b> &mdash; background(t) minus '
        'the previous analysis &mdash; is what the model does on its own '
        'between two analyses. Healthy, it is the barotropic response to the '
        'winds: about 1.5 cm RMS, large scale, and the same sign from one '
        'cycle to the next. When the analysis hands the model something it '
        'cannot hold, this is where it shows first: the tendency grows, its '
        'large-scale part reverses sign every cycle (the ocean ringing '
        'through a geostrophic adjustment), and the maps show basin-scale '
        'patterns that have nothing to do with the increment.</p>'
        '<p class="lede">Maps of the latest cycle, the low-pass tendency of the '
        'last four cycles side by side, then four series with their trend; '
        'the definitions are under <i>How this is computed</i> at the top of '
        'this section.</p>')
    return lede + figure_menu('stability', 'experiment', entries) + '</section>'


def frontal_widget(cfg, figs):
    """Configured strong-current maps plus the narrow-jet profile diagnostic.

    plot_fronts.py draws them for the cycles build_comparison.py --hours
    selects (00z plus the latest by default), tagged '_<cycle>' like the
    state-space figures, so each current gets a date menu inside its chip.
    """
    frontal = cfg.get('frontal_analysis') or {}
    if not frontal.get('enabled'):
        return ''
    entries = frontal.get('regions') or []
    entries = (entries.values() if isinstance(entries, dict) else entries)
    order = [str(c) for c in cfg['cycles']]
    regions = {}
    for entry in entries:
        if not entry or not entry.get('name'):
            continue
        base = 'front_strong_%s' % P.slug(entry['name'])
        dates = rendered_cycles(figs, re.escape(base), order)
        if dates:
            regions[entry['name']] = (base, dates)
    if not regions:
        return ''
    picker = picker_widget(
        'fronts', list(regions),
        label_fn=lambda name: name,
        panel_fn=lambda name: cycle_menu(
            'fronts-%s' % P.slug(name), regions[name][1],
            lambda c, name=name: img(
                fig_path(figs, regions[name][0], c),
                '%s: geostrophic current speed and strong-current footprint, %s'
                % (name, PS.cycle_row_label(c)), optional=True)
            + img(fig_path(figs, 'front_sst_%s' % P.slug(name), c),
                  '%s: sea surface temperature, OSTIA beside each analysis, %s'
                  % (name, PS.cycle_row_label(c)), optional=True)),
        collapsible='choose current')
    profile = cycle_menu(
        'frontprof', rendered_cycles(figs, 'front_profiles', order),
        lambda c: img(fig_path(figs, 'front_profiles', c),
                      'Cross-front structure for selected coherent jets, %s'
                      % PS.cycle_row_label(c), optional=True))
    return (
        '<section class="subsection"><h3>Frontal-current placement</h3>'
        '<p class="lede">Broad and branching currents are shown as the area '
        'where geostrophic speed reaches a fixed <b>absolute threshold</b>, '
        'rather than being forced into one artificial axis. Each panel is one '
        'analysis cycle &mdash; pick the current, then the cycle; each '
        'current comes with its sea surface temperature, OSTIA beside each '
        'analysis with the same outlines, so the jet can be read against the '
        'temperature front it should sit on. The '
        'configured threshold is identical for Copernicus '
        'L4 ADT and every experiment in that current&rsquo;s box. The Global '
        'box is too big for an outline to read, so it shows the share of each '
        '2&deg; square at or above the threshold instead, with each '
        'experiment&rsquo;s difference from Copernicus beneath it. Copernicus L4 '
        'is a higher-resolution mapped analysis of much of the same altimeter '
        'information, not independent truth.</p>%s%s</section>' % (picker, profile))


def obstype_dropdown_widget(cycles, figs, base, label, cfg=None):
    """Obs-type dropdown + one panel per type.

    Shared by 'fit to observations' (obsfit_type_*.png) and 'observation
    usage' (obscount_type_*.png) -- both are per-obs-type figure sets with
    20-30 items, too many for even a collapsed chip row to feel light, so
    both use picker_widget's ``dropdown`` mode (a native <select>) instead.

    Filtered to types whose PNG actually exists rather than every type in
    the cache: a type can be present with nothing plottable for it (e.g.
    every departure QC'd away), which the figure-writing side then draws
    nothing for -- without this filter that type still got an option, just
    one that opened onto a blank panel.
    """
    types = sorted({t for d in cycles.values() for t in d.get('obs', {})})

    def path(t):
        # 'obsfit_type_<slug>.png' / 'obscount_type_<slug>.png', but the
        # per-type cycling figures are plain 'cycle_<type>.png'
        return os.path.join(figs, ('cycle_%s.png' % t if base == 'cycle'
                                   else '%s_type_%s.png' % (base, P.slug(t))))
    types = [t for t in types if os.path.exists(path(t))]

    def panel(t):
        whole = img(path(t), '%s: %s' % (P.short(t), label), optional=True)
        if base != 'obsfit':
            return whole
        # Section 01 also has the same pair per region stratum
        # (obsfit_type_<type>_region_<slug>.png, from the strata the cache
        # carries: basins, `regions:` boxes, NH/SH for ice). Region chips
        # inside the type panel, 'global' first; only the regions this type
        # actually has files for.
        stem = os.path.join(figs, '%s_type_%s_region_' % (base, P.slug(t)))
        regs = sorted(f[len(stem):-4] for f in glob.glob(stem + '*.png'))
        if not regs:
            return whole
        basin_color = basin_colors(cfg) if cfg else {}
        return picker_widget(
            '%s-%s' % (base, P.slug(t)), ['global'] + regs,
            label_fn=lambda r: r.replace('_', ' '),
            panel_fn=lambda r: whole if r == 'global' else img(
                '%s%s.png' % (stem, r),
                '%s, %s: %s' % (P.short(t), r.replace('_', ' '), label),
                optional=True),
            style_fn=lambda r: _basin_chip_style(basin_color, r))

    return picker_widget(base, types, label_fn=P.short, panel_fn=panel,
                         dropdown='obs type')


def binned_widget(cycles, cfg, figs):
    """Obs type -> view -> date, over plot_obsbins.py's figures.

    Dates come from the files ('_<cycle>' tags plus '_all' for the pooled
    figure), so a type drawn for fewer dates simply lists fewer.
    """
    order = [str(c) for c in sorted(cycles)]
    types = sorted({t for d in cycles.values() for t in d.get('obs', {})})
    views = [('map', 'Maps: count, O-B, O-A, obs error')]
    # the profile types' maps per depth layer (lv_obsbins.layers)
    import lv_obsbins as B
    for lo, hi in B.layers(cfg):
        slug = B.layer_slug(lo, hi)
        views.append(('map_%s' % slug, 'Maps, %s (profiles)'
                      % slug.replace('m-bottom', ' m to the bottom')
                      .replace('m', ' m')))
    views += [('reg', 'Regression: observation against model'),
              ('sec', 'Depth x latitude (profiles)')]

    def date_menu(t, view):
        base = 'obsbins_%s_%s' % (view, P.slug(t))
        dates = rendered_cycles(figs, re.escape(base), order)
        pooled = os.path.join(figs, '%s_all.png' % base)
        entries = []
        if os.path.exists(pooled):
            entries.append(('all', img(pooled, '%s: %s, every cached cycle '
                                       'pooled' % (P.short(t), view),
                                       optional=True)))
        entries += [(c, img(fig_path(figs, base, c),
                            '%s: %s, %s' % (P.short(t), view,
                                            PS.cycle_row_label(c)),
                            optional=True)) for c in dates]
        entries = [(c, h) for c, h in entries if h]
        if not entries:
            return ''
        by = dict(entries)
        return picker_widget(
            'binned-%s-%s' % (view, P.slug(t)), [c for c, _h in entries],
            label_fn=lambda c: 'all cycles' if c == 'all'
            else PS.cycle_row_label(c),
            panel_fn=lambda c: by[c], dropdown='cycle',
            default='all')

    def type_panel(t):
        return figure_menu('binned-%s' % P.slug(t), 'view', [
            (label, date_menu(t, view)) for view, label in views])

    types = [t for t in types if any(
        rendered_cycles(figs, re.escape('obsbins_%s_%s' % (v, P.slug(t))),
                        order)
        or os.path.exists(os.path.join(figs, 'obsbins_%s_%s_all.png'
                                       % (v, P.slug(t))))
        for v, _l in views)]
    if not types:
        return ''
    return picker_widget('binned', types, label_fn=P.short,
                         panel_fn=type_panel, dropdown='obs type')


def obsfit_widget(cycles, cfg, figs):
    """Obs-type picker + one panel per type for 'fit to observations'."""
    return obstype_dropdown_widget(cycles, figs, 'obsfit',
                                   'RMS and bias of the fit to observations '
                                   'across cycles', cfg=cfg)


def counts_widget(cycles, cfg, figs):
    """Obs-type picker + one panel per type for 'observation usage'."""
    return obstype_dropdown_widget(cycles, figs, 'obscount',
                                   'observations assimilated per cycle')


def profiles_widget(data, cfg, figs, cycle=None):
    """Region picker + one panel per region for profile obs departures.

    Same pure-CSS tabs and basin colouring as verif_widget. Filtered to
    regions whose PNG actually exists (mirroring obsfit_widget) since a
    region can be configured with no stratified profile data at all -- e.g.
    a `regions:` box drawn for the gridded-analysis or correlation-length
    figures, which fig_profiles then has nothing to plot for.
    """
    types = [t for t in sorted(data['obs']) if data['obs'][t].get('is_profile')]
    if not types:
        return ''
    regions = ['global'] + region_list(cfg)
    regions = [r for r in regions if os.path.exists(
        fig_path(figs, 'obs_profiles_region_%s' % P.slug(r), cycle))]
    if not regions:
        return ''

    basin_color = basin_colors(cfg)
    return picker_widget(
        'profiles-%s' % (cycle or 'last'), regions,
        label_fn=lambda r: r.replace('_', ' '),
        panel_fn=lambda r: img(
            fig_path(figs, 'obs_profiles_region_%s' % P.slug(r), cycle),
            '%s: profile observations against depth, %s'
            % (r.replace('_', ' '), PS.cycle_row_label(cycle)), optional=True),
        style_fn=lambda r: _basin_chip_style(basin_color, r))


def regional_widget(group, cfg, figs, last, base, label):
    """Region picker + one panel per region for a fig_regional_profiles figure.

    Shared by 'RMS increment by region' (state space) and 'background mean
    by region' (background state) -- both come from the one
    fig_regional_profiles() in plot_statespace.py, which writes one
    '<base>_region_<slug>[_<cycle>].png' per region instead of one grid with
    a column per region. fig_path() resolves the per-cycle tag the same way
    every other state-space figure does.
    """
    regions = ['global'] + region_list(cfg)
    regions = [r for r in regions if os.path.exists(
        fig_path(figs, '%s_region_%s' % (base, P.slug(r)), last))]
    if not regions:
        return ''

    basin_color = basin_colors(cfg)
    return picker_widget(
        group, regions,
        label_fn=lambda r: r.replace('_', ' '),
        panel_fn=lambda r: img(
            fig_path(figs, '%s_region_%s' % (base, P.slug(r)), last),
            '%s: %s by region' % (r.replace('_', ' '), label), optional=True),
        style_fn=lambda r: _basin_chip_style(basin_color, r))


_TAGGED = re.compile(r'^(.*)_(\d{10})\.png$')


def rendered_cycles(figs, stem_re, cycles):
    """Cached cycles that have a per-cycle figure whose stem matches
    ``stem_re`` (a regex, matched in full against the name before the
    '_<cycle>.png' tag), oldest first.

    plot_statespace.py draws the per-date figures for a subset of the cached
    cycles (build_comparison.py --hours: the 00z ones plus the latest), so
    the menu is read off the files that exist rather than off the cache. A
    single-cycle run writes untagged names; those count as the last cycle,
    which is how fig_path() resolves them too.
    """
    pat = re.compile(stem_re)
    order = sorted(str(c) for c in cycles)
    tagged, untagged = set(), False
    for f in os.listdir(figs):
        m = _TAGGED.match(f)
        if m and pat.fullmatch(m.group(1)):
            tagged.add(m.group(2))
        elif not m and f.endswith('.png') and pat.fullmatch(f[:-4]):
            untagged = True
    have = [c for c in order if c in tagged]
    if not have and untagged:
        have = order[-1:]
    return have


def cycle_menu(group, dates, panel_fn):
    """A date <select> over per-cycle figure blocks, opening on the latest.

    ``panel_fn`` takes a cycle string and returns that date's block; empty
    blocks drop out and a lone survivor is returned bare, like figure_menu.
    """
    entries = [(c, panel_fn(c)) for c in dates]
    entries = [(c, h) for c, h in entries if h]
    if not entries:
        return ''
    if len(entries) == 1:
        return entries[0][1]
    by_cycle = dict(entries)
    default = DEFAULT_CYCLE if DEFAULT_CYCLE in by_cycle else entries[-1][0]
    return picker_widget(group, [c for c, _h in entries],
                         label_fn=PS.cycle_row_label,
                         panel_fn=lambda c: by_cycle[c],
                         dropdown='cycle', default=default)


def figure_menu(group, label, entries):
    """A <select> over several finished figure blocks; one visible at a time.

    The state and background sections carry a dozen tall map grids between
    them, which as a flat stack is minutes of scrolling to reach the last
    one. ``entries`` is [(menu label, html)]; empty blocks drop out, and a
    lone survivor is returned bare rather than behind a one-item menu.

    Entries may themselves contain a picker (the region and field pickers
    are passed straight in) -- see the ":scope >" note in picker_widget for
    what makes that nesting safe.
    """
    entries = [(t, h) for t, h in entries if h]
    if not entries:
        return ''
    if len(entries) == 1:
        return entries[0][1]
    by_label = dict(entries)
    return picker_widget(group, [t for t, _h in entries],
                         label_fn=lambda t: t,
                         panel_fn=lambda t: by_label[t],
                         dropdown=label)


def sections_widget(group, cfg, figs, last, kind):
    """Field picker + one panel per field for fig_sections().

    plot_statespace.py writes one 'state_sections_<kind>_<var>[_<cycle>].png'
    per field rather than one grid carrying every field and every transect.
    The field list is derived the way regional_widget derives its regions --
    by checking which files were actually written, so a field with no vertical
    structure (ave_ssh, MLD) simply never appears.
    """
    order = {'incr': cfg.get('state_vars', {}).get('ocean', []),
             'bkg': cfg.get('background_vars', {}).get('ocean', []),
             'woa_bias': list(PS.lv_woa.VARS)}[kind]
    base = 'state_sections_%s' % kind
    what = {'incr': 'analysis increment', 'bkg': 'background state',
            'woa_bias': 'background minus the WOA23 climatology'}[kind]
    # field x depth view, flat rather than nested: a shallow cut and a full
    # column are two views of one field, and a second picker level to reach
    # them costs more than the flat list of labels does.
    paths = {}
    for v in order:
        for _cut, suffix, label in PS.section_depth_views(cfg):
            p = fig_path(figs, '%s_%s%s' % (base, v, suffix), last)
            if os.path.exists(p):
                paths['%s, %s' % (v, label)] = (v, p)
    if not paths:
        return ''
    return picker_widget(
        group, list(paths),
        label_fn=lambda t: t,
        panel_fn=lambda t: img(
            paths[t][1],
            '%s: %s along each configured vertical section (%s)'
            % (paths[t][0], what, t.split(', ', 1)[1]),
            optional=True))


def sequence_widget(group, cfg, figs, cycles):
    """Field menu, then a date menu, over the across-date increment figures.

    plot_statespace.py writes one 'seq_<realm>_incr_<field>_k<lev>[_<hemi>]
    _<cycle>.png' per configured field and cached cycle (it used to be one
    tall grid of up to eight subsampled dates per field). The outer menu is
    ordered from `state_vars` -- the same list that decides which sequences
    get drawn -- so it reads in config order rather than the alphabetical
    order a plain directory glob produced, and the level labels carry their
    approximate depth the way the figures' own titles do. The inner menu
    lists every date that has the figure, oldest first, and shows one at a
    time. Any seq_ file the config no longer names is still appended rather
    than silently dropped, so a stale figure is visible instead of invisible.
    The '_all' figure -- the mean over every cached cycle, on its own
    colour scale -- heads the date menu and is the default, as for the
    binned departures: the systematic increment is the first thing to look
    at, the dates are how it came about.
    """
    order = sorted(cycles)
    data = cycles[order[-1]]
    levels = cfg.get('map_levels', [0])
    svars = cfg.get('state_vars', {})
    want = []
    for v in svars.get('ocean', []):
        flat = v in PS.NO_LEVEL_LABEL
        for k in ([0] if flat else levels):
            want.append(('seq_ocean_incr_%s_k%d' % (v, k),
                         v if flat else '%s, %s' % (v, PS.level_label(data, k))))
    for v in svars.get('ice', []):
        for h in ('nh', 'sh'):
            want.append(('seq_ice_incr_%s_k0_%s' % (v, h),
                         '%s, %s' % (v, PS.HEMIS[h][0])))
    named = {b for b, _label in want}
    tagged = re.compile(r'^(seq_.+)_(\d{10})\.png$')
    stale = sorted({m.group(1) for m in map(tagged.match, os.listdir(figs))
                    if m and m.group(1) not in named})
    want += [(b, b[4:]) for b in stale]

    entries = []
    for base, label in want:
        dates = [c for c in order
                 if os.path.exists(os.path.join(figs, '%s_%s.png' % (base, c)))]
        dates += sorted({m.group(2) for m in map(tagged.match, os.listdir(figs))
                         if m and m.group(1) == base and m.group(2) not in order})
        if os.path.exists(os.path.join(figs, '%s_all.png' % base)):
            dates.insert(0, 'all')
        if not dates:
            # A configured field legitimately has no sequence when only one
            # cycle is cached, or when no experiment writes it; figure_menu
            # drops the empty entry and unwraps a lone survivor.
            entries.append((label, ''))
            continue
        entries.append((label, picker_widget(
            '%s-%s' % (group, P.slug(base)), dates,
            label_fn=lambda c: ('mean over all cycles' if c == 'all'
                                else PS.cycle_row_label(c)),
            panel_fn=lambda c, base=base, label=label: img(
                os.path.join(figs, '%s_%s.png' % (base, c)),
                'Increment across dates: %s, %s'
                % (label, 'mean over every cached cycle' if c == 'all'
                   else PS.cycle_row_label(c)), optional=True),
            dropdown='date', default='all' if dates[0] == 'all' else None)))
    return figure_menu(group, 'increment sequence', entries)


def table(headers, rows, cls=''):
    h = ''.join('<th>%s</th>' % c for c in headers)
    body = ''.join('<tr>%s</tr>'
                   % ''.join('<td>%s</td>' % c for c in r) for r in rows)
    return ('<div class="tbl-scroll"><table class="%s"><thead><tr>%s</tr>'
            '</thead><tbody>%s</tbody></table></div>' % (cls, h, body))


def fmt(v, nd=3):
    return '&mdash;' if v is None or not np.isfinite(v) else '%.*f' % (nd, v)


def delta_cell(v, ref, nd=4, lower_better=True):
    if not (np.isfinite(v) and np.isfinite(ref)) or ref == 0:
        return fmt(v, nd)
    pc = 100.0 * (v - ref) / abs(ref)
    good = (pc < 0) if lower_better else (pc > 0)
    cls = 'good' if abs(pc) >= 1 and good else ('bad' if abs(pc) >= 1 else 'flat')
    return '%.*f <span class="delta %s">%+.1f%%</span>' % (nd, v, cls, pc)


def _depth_note(z):
    return '' if z is None else ('~%s m' % (('%.0f' % z) if z >= 10
                                            else ('%.1f' % z)))


def target_cell(v, nd=3):
    """Colour by distance from 1 in log space."""
    if not np.isfinite(v) or v <= 0:
        return fmt(v, nd)
    d = abs(np.log(v))
    cls = 'good' if d < 0.18 else ('flat' if d < 0.5 else 'bad')
    return '<span class="chip-%s">%.*f</span>' % (cls, nd, v)


# ---------------------------------------------------------------------------
# page layout: one page per section, mechanically split at the same
# boundaries the single-page report used to scroll through.
# ---------------------------------------------------------------------------

SECTIONS = [
    ('01', 'fit', 'Fit to observations'),
    ('02', 'usage', 'Observation usage'),
    ('03', 'state', 'State space'),
    ('04', 'background', 'Background state'),
    ('05', 'dates', 'Increments across dates'),
    ('06', 'verif', 'Fit to gridded analyses'),
    ('07', 'cycling', 'Cycling behaviour'),
    ('08', 'calibration', 'Ensemble calibration'),
    ('09', 'binned', 'Binned departures'),
    ('10', 'forcing', 'Atmospheric forcing'),
]


def page_names(out):
    """Output path for each of the 8 pages. Page 1 keeps ``out`` exactly, so
    any existing link to the report's original filename still resolves."""
    stem, ext = os.path.splitext(out)
    names = [out]
    for num, slug, _title in SECTIONS[1:]:
        names.append('%s_%s_%s%s' % (stem, num, slug, ext))
    return names


def _nav(current, names_out):
    items = []
    for i, (num, _slug, title) in enumerate(SECTIONS):
        cls = ' class="on"' if i == current else ''
        items.append('<a href="%s"%s><span class="sec-n">%s</span>%s</a>'
                     % (html.escape(os.path.basename(names_out[i])), cls,
                        num, html.escape(title)))
    return '<nav class="secnav">%s</nav>' % ''.join(items)


# ---------------------------------------------------------------------------

def is_passive(cycles, obstype):
    """Monitored, not assimilated, in every experiment that has it."""
    flags = [v for d in cycles.values()
             for v in (d.get('obs', {}).get(obstype, {}).get('passive') or {}).values()]
    return bool(flags) and all(flags)


def _series(cycles, obstype, name, key, sample):
    """Metric value per cycle for one experiment, NaN where absent."""
    return np.array([P.get(cycles[c].get('obs', {}).get(obstype, {}),
                           sample, name, 'all', key)
                     for c in sorted(cycles)], dtype='f8')


def _mean(v):
    """nanmean that returns NaN instead of warning on an all-NaN slice."""
    return float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan


def present(data, name):
    """Whether an experiment actually has data at this cycle.

    After compute_cycle.py --rejoin every cycle lists the whole registry
    under 'experiments', so exp_names() is no test; what says an experiment
    was really there is a state block or an own-sample obs block of its own.
    """
    if name in (data.get('state') or {}):
        return True
    return any(name in (o.get('own') or {}) for o in data.get('obs', {}).values())


def _cycle_set(cycles, name):
    return {c for c in cycles if present(cycles[c], name)}


def latest_complete(cycles, names):
    """The most recent cycle where EVERY experiment has data, else the most
    recent cycle. A run's reference usually stops before the newest cycle
    (it is fetched from an archive), and defaulting every single-date view
    and table to the newest cycle showed an empty reference column."""
    order = sorted(cycles)
    for c in reversed(order):
        if all(present(cycles[c], n) for n in names):
            return c
    return order[-1]


# Set by build(): the cycle every date menu opens on.
DEFAULT_CYCLE = None


def build(cfg, cycles, out):
    global DEFAULT_CYCLE
    newest = sorted(cycles)[-1]
    # The union across every cached cycle, not just exp_names() on one:
    # see all_exp_names(). ``last`` is the latest COMPLETE cycle -- where
    # every experiment has data -- and is what the single-date tables and
    # the date menus open on; ``newest`` is only reported in the masthead.
    names = P.all_exp_names(cfg, cycles)
    last = DEFAULT_CYCLE = latest_complete(cycles, names)
    data = cycles[last]
    labels = {e.name: e.label for e in cfg['experiments']}
    ref = cfg.get('reference') or names[0]
    others = [n for n in names if n != ref]
    col = P.color_map(names)
    figs = cfg['figs']

    # Per-cycle families behind a date menu: everything the plot stages
    # drew for more than one cycle (build_comparison.py --hours). Each view
    # keeps its own menu so stepping through dates holds the view fixed --
    # the comparison that matters -- and a nested region/field picker gets a
    # per-cycle group name so its ids stay unique across the dates.
    def dates(stem_re):
        return rendered_cycles(figs, stem_re, cycles)

    def dated(group, base, alt, optional=False):
        """Date menu over one '<base>[_<cycle>].png' family."""
        return cycle_menu(group, dates(re.escape(base)), lambda c: img(
            fig_path(figs, base, c), '%s, %s' % (alt, PS.cycle_row_label(c)),
            optional=optional))

    def dated_hemi(group, base, alt, optional=False):
        return cycle_menu(group, dates(re.escape(base) + '_nh'),
                          lambda c: hemi_imgs(figs, base, c, '%s, %s'
                                              % (alt, PS.cycle_row_label(c)),
                                              optional=optional))

    legend = ''.join(
        '<span class="exp"><i style="background:%s"></i>%s'
        '<em>%s</em></span>'
        % (col[n], html.escape(labels[n]),
           ' &middot; reference' if n == ref else '')
        for n in names)

    # -- section 1: fit ------------------------------------------------------
    # Averaged over every cycle each experiment actually has, not read off
    # one snapshot cycle -- the earlier per-cycle version showed nothing at
    # all for an experiment whose window excludes ``last``. The common
    # sample is used wherever compute_cycle.py --rejoin actually joined it;
    # cycles it could not join (nothing to join against, or not every
    # experiment resolved the obs type) fall back to each experiment's own
    # sample rather than being left blank. This is chosen per obs type (not
    # once for the whole table): a type only one experiment assimilates must
    # not push every OTHER, fully-shared type onto its own sample too.
    types = sorted({t for d in cycles.values() for t in d.get('obs', {})})
    rows = []
    for t in types:
        sample = P.type_sample(cycles, t)
        r = _mean(_series(cycles, t, ref, 'ombg_rms', sample))
        ra = _mean(_series(cycles, t, ref, 'oman_rms', sample))
        # own_vals is keyed once per experiment so an obs type the REFERENCE
        # doesn't assimilate (its r/ra above are then NaN) still gets a row
        # -- gating on the reference alone dropped every type it lacks even
        # when another experiment had real stats for it.
        own_vals = {n: (_mean(_series(cycles, t, n, 'ombg_rms', sample)),
                        _mean(_series(cycles, t, n, 'oman_rms', sample)))
                    for n in others}
        if not any(np.isfinite(v) for v in (r, ra) + tuple(
                x for pair in own_vals.values() for x in pair)):
            continue
        cells = [P.short(t) + (' <span class="dim">passive</span>'
                               if is_passive(cycles, t) else ''),
                 fmt(r, 4), fmt(ra, 4)]
        for n in others:
            ob, oa = own_vals[n]
            cells.append(delta_cell(ob, r))
            cells.append(delta_cell(oa, ra))
        rows.append(cells)
    hdr = ['obs type', '%s O&minus;B' % ref, '%s O&minus;A' % ref]
    for n in others:
        hdr += ['%s O&minus;B' % n, '%s O&minus;A' % n]
    t1 = table(hdr, rows)
    if any(is_passive(cycles, t) for t in types):
        t1 += ('<p class="lede"><b>passive</b> marks an observation type the '
               'DA carried but gave no weight (QC flag <i>passive</i>; '
               'SMAP/SMOS salinity here). Its O&minus;B is a genuine '
               'monitor of the background against an independent '
               'measurement; its O&minus;A only shows what the other '
               'observations did to the analysis at those points, not a '
               'fit that was asked for.</p>')

    # An experiment sharing NO cached cycle with the reference is not a
    # controlled comparison -- its column above is its own separate period
    # sitting next to the reference's, not the same dates. Own-sample
    # fallback (above) still prints real numbers for it; this says plainly
    # that they are not paired.
    ref_cycles = _cycle_set(cycles, ref)
    disjoint = [n for n in others if not (_cycle_set(cycles, n) & ref_cycles)]
    overlap_warning = ''
    if disjoint:
        overlap_warning = (
            '<p class="warn"><b>%s</b> share%s no cached cycle with the '
            'reference (%s) &mdash; %s column%s above %s its own separate '
            'period next to the reference\'s, not a same-dates comparison.'
            '</p>'
            % (html.escape(' / '.join(labels[n] for n in disjoint)),
               '' if len(disjoint) > 1 else 's',
               html.escape(labels[ref]),
               'their' if len(disjoint) > 1 else 'its',
               's' if len(disjoint) > 1 else '',
               'are' if len(disjoint) > 1 else 'is'))

    # -- calibration ---------------------------------------------------------
    # The Desroziers ratios and the rank-histogram end ratio are still computed
    # and still drive scorecard.md; they are off this table because reading a
    # tuning decision off them belongs with the figures, not a wide grid.
    # Averaged over cycles like section 1, for the same reason: an ensemble
    # experiment whose window excludes ``last`` would otherwise never appear.
    ens = [n for n in names
           if any(np.any(np.isfinite(_series(cycles, t, n, 'spread_b',
                                             P.type_sample(cycles, t))))
                  for t in types)]
    # Rank histograms, spread-skill, consistency and spread-reduction figures
    # all read ensemble spread, which a var-only comparison never has -- those
    # panels are then correctly absent rather than a renamed figure gone
    # missing, so only require them when some experiment is an ensemble.
    has_ens = bool(ens)
    rows = []
    for t in types:
        sample = P.type_sample(cycles, t)
        for n in ens:
            cr = _mean(_series(cycles, t, n, 'consistency_ratio', sample))
            ss = _mean(_series(cycles, t, n, 'spread_skill', sample))
            sr = _mean(_series(cycles, t, n, 'spread_ratio', sample))
            cp = _mean(_series(cycles, t, n, 'crps', sample))
            if not any(np.isfinite(v) for v in (cr, ss, sr, cp)):
                continue
            rows.append([
                P.short(t), html.escape(labels[n]),
                target_cell(cr), target_cell(ss), fmt(sr), fmt(cp, 4)])
    t2 = table(['obs type', 'experiment', 'consistency ratio', 'spread / skill',
                '&sigma;<sub>a</sub>/&sigma;<sub>b</sub>', 'CRPS'], rows)

    # -- state ---------------------------------------------------------------
    # Averaged over every cycle each experiment has, like the obs table: a
    # single-cycle snapshot read off the newest date showed an empty column
    # for a reference that stops earlier, and one date is a poor summary of
    # a cycling run anyway. The count of cycles behind each column is in
    # the header.
    levels = cfg.get('map_levels', [0])

    def level_mean(key, n, realm, var, k):
        vals = []
        for c in cycles:
            p = P.get(cycles[c]['state'], n, realm, key, var, default=None)
            if p and k < len(p) and p[k] is not None:
                vals.append(p[k])
        return _mean(np.array(vals, dtype='f8')) if vals else np.nan, len(vals)

    rows, n_incr = [], {n: 0 for n in names}
    for realm in ('ocean', 'ice'):
        for var in cfg.get('state_vars', {}).get(realm, []):
            avail = [P.get(cycles[c]['state'], n, realm, 'incr_rms', var,
                           default=None) for c in cycles for n in names]
            avail = [p for p in avail if p]
            if not avail:
                continue
            ks = levels if max(len(p) for p in avail) > 1 else [0]
            for k in ks:
                cells = ['%s <span class="dim">%s</span>' % (var, realm),
                         # a level index means nothing on its own: level 30 is
                         # 102 m on this grid, level 10 is 21 m
                         '%d <span class="dim">%s</span>'
                         % (k, _depth_note(PS.level_depth(data, k)))]
                for n in names:
                    v, cnt = level_mean('incr_rms', n, realm, var, k)
                    n_incr[n] = max(n_incr[n], cnt)
                    cells.append(fmt(v, 5) if np.isfinite(v) else '&mdash;')
                for n in names:
                    # signed, so a value is shown with its sign; caches
                    # written before incr_mean existed show a dash
                    v, _cnt = level_mean('incr_mean', n, realm, var, k)
                    cells.append(('%+.5f' % v) if np.isfinite(v) else '&mdash;')
                for n in names:
                    v, _cnt = level_mean('spread_ratio', n, realm, var, k)
                    cells.append(fmt(v) if np.isfinite(v) else '&mdash;')
                rows.append(cells)
    t4 = table(['field', 'level']
               + ['RMS incr %s <span class="dim">%d cyc</span>' % (n, n_incr[n])
                  for n in names]
               + ['mean incr %s' % n for n in names]
               + ['&sigma;<sub>a</sub>/&sigma;<sub>b</sub> %s' % n
                  for n in names], rows)

    f_increg = cycle_menu(
        'increg', rendered_cycles(figs, 'state_increment_regions_region_global',
                                  cycles),
        lambda c: regional_widget('increg-%s' % c, cfg, figs, c,
                                  'state_increment_regions', 'RMS increment'))
    f_cons = cycle_menu('cons', dates('obs_consistency'), lambda c: img(
        fig_path(figs, 'obs_consistency', c),
        'Departure against the spread that should match it, %s'
        % PS.cycle_row_label(c), optional=not has_ens))
    drift = img(os.path.join(figs, 'cycle_background_drift.png'),
                'Background global means across cycles')

    any_own = any(P.type_sample(cycles, t) == 'own' for t in types)
    sample_note = (
        'Headline comparisons use the <b>common sample</b> &mdash; observations '
        'present and passing quality control in <em>every</em> experiment '
        '&mdash; because the configurations do not assimilate the same '
        'observations.' if not any_own else
        '<b>Some obs types above use own-sample scores.</b> No '
        'cross-experiment common sample exists for them, because an '
        'experiment does not assimilate that type or the caches were '
        'computed in separate runs, so each experiment is scored on the '
        'observations it assimilated; differences there are confounded by '
        'thinning and QC. Run <code>compute_cycle.py --rejoin</code> to '
        'rebuild a true common sample where possible.')

    seq_figs = sequence_widget('seq', cfg, figs, cycles)
    hov = img(os.path.join(figs, 'cycle_increment_hovmoller.png'),
              'RMS increment against depth and cycle', optional=True)
    # the signed mean beside the magnitude, per region (chips), from the
    # incr_mean_region block plot_timeseries.fig_increment_hovmoller_means
    # writes one file per region for
    hov_mean = regional_series_widget(
        'hovmean', cfg, figs, 'cycle_increment_hovmoller_mean_region',
        'area-weighted mean increment against depth and cycle')
    # the same signed mean as a line per cycle, on its own axis -- under the
    # RMS in cycle_increment_2d.png it is flat against zero
    mean_series = regional_series_widget(
        'incmeanser', cfg, figs, 'cycle_increment_mean_region',
        'area-weighted mean increment across cycles: surface (solid), '
        'column mean (dashed)')
    f_incmean = cycle_menu(
        'incmean', rendered_cycles(figs, 'state_increment_mean_regions_region_global',
                                   cycles),
        lambda c: regional_widget('incmean-%s' % c, cfg, figs, c,
                                  'state_increment_mean_regions', 'mean increment'))
    inc2d = img(os.path.join(figs, 'cycle_increment_2d.png'),
                '2-D field increment magnitude across cycles', optional=True)

    # Behind an obs-type menu, like sections 01/02: thirty of these stacked
    # flat put the stability block a very long scroll away.
    cycle_figs = obstype_dropdown_widget(cycles, figs, 'cycle',
                                         'headline metrics across cycles')

    ncyc = len(cycles)
    cyc_note = ('Only one cycle is cached, so these panels are dot plots. '
                'Add cycles with <code>compute_cycle.py</code> and they become '
                'time series; the paired significance test in '
                '<code>scorecard.md</code> needs at least six.'
                if ncyc < 6 else
                '%d cycles cached.' % ncyc)

    subs = dict(
        cycle=PS.cycle_row_label(last), newest=PS.cycle_row_label(newest),
        ncyc=ncyc,
        first_cycle=PS.cycle_row_label(sorted(cycles)[0]),
        nexp=len(names), ref=html.escape(ref),
        legend=legend,
        t1=t1, t2=t2, t4=t4,
        f_departures=dated('departures', 'obs_departures',
                           'Background and analysis fit to observations, '
                           'common sample'),
        f_spread=dated('spread', 'obs_spread',
                       'Consistency ratio and posterior/prior spread',
                       optional=not has_ens),
        f_rank=dated('rank', 'obs_rank_histograms',
                     'Rank histograms of the observation within the prior '
                     'ensemble', optional=not has_ens),
        f_ss=dated('ss', 'obs_spread_skill',
                   'Spread-skill relationship, binned by ensemble spread',
                   optional=not has_ens),
        f_prof=cycle_menu('profiles', dates('obs_profiles_region_global'),
                          lambda c: profiles_widget(cycles[c], cfg, figs, c)),
        # only written when `ocean_basin_mask:` is configured
        f_regions=img(os.path.join(figs, 'ocean_regions.png'),
                      'Ocean basins and named regions used for every '
                      'regional breakdown in this report', optional=True),
        f_counts=counts_widget(cycles, cfg, figs),
        f_incr=dated('incr', 'state_increment_profiles',
                     'RMS analysis increment against depth'),
        f_sprprof=dated('sprprof', 'state_spread_profiles',
                        'Prior and posterior ensemble spread against depth',
                        optional=not has_ens),
        f_sprreg=dated('sprreg', 'state_spread_regions',
                       'Ensemble spread and spread reduction by region',
                       optional=not has_ens),
        f_map_ocn=dated('mapocn', 'state_maps_ocean_increment',
                        'Ocean analysis increment maps'),
        f_map_spr=dated('mapspr', 'state_maps_ocean_spread_reduction',
                        'Ocean ensemble spread reduction maps',
                        optional=not has_ens),
        f_map_ice=dated_hemi('mapice', 'state_maps_ice_increment',
                             'Sea-ice analysis increment maps'),
        f_ice=dated_hemi('icespr', 'state_maps_ice_spread_reduction',
                         'Sea-ice ensemble spread reduction maps',
                         optional=not has_ens),
        # only written when the post-inflation variance file is present, so
        # optional -- but reported as missing if it is, like every other figure
        f_map_inf=dated('mapinf', 'state_maps_ocean_inflation',
                        'Ocean applied-inflation maps', optional=True),
        f_ice_inf=dated_hemi('iceinf', 'state_maps_ice_inflation',
                             'Sea-ice applied-inflation maps', optional=True),
        f_corr=dated('corr', 'state_correlation_lengths',
                     'Horizontal correlation length of the increment by '
                     'region', optional=True),
        f_stab=img(os.path.join(figs, 'cycle_stability.png'),
                   'Cycling stability of the background fit', optional=True),
        # empty when `verification:` names no products, or none resolved to
        # a real file for any cached cycle
        f_verif=verif_widget(cycles, cfg, names, figs),
        f_obsfit=obsfit_widget(cycles, cfg, figs),
        f_binned=binned_widget(cycles, cfg, figs),
        f_atmos_series=regional_series_widget(
            'atmos', cfg, figs, 'atmos_region',
            'atmospheric forcing over the ocean against cycle'),
        f_atmos_maps=dated('atmosmaps', 'atmos_maps',
                           'Atmospheric forcing over the ocean, reference '
                           'experiment', optional=True),
        f_atmos_diff=dated('atmosdiff', 'atmos_diff_maps',
                           'Atmospheric forcing minus the reference, per '
                           'experiment', optional=True),
        f_stability=stability_widget(cfg, figs),
        bin_deg='%g' % float((cfg.get('obs_bins') or {}).get('deg', 1.0)),
        # fields then their difference, per product: the fields say whether
        # the model reproduces the product, the difference is where the error
        # is actually legible. One date menu per product.
        f_verif_maps=figure_menu('verifmaps', 'gridded product', [
            ('%s: fields, then model minus product' % p.upper(),
             cycle_menu('verifmaps-%s' % p,
                        dates(r'verif_maps_%s' % p
                              + ('_nh' if LV.product_realm(p) == 'ice' else '')),
                        lambda c, p=p: _verif_pair(figs, p, c)))
            for p in LV.PRODUCTS]),
        f_fronts=frontal_widget(cfg, figs),
        sample_note=sample_note, overlap_warning=overlap_warning,
        **{'m_%s' % k: methods(k) for k in METHODS},
        sample_word='common' if not any_own else 'common (partly own)',
        cycle_figs=cycle_figs, cyc_note=cyc_note,
        seq_figs=seq_figs, hov=hov, hov_mean=hov_mean,
        mean_series=mean_series, inc2d=inc2d,
        f_increg=f_increg, f_incmean=f_incmean, f_cons=f_cons,
        f_incrsec=cycle_menu(
            'incrsec', dates(r'state_sections_incr_.*'),
            lambda c: sections_widget('incrsec-%s' % c, cfg, figs, c, 'incr')),
        f_bkgprof=cycle_menu('bkgprof', dates('bkg_profiles'), lambda c: img(
            fig_path(figs, 'bkg_profiles', c),
            'Background mean temperature and salinity against depth, %s'
            % PS.cycle_row_label(c))),
        f_bkgreg=cycle_menu(
            'bkgreg', dates('bkg_profiles_regions_region_global'),
            lambda c: regional_widget('bkgreg-%s' % c, cfg, figs, c,
                                      'bkg_profiles_regions',
                                      'background mean')),
        f_bkgocn=cycle_menu('bkgocn', dates('bkg_maps_ocean'), lambda c: img(
            fig_path(figs, 'bkg_maps_ocean', c),
            'Ocean background state, %s' % PS.cycle_row_label(c))),
        f_bkgsec=cycle_menu(
            'bkgsec', dates(r'state_sections_bkg_.*'),
            lambda c: sections_widget('bkgsec-%s' % c, cfg, figs, c, 'bkg')),
        f_bkgice=cycle_menu('bkgice', dates('bkg_maps_ice_nh'), lambda c: hemi_imgs(
            figs, 'bkg_maps_ice', c,
            'Sea-ice background state, %s' % PS.cycle_row_label(c))),
        f_woaprof=cycle_menu('woaprof', dates('woa_bias_profiles'), lambda c: img(
            fig_path(figs, 'woa_bias_profiles', c),
            'Background and the WOA23 climatology against depth, and their '
            'difference, %s' % PS.cycle_row_label(c), optional=True)),
        f_woareg=cycle_menu(
            'woareg', dates('woa_bias_regions_region_global'),
            lambda c: regional_widget('woareg-%s' % c, cfg, figs, c,
                                      'woa_bias_regions',
                                      'background $-$ WOA23')),
        f_woabias=cycle_menu('woabias', dates('woa_bias_maps_ocean'), lambda c: img(
            fig_path(figs, 'woa_bias_maps_ocean', c),
            'Background minus the WOA23 climatology, %s'
            % PS.cycle_row_label(c), optional=True)),
        f_woasec=cycle_menu(
            'woasec', dates(r'state_sections_woa_bias_.*'),
            lambda c: sections_widget('woasec-%s' % c, cfg, figs, c,
                                      'woa_bias')),
        drift=drift)

    # Group the tall blocks behind dropdowns. Stacked flat, sections 03 and
    # 04 are a dozen full-width map grids between them -- minutes of
    # scrolling to reach the last one, and no way to put two of them
    # side by side in the eye. One menu per family, one panel at a time.
    def note(html_note, block):
        """Prefix a panel's own explanation, but only if the panel exists."""
        return (html_note + block) if block else ''

    incr_sec_note = (
        '<p class="lede"><b>Vertical sections</b> cut the increment along the '
        'transects named by <code>sections:</code> &mdash; one row per line, '
        'one column per experiment, with depth taken from the background '
        'layer thickness so the sea floor is the model\'s own. A map at a few '
        'levels cannot say how deep an update reaches, or whether it follows '
        'the thermocline rather than cutting across it; a transect can. Note '
        'that a zonal section is a single grid <i>row</i>, a true latitude '
        'circle only as far as about 64&deg;N &mdash; north of that the rows '
        'bend around the two northern poles, and any such panel is '
        'marked.</p>')
    bkg_sec_note = (
        '<p class="lede">The same transects as section 03, against the '
        'background state itself rather than the update &mdash; the '
        'stratification the increments are working on, and where a drifting '
        'thermocline or a collapsing halocline shows as structure rather than '
        'as a shifted global mean.</p>')

    subs['m_state_prof'] = figure_menu('stateprof', 'Profile view', [
        ('Increment, global', subs['f_incr']),
        ('Increment by region', subs['f_increg']),
        ('Mean increment by region', subs['f_incmean']),
        ('Ensemble spread, global', subs['f_sprprof']),
        ('Ensemble spread by region', subs['f_sprreg'])])
    subs['m_state_maps'] = figure_menu('statemaps', 'Map view', [
        ('Ocean increment', subs['f_map_ocn']),
        ('Ocean increment sections', note(incr_sec_note, subs['f_incrsec'])),
        ('Increment correlation lengths', subs['f_corr']),
        ('Ocean spread reduction', subs['f_map_spr']),
        ('Ocean applied inflation', subs['f_map_inf']),
        ('Sea-ice increment', subs['f_map_ice']),
        ('Sea-ice spread reduction', subs['f_ice']),
        ('Sea-ice applied inflation', subs['f_ice_inf'])])
    woa_note = (
        '<p class="lede"><b>WOA23 is a climatology, not an analysis.</b> It is '
        'the 1955&ndash;2022 decadal mean for this cycle&rsquo;s day of year, '
        'interpolated between the two mid-month fields bracketing it, monthly '
        'above 1500&nbsp;m and annual below. The difference therefore carries '
        'the ocean&rsquo;s real interannual and eddy anomaly as well as any '
        'model error, and a non-zero value is not by itself a fault. Read it '
        'for <i>structure</i> &mdash; a thermocline at the wrong depth, a '
        'collapsing halocline, a basin-wide offset &mdash; not as a score. '
        'WOA distributes in-situ temperature and the model writes potential '
        'temperature, so the climatology is converted (EOS-80) before '
        'differencing; without that the deep bias would be dominated by the '
        '0.35&nbsp;&deg;C adiabatic offset at 4000&nbsp;m rather than by the '
        'ocean.</p>')
    subs['m_woa'] = figure_menu('woaview', 'WOA23 view', [
        ('Departure against depth', note(woa_note, subs['f_woaprof'])),
        ('Departure by region', subs['f_woareg']),
        ('Departure maps', subs['f_woabias']),
        ('Departure vertical sections', subs['f_woasec'])])
    subs['m_bkg'] = figure_menu('bkgview', 'Background view', [
        ('Mean state against depth', subs['f_bkgprof']),
        ('Mean state by region', subs['f_bkgreg']),
        ('Ocean maps', subs['f_bkgocn']),
        ('Ocean sections', note(bkg_sec_note, subs['f_bkgsec'])),
        ('Sea-ice maps', subs['f_bkgice']),
        ('Global-mean drift across cycles', subs['drift'])])

    names_out = page_names(out)
    tail = Template(TAIL).substitute(subs)
    pages = []
    for i, (_num, _slug, title) in enumerate(SECTIONS):
        page_subs = dict(subs, nav=_nav(i, names_out), page_title=title)
        head = Template(HEAD).substitute(page_subs)
        body = Template(SECTION_TEMPLATES[i]).substitute(subs)
        pages.append((names_out[i], head + body + tail))
    return pages


STYLE = r"""<title>Marine DA Verification &mdash; ${page_title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
:root{
  color-scheme: light;
  --ground:#f1f4f7; --surface:#ffffff; --surface-2:#f7f9fb;
  --ink:#0f1720; --ink-2:#4e5a67; --ink-3:#7d8894;
  --rule:#dfe5ec; --rule-2:#eef2f6;
  --accent:#2a78d6; --accent-soft:#e8f0fb;
  --ok:#177a52; --ok-bg:#e3f3ec;
  --warn:#9c6206; --warn-bg:#faeed8;
  --alert:#b23a2f; --alert-bg:#fbe7e4;
  --shadow:0 1px 2px rgba(15,23,32,.05), 0 8px 24px -16px rgba(15,23,32,.28);
  --sans:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
  --serif:"IBM Plex Serif",ui-serif,Georgia,Cambria,"Times New Roman",serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --ground:#0d1116; --surface:#151b22; --surface-2:#1b222a;
    --ink:#e7ecf2; --ink-2:#a3b0bd; --ink-3:#78848f;
    --rule:#28313b; --rule-2:#1f2730;
    --accent:#5b9ae0; --accent-soft:#17263a;
    --ok:#3aa47a; --ok-bg:#122a22;
    --warn:#d09a3c; --warn-bg:#2e2416;
    --alert:#e0796c; --alert-bg:#331d1a;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#0d1116; --surface:#151b22; --surface-2:#1b222a;
  --ink:#e7ecf2; --ink-2:#a3b0bd; --ink-3:#78848f;
  --rule:#28313b; --rule-2:#1f2730;
  --accent:#5b9ae0; --accent-soft:#17263a;
  --ok:#3aa47a; --ok-bg:#122a22;
  --warn:#d09a3c; --warn-bg:#2e2416;
  --alert:#e0796c; --alert-bg:#331d1a;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1140px; margin:0 auto; padding:clamp(28px,5vw,64px) clamp(16px,4vw,32px) 96px;
  display:flex; flex-direction:column; gap:44px}

/* ---- masthead ---- */
.mast{display:flex; flex-direction:column; gap:18px;
  border-bottom:2px solid var(--ink); padding-bottom:22px}
.eyebrow{font-family:var(--mono); font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-3)}
h1{font-family:var(--serif); font-weight:600; font-size:clamp(30px,4.4vw,44px);
  line-height:1.12; margin:0; text-wrap:balance; letter-spacing:-.015em}
.sub{color:var(--ink-2); max-width:66ch; margin:0}
.meta{display:flex; flex-wrap:wrap; gap:10px 26px; font-family:var(--mono);
  font-size:12.5px; color:var(--ink-2)}
.meta b{color:var(--ink); font-weight:500}
.exps{display:flex; flex-wrap:wrap; gap:8px 20px; align-items:center}
.exp{display:inline-flex; align-items:center; gap:8px; font-size:13.5px;
  font-weight:500}
.exp i{width:11px; height:11px; border-radius:3px; display:inline-block;
  flex:none}
.exp em{font-style:normal; font-family:var(--mono); font-size:11.5px;
  color:var(--ink-3)}

/* ---- section nav ---- */
.secnav{display:flex; flex-wrap:wrap; gap:10px}
.secnav a{display:inline-flex; align-items:baseline; gap:9px;
  padding:9px 16px; border-radius:8px; text-decoration:none;
  border:1.5px solid var(--accent); background:var(--surface);
  color:var(--accent); font-size:13.5px; font-weight:600;
  box-shadow:var(--shadow)}
.secnav a .sec-n{font-family:var(--mono); font-size:11px; color:var(--ink-3);
  font-weight:600}
.secnav a:hover{background:var(--accent-soft)}
/* the page being read: filled, and not a link to anywhere */
.secnav a.on{background:var(--accent); color:#fff; box-shadow:none;
  cursor:default}
.secnav a.on .sec-n{color:#fff}

/* ---- sections ---- */
section{display:flex; flex-direction:column; gap:16px}
.sec-head{display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
  border-bottom:1px solid var(--rule); padding-bottom:10px}
.sec-n{font-family:var(--mono); font-size:12px; color:var(--accent);
  font-weight:600}
h2{font-family:var(--serif); font-weight:600; font-size:23px; margin:0;
  letter-spacing:-.01em}
h3{font-family:var(--sans); font-weight:600; font-size:15px; margin:14px 0 0;
  color:var(--ink)}
.lede{margin:0; color:var(--ink-2); max-width:74ch}
details.methods{margin:4px 0 6px; max-width:78ch; font-size:13.5px; color:var(--ink-2)}
details.methods summary{cursor:pointer; font-weight:600; color:var(--ink-3); font-size:12.5px;
  letter-spacing:.04em; text-transform:uppercase; list-style:none}
details.methods summary::-webkit-details-marker{display:none}
details.methods summary::before{content:"\25B8  "; font-size:11px}
details.methods[open] summary::before{content:"\25BE  "}
details.methods p, details.methods ul{margin:8px 0; line-height:1.55}
details.methods ul{padding-left:20px}

/* ---- figures ---- */
.fig{margin:0; background:var(--surface); border:1px solid var(--rule);
  border-radius:10px; padding:12px; box-shadow:var(--shadow);
  display:flex; flex-direction:column; gap:8px}
.fig-scroll{overflow-x:auto}
.fig img{display:block; max-width:100%; height:auto; margin:0 auto;
  border-radius:4px}
.fig-zoom{display:block; cursor:zoom-in}
/* lightbox: one <dialog> per page, filled by lvZoom() */
#lv-zoom{padding:0; border:0; background:transparent; max-width:96vw;
  max-height:96vh; overflow:auto; cursor:zoom-out}
#lv-zoom::backdrop{background:rgba(8,12,18,.82)}
#lv-zoom img{display:block; max-width:96vw; height:auto; cursor:zoom-in;
  background:#fff; border-radius:6px}
#lv-zoom.full img{max-width:none; cursor:zoom-out}
figcaption{font-size:12px; color:var(--ink-3); font-family:var(--mono);
  line-height:1.45}

/* ---- tables ---- */
.tbl-scroll{overflow-x:auto; background:var(--surface);
  border:1px solid var(--rule); border-radius:10px; box-shadow:var(--shadow)}
table{border-collapse:collapse; width:100%; font-size:13px;
  font-variant-numeric:tabular-nums}
th,td{padding:8px 12px; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--rule-2)}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
thead th{position:sticky; top:0; background:var(--surface-2); color:var(--ink-2);
  font-family:var(--mono); font-size:11px; font-weight:500; letter-spacing:.03em;
  border-bottom:1px solid var(--rule); z-index:1}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--surface-2)}
td{font-family:var(--mono)}
td:first-child,td:nth-child(2){font-family:var(--sans)}
.dim{color:var(--ink-3); font-size:11px}
.delta{font-size:11px; padding-left:4px}
.delta.good{color:var(--ok)}
.delta.bad, .bad{color:var(--alert); font-weight:600}
.delta.flat{color:var(--ink-3)}
.chip-good{color:var(--ok); font-weight:500}
.chip-flat{color:var(--ink-2)}
.chip-bad{color:var(--alert); font-weight:500}

/* ---- picker widget (fit to gridded analyses, fit to observations) ---- */
.picker-widget{display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start}
.picker-tab{position:absolute; opacity:0; width:1px; height:1px}
.picker-chip{display:inline-flex; align-items:center; padding:6px 14px;
  border-radius:999px; border:1.5px solid var(--rule); background:var(--surface);
  color:var(--ink-2); font-size:13px; font-weight:500; cursor:pointer;
  user-select:none}
.picker-chip:hover{border-color:var(--accent); color:var(--ink)}
/* outline, not border/background -- a basin chip's colour is set inline
   (to match ocean_regions.png) and inline style always wins over these
   class rules, so the "selected" cue has to be a property inline never
   touches */
.picker-tab:checked+.picker-chip{outline:2px solid var(--ink); font-weight:700}
.picker-tab:focus-visible+.picker-chip{outline:2px solid var(--accent);
  outline-offset:2px}
.picker-panels{flex:1 0 100%; margin-top:4px}
.picker-panel{display:none}
/* collapsible chooser (many-item pickers, e.g. obs type) -- wraps the same
   .picker-tab/.picker-chip pairs above, just tucked behind a <summary> */
.picker-chooser{flex:1 0 100%}
.picker-chooser summary{cursor:pointer; list-style:none; font-size:13px;
  font-weight:500; color:var(--ink-2); padding:4px 2px; user-select:none}
.picker-chooser summary::-webkit-details-marker{display:none}
.picker-chooser summary::before{content:'\25b8'; display:inline-block;
  width:1em; transition:transform .12s ease}
.picker-chooser[open] summary::before{transform:rotate(90deg)}
.picker-chooser summary:hover{color:var(--ink)}
.picker-chips{display:flex; flex-wrap:wrap; gap:8px; padding-top:8px}
/* dropdown chooser (picker_widget(dropdown=...)) -- JS (see lvPickerShow)
   swaps .picker-panel visibility on change; .default is this mode's only
   static starting state, since there is no :checked to key a CSS rule off */
.picker-row{flex:1 0 100%; display:flex; align-items:center; gap:10px}
.picker-label{font-family:var(--mono); font-size:11.5px; letter-spacing:.06em;
  text-transform:uppercase; color:var(--ink-3); white-space:nowrap;
  min-width:9ch}
.picker-select{flex:1 1 auto; font:inherit; font-size:13px; padding:7px 12px;
  border-radius:8px; border:1.5px solid var(--rule); background:var(--surface);
  color:var(--ink); cursor:pointer; max-width:100%; min-width:0}
.picker-select:hover{border-color:var(--accent)}
.picker-select:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
.picker-panel.default{display:block}

/* ---- warning ---- */
p.warn{margin:0; padding:12px 16px; border-radius:8px; font-weight:600;
  color:var(--alert); background:var(--alert-bg); border:1px solid var(--alert)}

/* ---- callout ---- */
.callout{background:var(--surface); border:1px solid var(--rule);
  border-left:3px solid var(--accent); border-radius:0 10px 10px 0;
  padding:16px 20px; display:flex; flex-direction:column; gap:10px}
.callout h3{margin:0}
.callout ul{margin:0; padding-left:20px; display:flex; flex-direction:column;
  gap:8px; color:var(--ink-2); font-size:14px}
.callout code, .lede code{font-family:var(--mono); font-size:12.5px;
  background:var(--accent-soft); color:var(--accent); padding:1px 5px;
  border-radius:4px}
footer{color:var(--ink-3); font-size:12.5px; font-family:var(--mono);
  border-top:1px solid var(--rule); padding-top:18px; line-height:1.7}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{animation:none!important;
  transition:none!important}}
</style>
"""

HEAD = STYLE + r"""
<dialog id="lv-zoom"><img alt=""></dialog>
<script>
function lvZoom(a){
  var d=document.getElementById('lv-zoom');
  if(!d||!d.showModal){return true}          /* no <dialog>: follow the link */
  var im=d.querySelector('img');
  im.src=a.href; im.alt=a.querySelector('img').alt; d.classList.remove('full');
  d.showModal(); d.scrollTop=0; d.scrollLeft=0;
  return false;
}
document.addEventListener('DOMContentLoaded',function(){
  var d=document.getElementById('lv-zoom'); if(!d){return}
  var im=d.querySelector('img');
  im.addEventListener('click',function(e){d.classList.toggle('full');
    e.stopPropagation()});                   /* fit <-> native pixels */
  d.addEventListener('click',function(){d.close()});   /* outside the image */
  d.addEventListener('close',function(){im.src=''});
});
</script>
<div class="wrap">

<header class="mast">
  <span class="eyebrow">Marine data assimilation &middot; verification</span>
  <h1>Marine analysis verification</h1>
  <p class="sub">Every number below is computed from the analysis output of
  ${nexp} experiments, scored on the ${sample_word} sample &mdash; see
  section 01 for what that means.</p>
  <div class="meta">
    <span>latest complete cycle <b>${cycle}</b></span>
    <span>newest cached <b>${newest}</b></span>
    <span>cached cycles <b>${ncyc}</b></span>
    <span>reference <b>${ref}</b></span>
  </div>
  <div class="exps">${legend}</div>
</header>

${nav}
"""

# -- "How this is computed" -------------------------------------------------
# One collapsed block per section, under its lede: the definitions behind the
# numbers and figures, written from the code that produces them, so a reader
# never has to guess what a panel is. Plain HTML; $-free so it can go through
# Template.substitute as a value.
METHODS = {
    '01': """
<p><b>Departures.</b> O&minus;B and O&minus;A are observation minus the
background / analysis equivalent from the DA's own diagnostic files, with QC
applied: only observations the DA accepted (EffectiveQC = 0, or <i>passive</i>
for monitored types) are scored. <b>RMS</b> is the root mean square of the
departures and <b>bias</b> their signed mean, both unweighted over
observations. <b>Common sample</b> means the observations present and passing
QC in <em>every</em> experiment, joined by location and time per obs type; a
type that could not be joined (assimilated by one experiment only, or caches
computed separately) is scored on each experiment's own sample and the figure
title says so.</p>
<p><b>Averages over cycles</b> in the table are plain means of the per-cycle
values over the cycles each experiment has cached. <b>Regions</b> in the
time-series picker are the basins of <code>ocean_basin_mask</code> and the
<code>regions:</code> boxes from the config, selected by observation
latitude/longitude; ice types are split NH/SH. <b>Profile</b> types (Argo,
gliders, moorings) are also binned in depth (<code>depth_bins:</code>), and
the profile figure draws RMS(O&minus;B) per bin against
&radic;(&sigma;<sub>b</sub><sup>2</sup>&nbsp;+&nbsp;R<sup>2</sup>), where
&sigma;<sub>b</sub> is the prior ensemble spread in observation space (zero
for a deterministic system) and R the assigned observation error.</p>
""",
    '02': """
<p><b>Counts</b> are observations per cycle and obs type with EffectiveQC = 0
in that experiment's own diagnostic file (<i>n_pass_own</i>), after the DA's
thinning and quality control; the grey line is the size of the common sample
(<i>n_common_pass</i>). Nothing is area-weighted here.</p>
""",
    '03': """
<p><b>Increment</b> is the analysis minus the background as the DA wrote it
(the JEDI increment file; for MOM6 the same fields go into the IAU over the
next 6 h). Every RMS and mean of a field is <b>area-weighted</b> over wet
cells of the model grid (cell area from the gridspec), per level; the
shaded band on a profile is the area-weighted standard deviation across the
cells of the region at that level. <b>Level</b> indices are mapped to a
nominal depth from the reference experiment's background layer thickness at
that cycle. <b>Regions</b> are <code>ocean_basin_mask</code> basins and the
<code>regions:</code> boxes, as masks on the model grid. Velocity
<b>u, v</b> are the eastward / northward components at the tracer point:
face values averaged to the centre (MOM6 history) and rotated with the
grid's cos_rot / sin_rot, which matters north of the tripolar seam
(~65&deg;N); <b>speed</b> is |(u, v)|. <b>Sea-ice thickness</b> is
CICE's grid-cell-mean thickness divided by concentration
(<code>hi_div_aice_h</code>), the quantity the analysis works on.
<b>Correlation length</b> of an increment is the 1/e distance of its
isotropic spatial autocorrelation inside each <code>corr_regions:</code>
box. Ensemble
<b>spread</b> panels read the LETKF's prior / posterior variance files; the
<b>&sigma;<sub>a</sub>/&sigma;<sub>b</sub></b> column is the RMS ratio of
posterior to prior spread.</p>
""",
    '04': """
<p><b>Background</b> is the model history valid at the analysis time (the
6-h forecast from the previous analysis, f006), read as stored; ocean
fields at the <code>map_levels:</code> listed, ice fields per hemisphere.
Regional profiles are area-weighted means per level with the area-weighted
spatial standard deviation as the band. <b>MLD</b> is the model's own
mixed-layer diagnostic and exists only where the history carries it.
The <b>drift</b> panels are the area-weighted global mean of each field
against cycle: a mean that trends is a system-wide bias building up, not
a local feature.</p>
""",
    '05': """
<p>The depth&ndash;cycle panels are the per-level, area-weighted <b>RMS</b>
of the ocean increment (top) and its <b>signed mean</b> (below, per region)
at every cached cycle; the colour of the mean panel is symmetric about
zero. The mean-increment series draw that mean per cycle on its own axis,
per region: the surface level (solid) and the column mean weighted by the
nominal layer thickness down to <code>depth_max:</code> (dashed; the full
column when unset). The 2-D series carry the same two numbers for fields
without a vertical axis (SSH, ice): RMS solid, mean dashed. The map sequence is the
increment at one level per cycle on one colour scale per field (the 99th
percentile of |increment| over the cycles shown), so the pattern can be
followed from date to date without the scale moving.</p>
""",
    '06': """
<p>Each product is a daily L4 field on a regular lat/lon grid, placed on
the model grid by index arithmetic (no interpolation), scored against the
background and against background&nbsp;+&nbsp;increment at every cycle
where both are valid. <b>RMS</b> and <b>bias</b> are area-weighted over
the common valid cells, globally and per region. <b>ADT</b> has the mean
of each field over the valid cells removed first (the product's reference
surface is a mean dynamic topography, the model's its own geoid), so its
bias is zero by construction. <b>OSTIA</b> is a foundation temperature
valid at 12Z, interpolated linearly in time to each cycle, so the 06Z and
18Z scores carry a few tenths of a degree of the model's diurnal cycle in
the tropics. OSTIA uses the AVHRR/VIIRS radiances the DA also assimilates,
so the SST score is a consistency check rather than independent validation.
<b>WOA23</b> comparisons interpolate the climatology to the model levels
and subtract; they are a full-depth but climatological reference. The
<b>frontal</b> diagnostic maps geostrophic-current speed above one fixed
threshold per region, in the Copernicus ADT and in each experiment, so
placement and strength are read on the same footing; the one-dimensional
axis offset in km is reported only for regions marked
<code>axis_diagnostic</code>, and front position converges over weeks of
cycling, so read structure as real and position as provisional.</p>
""",
    '07': """
<p><b>Per-obs-type series</b> are the section-01 metrics (RMS and bias of
O&minus;B / O&minus;A, spread, consistency ratio, Desroziers ratio) at every
cycle on the common sample, with no smoothing. The <b>SSH cycling</b> block
works on the ocean surface height of the model history and the increment,
area-weighted over wet cells:</p>
<ul>
<li><b>6-h tendency</b> = background(t) &minus; [background(t&minus;6h)
+ increment(t&minus;6h)]: the change the model made on its own between two
analyses, with the analysis update taken out. Its <b>low-pass</b> is the
mean over 5&deg;&times;5&deg; blocks (20&times;20 cells at 1/4&deg;).</li>
<li><b>lag-6 h correlation</b> is the spatial correlation between the
low-pass tendency at t and at t&minus;6h over the region: positive when the
wind-driven adjustment continues from one cycle to the next, negative when
the large-scale pattern reverses every cycle (a barotropic adjustment
ringing).</li>
<li><b>increment RMS</b> of SSH per cycle; at cycles with no altimetry it is
the steric response to the T/S increments alone.</li>
<li><b>u/v deep / surface ratio</b>: RMS of the velocity increment on four
levels below 1000 m divided by the RMS on four levels above 50 m, from the
increment MOM6 ingests. A geostrophic increment decays with depth, so this
is well below 1; above 1 the analysis is handing the model a barotropic
transport.</li>
<li><b>trend table</b>: a least-squares line through each series over the
cached cycles, reported as percent of the series mean per day with the
slope's t statistic; red when |t| &gt; 3 and the sign is the adverse one
(growth for the RMS metrics and the u/v ratio, a fall for the lag
correlation).</li>
</ul>
<p>The maps are the latest cycle's background, increment, tendency and
low-pass tendency, and the low-pass tendency of the last four cycles
(24 h) side by side, all on &plusmn;0.1 m except the background.</p>
""",
    '08': """
<p>All on the common sample, per obs type and cycle. <b>Consistency
ratio</b> = (RMS(&sigma;<sub>b</sub>)<sup>2</sup> + RMS(R)<sup>2</sup>) /
RMS(O&minus;B)<sup>2</sup>: the variance the system claims against the
departure variance actually observed, 1 when calibrated, below 1
under-dispersive, above 1 over-dispersive. <b>Spread / skill</b> =
RMS(&sigma;<sub>b</sub>) / RMS(O&minus;B), also targeting 1 once R is
small. <b>Desroziers</b>:
&radic;mean(O&minus;B &middot; O&minus;A) estimates R and
&radic;mean((O&minus;B)(B&minus;A)) estimates HBH<sup>T</sup>; the ratios
to the assigned R and to the ensemble spread say whether each was right.
<b>Rank histograms</b> place each observation among the sorted ensemble
members; flat is calibrated, U-shaped under-dispersive, the reported end
ratio is the mean of the two end bins over the flat expectation.
<b>CRPS</b> is the continuous ranked probability score of the ensemble
against the observation, lower is better.</p>
""",
    '09': """
<p>Observations on the common sample are binned on a regular
latitude&ndash;longitude grid (profile types on depth &times; latitude);
each bin reports the count, the mean and RMS of O&minus;B and O&minus;A,
the RMS assigned observation error and the RMS effective error (after QC
and inflation), and RMS(O&minus;B) divided by each. <b>All cycles</b> pools
every cached cycle before binning. The <b>regression</b> view is the 2-D
density of observation against background and against analysis at the
observation points, with the least-squares line and its slope and
correlation.</p>
""",
    '10': """
<p>From the coupled atmosphere's surface history at the same f006 the ocean
background is, on the atmosphere's Gaussian grid, placed on the ocean grid
by nearest cell and restricted to ocean points. <b>wind10</b> =
|(ugrd10m, vgrd10m)|; <b>tau</b> = |(uflx_ave, vflx_ave)|; <b>qnet</b> =
dswrf &minus; uswrf + dlwrf &minus; ulwrf &minus; lhtfl &minus; shtfl
(positive into the ocean); <b>prate</b> in mm/day; <b>t2m</b> in &deg;C. The
<i>_ave</i> fluxes are the model's means over the 6 h ending at the analysis
time. Time series are area-weighted means per region; differences are
experiment minus reference at the same cycle, and the maps of difference
are on fixed &plusmn; scales per field (wind 4 m/s, stress 0.1 N/m&sup2;,
heat flux 100 W/m&sup2;, precipitation 10 mm/day, temperature 3 &deg;C,
overridable under <code>map_limits.atmos_diff</code>).</p>
""",
}


def methods(section):
    """The section's collapsed 'How this is computed' block."""
    body = METHODS.get(section)
    if not body:
        return ''
    return ('<details class="methods"><summary>How this is computed</summary>'
            '%s</details>' % body)


SEC_FIT = r"""
<section>
  <div class="sec-head"><span class="sec-n">01</span>
    <h2>Fit to observations</h2></div>
  ${m_01}
  <p class="lede">How close each background and analysis lands to the
  observations it was scored against, averaged over every cycle each
  experiment has cached. Lower is better; percentages are the change against
  <code>${ref}</code>. In the profile figure each depth bin carries the whole
  error budget on one axis:
  RMS(O&minus;B) against
  &radic;(&sigma;<sub>b</sub><sup>2</sup>&nbsp;+&nbsp;R<sup>2</sup>), the value
  it should equal when the ensemble and the assigned observation error are
  consistent, with those two contributions drawn behind it.</p>
  <p class="lede">${sample_note}</p>
  ${f_regions}
  ${overlap_warning}
  ${t1}
  ${f_departures}
  ${f_obsfit}
  ${f_prof}
</section>
"""

SEC_USAGE = r"""
<section>
  <div class="sec-head"><span class="sec-n">02</span>
    <h2>Observation usage</h2></div>
  ${m_02}
  <p class="lede">How many observations each experiment actually assimilated,
  cycle by cycle, with the size of the common sample alongside. A count that
  steps or collapses mid-run is usually the first sign of a problem upstream of
  the analysis. The per-code <code>EffectiveQC</code> breakdown is in
  <code>scorecard.md</code>.</p>
  ${f_counts}
</section>
"""

SEC_STATE = r"""
<section>
  <div class="sec-head"><span class="sec-n">03</span>
    <h2>State space</h2></div>
  ${m_03}
  <p class="lede">Where each system puts its update, and how far the analysis
  cut the ensemble spread. <b>RMS increment</b> at a level is the
  area-weighted root-mean-square of (analysis &minus; background) over the wet
  cells of that level, in the field's own units &mdash; a magnitude, so
  increments of opposite sign add rather than cancel, and it is never negative.
  It says how hard the analysis pushed, not in which direction; the maps below
  carry the sign. The weights are the model grid cell areas under the land
  mask. Both the increment magnitude and the ensemble spread are shown globally
  and then broken out by region, because neither localization nor inflation
  acts uniformly in latitude.</p>
  <p class="lede">The ensemble spread is reported at three stages:
  <b>&sigma;<sub>b</sub></b> the background, <b>&sigma;<sub>a</sub></b> the
  analysis before inflation, and <b>&sigma;<sub>an</sub></b> the analysis after
  it. <b>Spread reduction</b> 1&minus;&sigma;<sub>a</sub>/&sigma;<sub>b</sub> is
  what the update removed; <b>applied inflation</b>
  &sigma;<sub>an</sub>/&sigma;<sub>a</sub> is what RTPS put back; and
  &sigma;<sub>an</sub>/&sigma;<sub>b</sub> is what actually propagates to the
  next cycle. Under RTPS the inflation factor is exactly 1 wherever no
  observation reached the column, so its map doubles as a picture of
  observational reach. It is drawn on a log&#8322; scale because the factor
  spans 1 to roughly 40 with a median near 1.5.</p>
  ${t4}
  ${m_state_prof}
  ${m_state_maps}
</section>
"""

SEC_BACKGROUND = r"""
<section>
  <div class="sec-head"><span class="sec-n">04</span>
    <h2>Background state</h2></div>
  ${m_04}
  <p class="lede">What the increments are correcting, and whether the mean
  state is holding still. Departures and increments can both look healthy while
  the model climate walks away &mdash; the global means below are the only view
  here that catches that, and they need several cycles to be worth reading.
  Depth comes from the background layer thickness, so it is the model's own
  geometry rather than a nominal axis.</p>
  <p class="lede">Where a <b>WOA23</b> column leads the maps and sections, or a
  dotted <b>WOA23</b> line sits on a profile, that is the 1955&ndash;2022
  climatology for this day of year on the same scale &mdash; a reference for
  the shape of the state, not another experiment. Section 06 carries the
  departure from it, and the caveats that go with a climatology.</p>
  <p class="lede">Each view carries a <b>cycle menu</b> over every date it was
  drawn for &mdash; the 00z cycles plus the latest by default
  (<code>build_comparison.py --hours</code>) &mdash; opening on the most
  recent. The colour scales are fixed by <code>map_limits:</code>, so the
  state is comparable from one date to the next.</p>
  ${m_bkg}
</section>
"""

SEC_DATES = r"""
<section>
  <div class="sec-head"><span class="sec-n">05</span>
    <h2>Increments across dates</h2></div>
  ${m_05}
  <p class="lede">The same fields at every cached cycle. A configuration that is
  behaving puts its increments in similar places each cycle; a pattern that
  wanders, or grows, is the signature the single-date maps in section 03 cannot
  show. The depth&ndash;cycle panels cover every cycle. Every field in
  <code>state_vars:</code> gets a map per cycle &mdash; pick the field, then the
  date; every date of one field shares one colour scale, so step through them
  and the increment is comparable from one to the next. The menu opens on the
  <b>mean over every cached cycle</b>: what averages away is the day-to-day
  correction, what remains is the systematic one &mdash; the bias the DA
  pushes against cycle after cycle. That mean is several times smaller than
  a single increment (tens of times for SSH and salinity), so it is drawn on
  its own scale; the per-date scale is quoted in its title.</p>
  ${hov}
  <p class="lede">The same depth&ndash;cycle view for the <b>signed,
  area-weighted mean</b> of the increment, by region. The RMS above says how
  large the update is; the mean says which way it goes. A mean that keeps its
  sign at the same depth cycle after cycle is a bias the model is rejecting
  each time it is corrected &mdash; the analysis and the forecast disagree
  systematically there &mdash; and that cannot be read off the magnitude.</p>
  ${hov_mean}
  <p class="lede">The mean increment as a <b>time series</b>, per region:
  one panel per field, the surface level solid and the thickness-weighted
  column mean dashed for the 3-D fields. The 2-D figure below carries the
  same mean under the RMS, where it is too small to read.</p>
  ${mean_series}
  ${inc2d}
  ${seq_figs}
</section>
"""

SEC_VERIF = r"""
<section>
  <div class="sec-head"><span class="sec-n">06</span>
    <h2>Fit to gridded analyses</h2></div>
  ${m_06}
  <p class="lede">Every other section scores this system against its own
  observations or against itself. This one scores the surface state against
  daily L4 products produced outside it: sea surface height against CMEMS
  ADT, salinity against CMEMS SSS, temperature against OSTIA, and sea-ice
  concentration against OSTIA&rsquo;s own ice field. Both the
  background and the analysis are scored, so the bar pairs show whether the
  analysis step moved the state toward the independent product or away from it.
  The <b>analysis</b> here is the background plus the increment: the DA writes
  an increment but no analysis state, so this assumes nothing else acts between
  the two.
  <b>ADT is compared with the mean of each field removed</b> over the points
  where both are valid &mdash; the product is referenced to a mean dynamic
  topography and the model to its own geoid, so the raw difference is dominated
  by a constant offset that carries no information about the ocean. Its bias is
  therefore zero by construction and only the RMS is meaningful. The maps show
  the <b>fields themselves</b> &mdash; the product beside each experiment's
  background and analysis, on one shared colour scale &mdash; rather than their
  difference, so what the model got right is as visible as what it got wrong
  &mdash; and each is followed by the difference, where the error is legible
  at all. At global scale a 0.45&nbsp;&deg;C error vanishes against a
  0&ndash;30&nbsp;&deg;C ramp, so the two are read together. Pick the product,
  then the cycle: the maps exist for every date the background views in
  section 04 do.</p>
  ${f_verif}
  ${f_verif_maps}
  ${f_fronts}
  <p class="lede">The three products above are surface-only and same-day. The
  <b>WOA23</b> views below are the complement: a full-depth reference, so the
  interior can be judged too &mdash; but a climatological one, which is a
  weaker claim. The two are read differently, and the note inside says how.</p>
  ${m_woa}
</section>
"""

SEC_CYCLING = r"""
<section>
  <div class="sec-head"><span class="sec-n">07</span>
    <h2>Cycling behaviour</h2></div>
  ${m_07}
  <p class="lede">${cyc_note} Slow drift is the failure mode a single cycle
  cannot reveal, and the one that most often decides whether a configuration is
  usable. Pick an observation type for its headline metrics against cycle;
  the SSH stability diagnostics follow.</p>
  ${f_stab}
  ${cycle_figs}
  ${f_stability}
</section>
"""

SEC_CALIBRATION = r"""
<section>
  <div class="sec-head"><span class="sec-n">08</span>
    <h2>Ensemble calibration</h2></div>
  ${m_08}
  <p class="lede">Whether the ensemble's own estimate of its error matches the
  error it actually makes. Two of these columns target <b>1</b>: the
  consistency ratio and spread/skill. Values are coloured green within about
  20% of target and red beyond a factor of 1.6. The Desroziers ratios are still
  computed and still scored in <code>scorecard.md</code>, and
  <code>figs/obs_desroziers.png</code> is still written.</p>
  ${t2}
  ${f_cons}
  ${f_spread}
  ${f_rank}
  ${f_ss}
</section>
"""

SEC_BINNED = r"""
<section>
  <div class="sec-head"><span class="sec-n">09</span>
    <h2>Binned departures</h2></div>
  ${m_09}
  <p class="lede">Where each system fits its observations, not just how well.
  Every observation on the common sample is binned on a
  ${bin_deg}&deg; grid: the count, mean and RMS of O&minus;B and O&minus;A,
  the <b>assigned</b> observation error and the <b>effective</b> one (what
  the analysis actually weighted with, after QC and any inflation), and
  RMS(O&minus;B) over each &mdash; the ratio that should sit near 1 if the
  error is right, and reads below 1 where the error is slack and above 1
  where the background is worse than the error budget allows. Profile types
  get the same on depth
  &times; latitude. The <b>regression</b> view puts observation against the
  model at the observation points, background and analysis, as a density
  with the least-squares line: a slope below 1 at high correlation is a
  system damping the signal it assimilates; an analysis line closer to 1:1
  than the background is the update working. Pick the obs type, the view,
  then the date &mdash; <b>all cycles</b> pools every cached cycle, and
  every date of one type shares its colour scale.</p>
  ${f_binned}
</section>
"""

SEC_FORCING = r"""
<section>
  <div class="sec-head"><span class="sec-n">10</span>
    <h2>Atmospheric forcing</h2></div>
  ${m_10}
  <p class="lede">What drove the ocean and ice backgrounds. Each experiment
  here is a coupled run with its own atmosphere, so part of any difference
  between their backgrounds is a difference in forcing rather than in the
  analysis &mdash; and the ocean state feeds back on the winds and fluxes
  above it. The fields are the coupled atmosphere&rsquo;s surface history at
  the same f006 the ocean background is: 10&nbsp;m wind speed, wind stress,
  net surface heat flux into the ocean (SW&nbsp;+&nbsp;LW&nbsp;&minus;&nbsp;LH
  &minus;&nbsp;SH), precipitation and 2&nbsp;m air temperature, over ocean
  points only. The fluxes are the model&rsquo;s averages over the 6&nbsp;h
  ending at the analysis time, so a single map of the heat flux is mostly
  the diurnal cycle at that hour &mdash; compare maps at the same hour, and
  read the daily mean off the time series, which carry every cycle.</p>
  <p class="lede">Coupled runs share the weather to first order, so the
  absolute fields look alike and the divergence between experiments is only
  legible as a <b>difference</b>. The time series are area means by region,
  with a second row of each experiment minus the reference; the maps show
  the reference experiment&rsquo;s forcing on fixed scales, then every other
  experiment as its difference from the reference on a diverging scale fixed
  per field. The reference is the configured one when it keeps an
  atmosphere history, otherwise the first experiment that does (3dvar-rt
  archives its restarts only, so it has no panels here and cannot be the
  reference).</p>
  ${f_atmos_series}
  ${f_atmos_maps}
  ${f_atmos_diff}
</section>
"""

SECTION_TEMPLATES = [SEC_FIT, SEC_USAGE, SEC_STATE, SEC_BACKGROUND, SEC_DATES,
                     SEC_VERIF, SEC_CYCLING, SEC_CALIBRATION, SEC_BINNED,
                     SEC_FORCING]

TAIL = r"""
<section>
  <div class="callout">
    <h3>What these files cannot tell you</h3>
    <ul>
      <li><b>Applied inflation is not recoverable.</b>
      <code>bg_ensvar</code> is bit-identical to <code>ensvar_prior</code>, so
      there is no pre-inflation field to divide by. To get a direct inflation
      panel, write the inflation factor (or the pre-inflation variance) from the
      LETKF.</li>
      <li><b>Sea-ice thickness is a derived ratio.</b> CICE history stores
      grid-cell-mean thickness, so it is divided by concentration to match the
      analysis variable; cells with little ice make that ratio noisy.</li>
      <li><b>Sea-ice comparisons are subset-limited.</b> The LETKF thins ice
      observations, so cross-experiment ice scores rest on the fraction that
      survives thinning; the counts in section 02 show the cost.</li>
      <li><b>3DVar does not update ice or snow thickness.</b> Those increments
      are identically zero, which is why those map panels are labelled rather
      than blank.</li>
    </ul>
  </div>
</section>

<footer>
  Generated by <code>letkf_verif/build_report.py</code> from
  <code>cache/&lt;cycle&gt;.json</code> &middot; ${ncyc} cycles,
  ${first_cycle} to ${newest}<br>
  Regenerate with <code>build_comparison.py</code> (rejoin, figures,
  scorecard, report)
</footer>

</div>
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
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
                    help='extra cache directory to read and merge (repeatable); '
                         'the first is where new results are written')
    ap.add_argument('--out', default=None)
    ap.add_argument('--embed', action='store_true',
                    help='inline every figure as base64 instead of linking '
                         'it from the figures directory (single-file pages, '
                         'tens of MB each)')
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    cycles = P.load_cycles(cfg)
    out = a.out or cfg['report']
    # The output directory is the user's, not the code's, so it may not
    # exist yet.
    page_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(page_dir, exist_ok=True)
    global EMBED, FIG_REL
    EMBED = a.embed
    # Pages link figures relative to their own directory; the tarball keeps
    # the same layout so the links hold once it is unpacked. A figures
    # directory outside the page directory is linked by absolute path and
    # still packed under figs/ -- the pages then only work from the tar.
    FIG_REL = os.path.relpath(cfg['figs'], page_dir).replace(os.sep, '/')
    if FIG_REL.startswith('..'):
        print('  i figures live outside the page directory (%s); pages link '
              'them by absolute path' % cfg['figs'])
        FIG_REL = cfg['figs']
    pages = build(cfg, cycles, out)
    over = []
    for path, content in pages:
        with open(path, 'w') as f:
            f.write(content)
        size = os.path.getsize(path) / 1e6
        print('wrote %s (%.1f MB)' % (path, size))
        if size > WARN_SIZE_MB:
            over.append((path, size))

    # One tarball of every report page plus every figure a page links, for
    # handing the whole thing off in one file -- the report is 8 separate
    # HTML pages so nav between sections works, and the figures are linked
    # rather than embedded, so "send me the report" needs all of it.
    html_files = sorted(f for f in os.listdir(page_dir) if f.endswith('.html'))
    packed = {os.path.abspath(p): rel for p, rel in _USED.items()}
    if FIG_REL == cfg['figs']:            # absolute links: pack under figs/
        packed = {p: 'figs/%s' % os.path.basename(p) for p in packed}
    print('%d pages link %d figures (%.0f MB) under %s'
          % (len(html_files), len(packed),
             sum(os.path.getsize(p) for p in packed) / 1e6, page_dir))
    tar_path = os.path.join(
        page_dir, os.path.splitext(os.path.basename(out))[0] + '.tar')
    with tarfile.open(tar_path, 'w') as tf:
        for f in html_files:
            tf.add(os.path.join(page_dir, f), arcname=f)
        for p, rel in sorted(packed.items()):
            tf.add(p, arcname=rel)
    print('wrote %s (%.0f MB)' % (tar_path, os.path.getsize(tar_path) / 1e6))

    if over:
        # Only reachable with --embed: a linked page is a few hundred KB.
        # Noted here rather than discovered at publish time -- but the pages
        # below were written in full, and nothing about them was reduced.
        print('  i %d page(s) over %.0f MB. Written in full and fine to view '
              'locally; only\n    publishing as an artifact is likely to '
              'refuse them:' % (len(over), WARN_SIZE_MB))
        for p, size in over:
            print('      %-52s %5.1f MB' % (os.path.basename(p), size))
        print('    If you need one smaller, in order of least loss: lower '
              'VERIF_MAP_DPI,\n    then MAP_DPI, in plot_statespace.py; '
              'shorten `regions:`; or drop fields\n    from '
              '`background_vars:` / `state_vars:`.')
    if _MISSING:
        # A figure the report expects but cannot find used to vanish in
        # silence, which is how renamed figures dropped out unnoticed.
        print('  ! %d expected figure(s) missing from %s -- rerun the plot '
              'scripts:' % (len(_MISSING), cfg['figs']))
        for name in sorted(set(_MISSING)):
            print('      %s' % name)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
