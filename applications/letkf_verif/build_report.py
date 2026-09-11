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
import html
import os
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


def hemi_imgs(figs, base, cycle, alt, optional=False):
    """Both hemispheres of a polar figure, in order."""
    return ''.join(
        img(fig_path(figs, '%s_%s' % (base, h), cycle),
            '%s (%s)' % (alt, label), optional=optional)
        for h, label in (('nh', 'Arctic'), ('sh', 'Antarctic')))


def img(path, alt, optional=False):
    """Embed a figure. A missing one is reported, never silently dropped.

    Silently returning '' let renamed figures vanish from the report while
    every script still reported success.
    """
    if not os.path.exists(path):
        if not optional:
            _MISSING.append(os.path.basename(path))
        return ''
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return ('<figure class="fig"><div class="fig-scroll">'
            '<img src="data:image/png;base64,%s" alt="%s"></div>'
            '<figcaption>%s</figcaption></figure>' % (b64, html.escape(alt),
                                                      html.escape(alt)))


def picker_widget(group, items, label_fn, panel_fn, style_fn=None,
                  collapsible=None, dropdown=None):
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
    match a companion map) respectively.
    """
    tabs, rules, panels, options = [], [], [], []
    for i, item in enumerate(items):
        s = P.slug(item)
        tid, pid = '%s-tab-%s' % (group, s), '%s-panel-%s' % (group, s)
        label = label_fn(item)
        default = i == 0
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
            '<select class="picker-select" aria-label="%s" '
            'onchange="lvPickerShow(this)">%s</select>'
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
            % (html.escape(dropdown), ''.join(options)))
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


def frontal_widget(cfg, figs):
    """Configured strong-current maps plus the narrow-jet profile diagnostic."""
    frontal = cfg.get('frontal_analysis') or {}
    if not frontal.get('enabled'):
        return ''
    entries = frontal.get('regions') or []
    entries = (entries.values() if isinstance(entries, dict) else entries)
    regions = {entry['name']: dict(entry or {}) for entry in entries
           if entry and entry.get('name') and os.path.exists(
             os.path.join(figs, 'front_strong_%s.png'
                  % P.slug(entry['name'])))}
    if not regions:
        return ''
    picker = picker_widget(
      'fronts', list(regions),
      label_fn=lambda name: name,
      panel_fn=lambda name: img(
        os.path.join(figs, 'front_strong_%s.png' % P.slug(name)),
            '%s: geostrophic current speed and strong-current footprint'
        % name, optional=True),
        collapsible='choose current')
    profile = img(os.path.join(figs, 'front_profiles.png'),
                  'Cross-front structure for selected coherent jets', optional=True)
    return (
        '<section class="subsection"><h3>Frontal-current placement</h3>'
        '<p class="lede">Broad and branching currents are shown as the area '
        'where geostrophic speed reaches a fixed <b>absolute threshold</b>, '
        'rather than being forced into one artificial axis. Each panel is one '
        'analysis cycle; its configured threshold is identical for Copernicus '
        'L4 ADT and every experiment in that current&rsquo;s box. Copernicus L4 '
        'is a higher-resolution mapped analysis of much of the same altimeter '
        'information, not independent truth.</p>%s%s</section>' % (picker, profile))


def obstype_dropdown_widget(cycles, figs, base, label):
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
    types = [t for t in types if os.path.exists(
        os.path.join(figs, '%s_type_%s.png' % (base, P.slug(t))))]
    return picker_widget(
        base, types,
        label_fn=P.short,
        panel_fn=lambda t: img(
            os.path.join(figs, '%s_type_%s.png' % (base, P.slug(t))),
            '%s: %s' % (P.short(t), label), optional=True),
        dropdown='choose obs type')


def obsfit_widget(cycles, cfg, figs):
    """Obs-type picker + one panel per type for 'fit to observations'."""
    return obstype_dropdown_widget(cycles, figs, 'obsfit',
                                   'RMS and bias of the fit to observations '
                                   'across cycles')


def counts_widget(cycles, cfg, figs):
    """Obs-type picker + one panel per type for 'observation usage'."""
    return obstype_dropdown_widget(cycles, figs, 'obscount',
                                   'observations assimilated per cycle')


def profiles_widget(data, cfg, figs):
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
        os.path.join(figs, 'obs_profiles_region_%s.png' % P.slug(r)))]
    if not regions:
        return ''

    basin_color = basin_colors(cfg)
    return picker_widget(
        'profiles', regions,
        label_fn=lambda r: r.replace('_', ' '),
        panel_fn=lambda r: img(
            os.path.join(figs, 'obs_profiles_region_%s.png' % P.slug(r)),
            '%s: profile observations against depth' % r.replace('_', ' '),
            optional=True),
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


def sequence_widget(group, cfg, figs, data):
    """Field picker over the across-date increment sequences.

    plot_statespace.py writes one 'seq_<realm>_incr_<field>_k<lev>.png' per
    configured field, split per hemisphere for ice. The menu is ordered from
    `state_vars` -- the same list that decides which sequences get drawn --
    so it reads in config order rather than the alphabetical order a plain
    directory glob produced, and the level labels carry their approximate
    depth the way the figures' own row labels do. Any seq_ file the config
    no longer names is still appended rather than silently dropped, so a
    stale figure is visible instead of invisible.
    """
    levels = cfg.get('map_levels', [0])
    svars = cfg.get('state_vars', {})
    want = []
    for v in svars.get('ocean', []):
        flat = v in PS.NO_LEVEL_LABEL
        for k in ([0] if flat else levels):
            want.append(('seq_ocean_incr_%s_k%d.png' % (v, k),
                         v if flat else '%s, %s' % (v, PS.level_label(data, k))))
    for v in svars.get('ice', []):
        for h in ('nh', 'sh'):
            want.append(('seq_ice_incr_%s_k0_%s.png' % (v, h),
                         '%s, %s' % (v, PS.HEMIS[h][0])))
    named = {f for f, _label in want}
    want += [(f, f[4:-4]) for f in sorted(os.listdir(figs))
             if f.startswith('seq_') and f not in named]
    # optional=True throughout: a configured field legitimately has no
    # sequence when only one cycle is cached, or when no experiment writes
    # it. figure_menu drops those empty entries and unwraps a lone survivor.
    return figure_menu(group, 'increment sequence', [
        (label, img(os.path.join(figs, f),
                    'Increment across dates: %s' % label, optional=True))
        for f, label in want])


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

def _series(cycles, obstype, name, key, sample):
    """Metric value per cycle for one experiment, NaN where absent."""
    return np.array([P.get(cycles[c].get('obs', {}).get(obstype, {}),
                           sample, name, 'all', key)
                     for c in sorted(cycles)], dtype='f8')


def _mean(v):
    """nanmean that returns NaN instead of warning on an all-NaN slice."""
    return float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan


def _cycle_set(cycles, name):
    return {c for c in cycles if name in P.exp_names(cycles[c])}


def build(cfg, cycles, out):
    last = sorted(cycles)[-1]
    data = cycles[last]
    # The union across every cached cycle, not just exp_names() on ``last``:
    # a config whose experiments cover disjoint cycle windows (see the
    # caveat in experiments.yaml) has no single cycle where all of them are
    # present, so reading the experiment list off one cycle -- even the
    # latest -- silently drops whichever experiments do not reach that date.
    names = P.all_exp_names(cfg, cycles)
    labels = {e.name: e.label for e in cfg['experiments']}
    ref = cfg.get('reference') or names[0]
    others = [n for n in names if n != ref]
    col = P.color_map(names)
    figs = cfg['figs']

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
        cells = [P.short(t), fmt(r, 4), fmt(ra, 4)]
        for n in others:
            ob, oa = own_vals[n]
            cells.append(delta_cell(ob, r))
            cells.append(delta_cell(oa, ra))
        rows.append(cells)
    hdr = ['obs type', '%s O&minus;B' % ref, '%s O&minus;A' % ref]
    for n in others:
        hdr += ['%s O&minus;B' % n, '%s O&minus;A' % n]
    t1 = table(hdr, rows)

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
    levels = cfg.get('map_levels', [0])
    rows = []
    for realm in ('ocean', 'ice'):
        for var in cfg.get('state_vars', {}).get(realm, []):
            prof = {n: P.get(data['state'], n, realm, 'incr_rms', var,
                             default=None) for n in names}
            avail = [p for p in prof.values() if p]
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
                    p = prof[n]
                    cells.append(fmt(p[k], 5) if p and k < len(p) else '&mdash;')
                for n in names:
                    r = P.get(data['state'], n, realm, 'spread_ratio', var,
                              default=None)
                    cells.append(fmt(r[k]) if r and k < len(r) else '&mdash;')
                rows.append(cells)
    t4 = table(['field', 'level']
               + ['RMS incr %s' % n for n in names]
               + ['&sigma;<sub>a</sub>/&sigma;<sub>b</sub> %s' % n
                  for n in names], rows)

    f_increg = regional_widget('increg', cfg, figs, last,
                               'state_increment_regions', 'RMS increment')
    f_bkgreg = regional_widget('bkgreg', cfg, figs, last,
                               'bkg_profiles_regions', 'background mean')
    f_cons = img(os.path.join(figs, 'obs_consistency.png'),
                 'Departure against the spread that should match it',
                 optional=not has_ens)
    f_bkgprof = img(fig_path(figs, 'bkg_profiles', last),
                    'Background mean temperature and salinity against depth')
    f_bkgocn = img(fig_path(figs, 'bkg_maps_ocean', last),
                   'Ocean background state')
    f_bkgice = hemi_imgs(figs, 'bkg_maps_ice', last, 'Sea-ice background state')
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

    seq_figs = sequence_widget('seq', cfg, figs, data)
    hov = img(os.path.join(figs, 'cycle_increment_hovmoller.png'),
              'RMS increment against depth and cycle', optional=True)
    inc2d = img(os.path.join(figs, 'cycle_increment_2d.png'),
                '2-D field increment magnitude across cycles', optional=True)

    cycle_figs = ''.join(
        img(os.path.join(figs, 'cycle_%s.png' % t),
            '%s: headline metrics across cycles' % P.short(t), optional=True)
        for t in types)

    ncyc = len(cycles)
    cyc_note = ('Only one cycle is cached, so these panels are dot plots. '
                'Add cycles with <code>compute_cycle.py</code> and they become '
                'time series; the paired significance test in '
                '<code>scorecard.md</code> needs at least six.'
                if ncyc < 6 else
                '%d cycles cached.' % ncyc)

    subs = dict(
        cycle=last, ncyc=ncyc,
        cycle_list=', '.join(sorted(cycles)),
        nexp=len(names), ref=html.escape(ref),
        legend=legend,
        t1=t1, t2=t2, t4=t4,
        f_departures=img(os.path.join(figs, 'obs_departures.png'),
                         'Background and analysis fit to observations, common sample'),
        f_spread=img(os.path.join(figs, 'obs_spread.png'),
                     'Consistency ratio and posterior/prior spread',
                     optional=not has_ens),
        f_rank=img(os.path.join(figs, 'obs_rank_histograms.png'),
                   'Rank histograms of the observation within the prior ensemble',
                   optional=not has_ens),
        f_ss=img(os.path.join(figs, 'obs_spread_skill.png'),
                 'Spread-skill relationship, binned by ensemble spread',
                 optional=not has_ens),
        f_prof=profiles_widget(data, cfg, figs),
        # only written when `ocean_basin_mask:` is configured
        f_regions=img(os.path.join(figs, 'ocean_regions.png'),
                      'Ocean basins and named regions used for every '
                      'regional breakdown in this report', optional=True),
        f_counts=counts_widget(cycles, cfg, figs),
        f_incr=img(fig_path(figs, 'state_increment_profiles', last), 'RMS analysis increment against depth'),
        f_sprprof=img(fig_path(figs, 'state_spread_profiles', last),
                      'Prior and posterior ensemble spread against depth',
                      optional=not has_ens),
        f_sprreg=img(fig_path(figs, 'state_spread_regions', last),
                     'Ensemble spread and spread reduction by region',
                     optional=not has_ens),
        f_map_ocn=img(fig_path(figs, 'state_maps_ocean_increment', last), 'Ocean analysis increment maps'),
        f_map_spr=img(fig_path(figs, 'state_maps_ocean_spread_reduction', last),
                      'Ocean ensemble spread reduction maps', optional=not has_ens),
        f_map_ice=hemi_imgs(figs, 'state_maps_ice_increment', last,
                            'Sea-ice analysis increment maps'),
        f_ice=hemi_imgs(figs, 'state_maps_ice_spread_reduction', last,
                        'Sea-ice ensemble spread reduction maps', optional=not has_ens),
        # only written when the post-inflation variance file is present, so
        # optional -- but reported as missing if it is, like every other figure
        f_map_inf=img(fig_path(figs, 'state_maps_ocean_inflation', last),
                      'Ocean applied-inflation maps', optional=True),
        f_ice_inf=hemi_imgs(figs, 'state_maps_ice_inflation', last,
                            'Sea-ice applied-inflation maps', optional=True),
        f_stab=img(os.path.join(figs, 'cycle_stability.png'),
                   'Cycling stability of the background fit', optional=True),
        # empty when `verification:` names no products, or none resolved to
        # a real file for any cached cycle
        f_verif=verif_widget(cycles, cfg, names, figs),
        f_obsfit=obsfit_widget(cycles, cfg, figs),
        # fields then their difference, per product: the fields say whether
        # the model reproduces the product, the difference is where the error
        # is actually legible
        f_verif_maps=''.join(
            img(fig_path(figs, 'verif_maps_%s' % p, last),
                '%s: product beside the model background and analysis'
                % p.upper(), optional=True)
            + img(fig_path(figs, 'verif_diff_%s' % p, last),
                  'Model minus %s' % p.upper(), optional=True)
            for p in ('adt', 'sss', 'sst')),
              f_fronts=frontal_widget(cfg, figs),
        sample_note=sample_note, overlap_warning=overlap_warning,
        cycle_figs=cycle_figs, cyc_note=cyc_note,
        seq_figs=seq_figs, hov=hov, inc2d=inc2d,
        f_bkgprof=f_bkgprof, f_bkgocn=f_bkgocn, f_bkgice=f_bkgice,
        f_increg=f_increg, f_bkgreg=f_bkgreg, f_cons=f_cons,
        f_incrsec=sections_widget('incrsec', cfg, figs, last, 'incr'),
        f_bkgsec=sections_widget('bkgsec', cfg, figs, last, 'bkg'),
        f_woasec=sections_widget('woasec', cfg, figs, last, 'woa_bias'),
        f_woaprof=img(fig_path(figs, 'woa_bias_profiles', last),
                      'Background and the WOA23 climatology against depth, '
                      'and their difference', optional=True),
        f_woabias=img(fig_path(figs, 'woa_bias_maps_ocean', last),
                      'Background minus the WOA23 climatology', optional=True),
        f_woareg=regional_widget('woareg', cfg, figs, last,
                                 'woa_bias_regions',
                                 'background $-$ WOA23'),
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
        ('Ensemble spread, global', subs['f_sprprof']),
        ('Ensemble spread by region', subs['f_sprreg'])])
    subs['m_state_maps'] = figure_menu('statemaps', 'Map view', [
        ('Ocean increment', subs['f_map_ocn']),
        ('Ocean increment sections', note(incr_sec_note, subs['f_incrsec'])),
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
.secnav{display:flex; flex-wrap:wrap; gap:4px 2px}
.secnav a{display:inline-flex; align-items:baseline; gap:7px;
  padding:7px 12px; border-radius:8px; text-decoration:none;
  color:var(--ink-2); font-size:13px; font-weight:500}
.secnav a .sec-n{font-family:var(--mono); font-size:11px; color:var(--ink-3);
  font-weight:600}
.secnav a:hover{background:var(--surface-2); color:var(--ink)}
.secnav a.on{background:var(--accent-soft); color:var(--accent)}
.secnav a.on .sec-n{color:var(--accent)}

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

/* ---- figures ---- */
.fig{margin:0; background:var(--surface); border:1px solid var(--rule);
  border-radius:10px; padding:12px; box-shadow:var(--shadow);
  display:flex; flex-direction:column; gap:8px}
.fig-scroll{overflow-x:auto}
.fig img{display:block; max-width:100%; height:auto; margin:0 auto;
  border-radius:4px}
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
.delta.bad{color:var(--alert)}
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
.picker-select{flex:1 0 100%; font:inherit; font-size:13px; padding:7px 12px;
  border-radius:8px; border:1.5px solid var(--rule); background:var(--surface);
  color:var(--ink); cursor:pointer; max-width:100%}
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
<div class="wrap">

<header class="mast">
  <span class="eyebrow">Marine data assimilation &middot; verification</span>
  <h1>Marine analysis verification</h1>
  <p class="sub">Every number below is computed from the analysis output of
  ${nexp} experiments. ${sample_note}</p>
  <div class="meta">
    <span>cycle <b>${cycle}</b></span>
    <span>cached cycles <b>${ncyc}</b></span>
    <span>reference <b>${ref}</b></span>
  </div>
  <div class="exps">${legend}</div>
</header>

${nav}
"""

SEC_FIT = r"""
<section>
  <div class="sec-head"><span class="sec-n">01</span>
    <h2>Fit to observations</h2></div>
  <p class="lede">How close each background and analysis lands to the
  observations it was scored against, averaged over every cycle each
  experiment has cached. Lower is better; percentages are the change against
  <code>${ref}</code>. In the profile figure each depth bin carries the whole
  error budget on one axis:
  RMS(O&minus;B) against
  &radic;(&sigma;<sub>b</sub><sup>2</sup>&nbsp;+&nbsp;R<sup>2</sup>), the value
  it should equal when the ensemble and the assigned observation error are
  consistent, with those two contributions drawn behind it.</p>
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
  ${m_bkg}
</section>
"""

SEC_DATES = r"""
<section>
  <div class="sec-head"><span class="sec-n">05</span>
    <h2>Increments across dates</h2></div>
  <p class="lede">The same fields at every cached cycle. A configuration that is
  behaving puts its increments in similar places each cycle; a pattern that
  wanders, or grows, is the signature the single-date maps in section 03 cannot
  show. The depth&ndash;cycle panels cover every cycle; the map sequences are
  subsampled evenly when there are many. Every field in <code>state_vars:</code>
  gets a sequence &mdash; pick one from the menu below.</p>
  ${hov}
  ${inc2d}
  ${seq_figs}
</section>
"""

SEC_VERIF = r"""
<section>
  <div class="sec-head"><span class="sec-n">06</span>
    <h2>Fit to gridded analyses</h2></div>
  <p class="lede">Every other section scores this system against its own
  observations or against itself. This one scores the surface state against
  three daily L4 products produced outside it: sea surface height against CMEMS
  ADT, salinity against CMEMS SSS, and temperature against OSTIA. Both the
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
  0&ndash;30&nbsp;&deg;C ramp, so the two are read together.</p>
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
  <p class="lede">${cyc_note} Slow drift is the failure mode a single cycle
  cannot reveal, and the one that most often decides whether a configuration is
  usable.</p>
  ${f_stab}
  ${cycle_figs}
</section>
"""

SEC_CALIBRATION = r"""
<section>
  <div class="sec-head"><span class="sec-n">08</span>
    <h2>Ensemble calibration</h2></div>
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

SECTION_TEMPLATES = [SEC_FIT, SEC_USAGE, SEC_STATE, SEC_BACKGROUND, SEC_DATES,
                     SEC_VERIF, SEC_CYCLING, SEC_CALIBRATION]

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
  Generated by <code>tools/letkf_verif/build_report.py</code> from
  <code>cache/${cycle}.json</code> &middot; cycles: ${cycle_list}<br>
  Regenerate with <code>compute_cycle.py &amp;&amp; plot_obsspace.py &amp;&amp;
  plot_statespace.py &amp;&amp; plot_timeseries.py &amp;&amp; scorecard.py
  &amp;&amp; build_report.py</code>
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
    a = ap.parse_args(argv)
    a.config = a.config or a.config_opt
    cfg = load_config(a.config, a.root, a.outdir, a.cache)
    cycles = P.load_cycles(cfg)
    out = a.out or cfg['report']
    # The output directory is the user's, not the code's, so it may not
    # exist yet.
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pages = build(cfg, cycles, out)
    over = []
    for path, content in pages:
        with open(path, 'w') as f:
            f.write(content)
        size = os.path.getsize(path) / 1e6
        print('wrote %s (%.1f MB)' % (path, size))
        if size > WARN_SIZE_MB:
            over.append((path, size))

    # One tarball of every report page, for handing the whole thing off in
    # one file -- the report is 8 separate HTML pages so nav between
    # sections works, but that also means "send me the report" needs all 8.
    page_dir = os.path.dirname(os.path.abspath(out))
    html_files = sorted(f for f in os.listdir(page_dir) if f.endswith('.html'))
    tar_path = os.path.join(
        page_dir, os.path.splitext(os.path.basename(out))[0] + '.tar')
    with tarfile.open(tar_path, 'w') as tf:
        for f in html_files:
            tf.add(os.path.join(page_dir, f), arcname=f)
    print('wrote %s (%d files)' % (tar_path, len(html_files)))

    if over:
        # Every figure is embedded as base64, so a page grows with the number
        # of FIELDS and REGIONS on it rather than with the number of cycles.
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
