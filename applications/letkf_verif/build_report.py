#!/usr/bin/env python3
"""Build the self-contained HTML verification report from the cache and figures.

Figures are embedded as data URIs so the page stands alone. Everything shown
comes from cache/<cycle>.json -- the report never re-reads the model output.
"""

import argparse
import base64
import html
import os
import sys
from string import Template

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lv_plot as P  # noqa: E402
import plot_statespace as PS  # noqa: E402
from lv_common import load_config  # noqa: E402

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

def build(cfg, cycles):
    last = sorted(cycles)[-1]
    data = cycles[last]
    names = P.exp_names(data)
    labels = {n: data['experiments'][n].get('label', n) for n in names}
    ref = cfg.get('reference') or names[0]
    others = [n for n in names if n != ref]
    types = sorted(data['obs'])
    col = P.color_map(names)
    figs = cfg['figs']

    legend = ''.join(
        '<span class="exp"><i style="background:%s"></i>%s'
        '<em>%s</em></span>'
        % (col[n], html.escape(labels[n]),
           ' &middot; reference' if n == ref else '')
        for n in names)

    # -- section 1: fit ------------------------------------------------------
    rows = []
    for t in types:
        r = P.get(data['obs'][t], 'common', ref, 'all', 'ombg_rms')
        ra = P.get(data['obs'][t], 'common', ref, 'all', 'oman_rms')
        cells = [P.short(t), fmt(r, 4), fmt(ra, 4)]
        for n in others:
            cells.append(delta_cell(
                P.get(data['obs'][t], 'common', n, 'all', 'ombg_rms'), r))
            cells.append(delta_cell(
                P.get(data['obs'][t], 'common', n, 'all', 'oman_rms'), ra))
        rows.append(cells)
    hdr = ['obs type', '%s O&minus;B' % ref, '%s O&minus;A' % ref]
    for n in others:
        hdr += ['%s O&minus;B' % n, '%s O&minus;A' % n]
    t1 = table(hdr, rows)

    # -- calibration ---------------------------------------------------------
    # The Desroziers ratios and the rank-histogram end ratio are still computed
    # and still drive scorecard.md; they are off this table because reading a
    # tuning decision off them belongs with the figures, not a wide grid.
    ens = [n for n in names
           if any(np.isfinite(P.get(data['obs'][t], 'common', n, 'all',
                                    'spread_b')) for t in types)]
    # Rank histograms, spread-skill, consistency and spread-reduction figures
    # all read ensemble spread, which a var-only comparison never has -- those
    # panels are then correctly absent rather than a renamed figure gone
    # missing, so only require them when some experiment is an ensemble.
    has_ens = bool(ens)
    rows = []
    for t in types:
        for n in ens:
            m = P.get(data['obs'][t], 'common', n, 'all', default={})
            rows.append([
                P.short(t), html.escape(labels[n]),
                target_cell(P.get(m, 'consistency_ratio')),
                target_cell(P.get(m, 'spread_skill')),
                fmt(P.get(m, 'spread_ratio')),
                fmt(P.get(m, 'crps'), 4)])
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

    f_increg = img(fig_path(figs, 'state_increment_regions', last),
                   'RMS increment by region, band = spatial spread')
    f_bkgreg = img(fig_path(figs, 'bkg_profiles_regions', last),
                   'Background mean by region, band = spatial spread')
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

    sample_note = (
        'Headline comparisons use the <b>common sample</b> &mdash; observations '
        'present and passing quality control in <em>every</em> experiment '
        '&mdash; because the configurations do not assimilate the same '
        'observations.' if data.get('common_valid', True) else
        '<b>Own-sample scores.</b> These caches were computed in separate runs, '
        'so no cross-experiment common sample exists and each experiment is '
        'scored on the observations it assimilated; differences are confounded '
        'by thinning and QC. Run <code>compute_cycle.py --rejoin</code> to '
        'rebuild a true common sample.')

    seq_figs = ''.join(
        img(os.path.join(figs, f), 'Increment across dates: %s' % f[4:-4])
        for f in sorted(os.listdir(figs)) if f.startswith('seq_'))
    hov = img(os.path.join(figs, 'cycle_increment_hovmoller.png'),
              'RMS increment against depth and cycle', optional=True)
    inc2d = img(os.path.join(figs, 'cycle_increment_2d.png'),
                '2-D field increment magnitude across cycles', optional=True)

    cycle_figs = ''.join(
        img(os.path.join(figs, 'cycle_%s.png' % t),
            '%s: headline metrics across cycles' % P.short(t))
        for t in types)

    ncyc = len(cycles)
    cyc_note = ('Only one cycle is cached, so these panels are dot plots. '
                'Add cycles with <code>compute_cycle.py</code> and they become '
                'time series; the paired significance test in '
                '<code>scorecard.md</code> needs at least six.'
                if ncyc < 6 else
                '%d cycles cached.' % ncyc)

    return Template(TEMPLATE).substitute(
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
        f_prof=img(os.path.join(figs, 'obs_profiles.png'),
                   'Profile observation departures by region, with the error budget'),
        f_counts=img(os.path.join(figs, 'cycle_obs_counts.png'),
                     'Observations passing QC per cycle, one panel per obs type'),
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
        # written only when `verification:` names products, hence optional --
        # but a miss is still reported rather than silently dropped
        # across-date, so not per-cycle tagged
        f_verif=img(os.path.join(figs, 'cycle_verif_scores.png'),
                    'Fit to gridded analyses per cycle', optional=True),
        f_obsfit=img(os.path.join(figs, 'cycle_obs_fit.png'),
                     'Background and analysis fit per cycle', optional=True),
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
        sample_note=sample_note,
        cycle_figs=cycle_figs, cyc_note=cyc_note,
        seq_figs=seq_figs, hov=hov, inc2d=inc2d,
        f_bkgprof=f_bkgprof, f_bkgocn=f_bkgocn, f_bkgice=f_bkgice,
        f_increg=f_increg, f_bkgreg=f_bkgreg, f_cons=f_cons,
        drift=drift)


TEMPLATE = r"""<title>Marine LETKF Scorecard</title>
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

<div class="wrap">

<header class="mast">
  <span class="eyebrow">Marine data assimilation &middot; verification</span>
  <h1>LETKF against a tuned 3DVar</h1>
  <p class="sub">Every number below is computed from the analysis output of
  ${nexp} experiments. ${sample_note}</p>
  <div class="meta">
    <span>cycle <b>${cycle}</b></span>
    <span>cached cycles <b>${ncyc}</b></span>
    <span>reference <b>${ref}</b></span>
  </div>
  <div class="exps">${legend}</div>
</header>

<section>
  <div class="sec-head"><span class="sec-n">01</span>
    <h2>Fit to observations</h2></div>
  <p class="lede">How close each background and analysis lands to the
  observations it was scored against. Lower is better; percentages are the
  change against <code>${ref}</code> on the common sample. In the profile
  figure each depth bin carries the whole error budget on one axis:
  RMS(O&minus;B) against
  &radic;(&sigma;<sub>b</sub><sup>2</sup>&nbsp;+&nbsp;R<sup>2</sup>), the value
  it should equal when the ensemble and the assigned observation error are
  consistent, with those two contributions drawn behind it.</p>
  ${t1}
  ${f_departures}
  ${f_obsfit}
  ${f_prof}
</section>

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
  ${f_incr}
  ${f_increg}
  ${f_sprprof}
  ${f_sprreg}
  ${f_map_ocn}
  ${f_map_spr}
  ${f_map_inf}
  ${f_map_ice}
  ${f_ice}
  ${f_ice_inf}
</section>

<section>
  <div class="sec-head"><span class="sec-n">04</span>
    <h2>Background state</h2></div>
  <p class="lede">What the increments are correcting, and whether the mean
  state is holding still. Departures and increments can both look healthy while
  the model climate walks away &mdash; the global means below are the only view
  here that catches that, and they need several cycles to be worth reading.
  Depth comes from the background layer thickness, so it is the model's own
  geometry rather than a nominal axis.</p>
  ${drift}
  ${f_bkgprof}
  ${f_bkgreg}
  ${f_bkgocn}
  ${f_bkgice}
</section>

<section>
  <div class="sec-head"><span class="sec-n">05</span>
    <h2>Increments across dates</h2></div>
  <p class="lede">The same fields at every cached cycle. A configuration that is
  behaving puts its increments in similar places each cycle; a pattern that
  wanders, or grows, is the signature the single-date maps in section 03 cannot
  show. The depth&ndash;cycle panels cover every cycle; the map sequences are
  subsampled evenly when there are many.</p>
  ${hov}
  ${inc2d}
  ${seq_figs}
</section>

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
</section>

<section>
  <div class="sec-head"><span class="sec-n">07</span>
    <h2>Cycling behaviour</h2></div>
  <p class="lede">${cyc_note} Slow drift is the failure mode a single cycle
  cannot reveal, and the one that most often decides whether a configuration is
  usable.</p>
  ${f_stab}
  ${cycle_figs}
</section>

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
    with open(out, 'w') as f:
        f.write(build(cfg, cycles))
    size = os.path.getsize(out) / 1e6
    print('wrote %s (%.1f MB)' % (out, size))
    if size > 16.0:
        # Publishing the report as an artifact caps at 16 MB, and every figure
        # is embedded as base64, so the file grows with the number of FIELDS
        # and REGIONS rather than with the number of cycles. Say so here rather
        # than let it be discovered at publish time.
        print('  ! over the 16 MB limit for publishing this as an artifact.\n'
              '    It grows with the field and region counts. In order of\n'
              '    least loss: lower VERIF_MAP_DPI, then MAP_DPI, in\n'
              '    plot_statespace.py; shorten `regions:`; or drop fields from\n'
              '    `background_vars:` / `state_vars:`.')
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
