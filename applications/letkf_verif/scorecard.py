#!/usr/bin/env python3
"""Cross-experiment scorecard: every experiment against the reference.

Metrics come in two flavours. "Lower is better" ones (departure RMS, CRPS) are
scored as a percentage change against the reference. "Target one" ones
(consistency ratio, spread/skill, the Desroziers ratios) are scored by how far
they sit from 1 in log space, so 0.5 and 2.0 count as equally wrong.

With more than one cycle a paired Wilcoxon signed-rank test over cycles is
reported, because the cycle-to-cycle spread of these metrics is large and a
single-cycle difference is rarely meaningful.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lv_plot as P  # noqa: E402
import plot_statespace as PS  # noqa: E402
from lv_common import load_config  # noqa: E402

LOWER_BETTER = [
    ('ombg_rms', 'RMS(O-B)'),
    ('oman_rms', 'RMS(O-A)'),
    ('crps', 'CRPS'),
]
TARGET_ONE = [
    ('consistency_ratio', 'consistency ratio'),
    ('spread_skill', 'spread / skill'),
    ('desroziers_R_ratio', 'Desroziers R / assigned R'),
    ('desroziers_HBHt_ratio', 'Desroziers HBHt / spread'),
]


def _series(cycles, obstype, name, key, sample='common', stratum='all'):
    """Metric value per cycle, NaN where absent."""
    out = []
    for c in sorted(cycles):
        d = cycles[c]
        out.append(P.get(d.get('obs', {}).get(obstype, {}), sample, name,
                         stratum, key))
    return np.array(out, dtype='f8')


def _wilcoxon(a, b):
    """Paired signed-rank p-value; None when there is too little to test."""
    d = (a - b)[np.isfinite(a) & np.isfinite(b)]
    d = d[d != 0]
    if d.size < 6:
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return None


def _fmt(v, nd=3):
    return '-' if v is None or not np.isfinite(v) else ('%.*f' % (nd, v))


def _pct(new, ref):
    if not (np.isfinite(new) and np.isfinite(ref)) or ref == 0:
        return None
    return 100.0 * (new - ref) / abs(ref)


def _mark(p):
    if p is None:
        return ''
    return ' **' if p < 0.05 else (' *' if p < 0.10 else '')


def build(cfg, cycles):
    names = P.all_exp_names(cfg, cycles)
    ref = cfg.get('reference') or names[0]
    others = [n for n in names if n != ref]
    types = sorted({t for d in cycles.values() for t in d.get('obs', {})})
    ncyc = len(cycles)

    L = []
    L.append('# LETKF verification scorecard')
    L.append('')
    # Chosen per obs type, not once for the whole scorecard: a type only one
    # experiment assimilates must not push every OTHER, fully-shared type
    # onto its own (unjoined) sample too. Caches merged from separate runs
    # cannot carry a real common sample either: each run only ever joined
    # its own experiment.
    any_own = any(P.type_sample(cycles, t) == 'own' for t in types)
    L.append('Reference experiment: **%s**  |  cycles: **%d** (%s)  |  '
             'sample: **%s**'
             % (ref, ncyc, ', '.join(sorted(cycles)),
                'own for some obs types' if any_own else 'common'))
    if any_own:
        L.append('')
        L.append('> **Some obs types use own-sample scores.** No '
                 'cross-experiment common sample exists for them -- an '
                 'experiment does not assimilate that type, or the caches '
                 'were computed in separate runs -- so each experiment is '
                 'scored on the observations it assimilated. Differences '
                 'there are confounded by thinning and QC. Run '
                 '`compute_cycle.py --rejoin` to rebuild a true common sample '
                 'where possible (it re-reads only the observation files).')
    if ncyc < 6:
        L.append('')
        L.append('> Significance testing needs at least 6 cycles; with %d '
                 'cycled date%s the differences below are indicative only.'
                 % (ncyc, '' if ncyc == 1 else 's'))
    L.append('')

    # ---- departure statistics ---------------------------------------------
    L.append('## 1. Fit to observations (lower is better)')
    L.append('')
    L.append('Percentages are change against %s; `**` marks p<0.05 and `*` '
             'p<0.10 on a paired Wilcoxon test over cycles.' % ref)
    for key, label in LOWER_BETTER:
        rows = []
        for t in types:
            sample = P.type_sample(cycles, t)
            r = _series(cycles, t, ref, key, sample)
            others_s = {n: _series(cycles, t, n, key, sample) for n in others}
            # An obs type the REFERENCE doesn't assimilate still gets a row
            # when another experiment has it -- gating on the reference alone
            # dropped every such type even though section 3 (usage) shows it
            # was actually assimilated.
            if not (np.any(np.isfinite(r))
                   or any(np.any(np.isfinite(s)) for s in others_s.values())):
                continue
            cells = [P.short(t), _fmt(np.nanmean(r), 4)]
            for n in others:
                s = others_s[n]
                pc = _pct(np.nanmean(s), np.nanmean(r))
                cells.append('%s (%s%%)%s'
                             % (_fmt(np.nanmean(s), 4),
                                '-' if pc is None else '%+.1f' % pc,
                                _mark(_wilcoxon(s, r))))
            rows.append(cells)
        if not rows:
            continue
        L.append('')
        L.append('### %s' % label)
        L.append('')
        hdr = [ref] + others
        L.append('| obs type | ' + ' | '.join(hdr) + ' |')
        L.append('|---|' + '---|' * len(hdr))
        for c in rows:
            L.append('| %s |' % ' | '.join(c))
    L.append('')

    # ---- ensemble calibration ---------------------------------------------
    ens = [n for n in names
           if any(np.any(np.isfinite(_series(cycles, t, n, 'spread_b',
                                             P.type_sample(cycles, t))))
                  for t in types)]
    if ens:
        L.append('## 2. Ensemble calibration (target = 1)')
        L.append('')
        L.append('| obs type | experiment | %s | sigma_a/sigma_b | rank-end / flat |'
                 % ' | '.join(lab for _, lab in TARGET_ONE))
        L.append('|---|---|' + '---|' * (len(TARGET_ONE) + 2))
        for t in types:
            sample = P.type_sample(cycles, t)
            for n in ens:
                vals = [np.nanmean(_series(cycles, t, n, k, sample))
                        for k, _ in TARGET_ONE]
                if not any(np.isfinite(v) for v in vals):
                    continue
                sr = np.nanmean(_series(cycles, t, n, 'spread_ratio', sample))
                er = np.nanmean([P.get(cycles[c], 'obs', t, 'rank', n,
                                       'ends_ratio')
                                 for c in sorted(cycles)])
                L.append('| %s | %s | %s | %s | %s |'
                         % (P.short(t), n,
                            ' | '.join(_fmt(v) for v in vals),
                            _fmt(sr), _fmt(er, 2)))
        L.append('')

    # ---- observation usage -------------------------------------------------
    L.append('## 3. Observation usage')
    L.append('')
    L.append('`loaded` is what the experiment read, `assimilated` what passed '
             'QC, `common` the intersection used for section 1.')
    L.append('')
    L.append('| obs type | experiment | loaded | assimilated | common | '
             'main QC rejections |')
    L.append('|---|---|---|---|---|---|')
    last = cycles[sorted(cycles)[-1]]
    for t in types:
        o = last.get('obs', {}).get(t)
        if not o:
            continue
        for n in names:
            cnt = o.get('counts', {}).get(n, {})
            qc = {k: v for k, v in o.get('qc', {}).get(n, {}).items()
                  if k != 'pass'}
            top = ', '.join('%s %d' % (k, v) for k, v in
                            sorted(qc.items(), key=lambda kv: -kv[1])[:3])
            # n_common_pass is blanked when caches were merged from separate
            # runs, because no cross-experiment join produced it
            shared = cnt.get('n_common_pass')
            L.append('| %s | %s | %d | %d | %s | %s |'
                     % (P.short(t), n, cnt.get('n_own', 0) or 0,
                        cnt.get('n_pass_own', 0) or 0,
                        '%d' % shared if shared is not None else 'n/a',
                        top or '-'))
    L.append('')

    # ---- against the gridded analyses ---------------------------------------
    import lv_verif as LV
    prods = [p for p in LV.PRODUCTS
             if any(P.get(last, 'state', n, 'ocean', 'verif', p, default=None)
                    for n in names)]
    if prods:
        L.append('## 4. Fit to gridded analyses (lower is better)')
        L.append('')
        L.append('RMS(model - product) over the global ocean. ADT has the mean '
                 'of each field removed over the points where both are valid, '
                 'so its bias is zero by construction and only the RMS is '
                 'meaningful.')
        L.append('')
        hdr = [ref] + others
        L.append('| product | state | ' + ' | '.join(hdr) + ' |')
        L.append('|---|---|' + '---|' * len(hdr))
        for p in prods:
            spec = LV.PRODUCTS[p]
            for state in LV.STATES:
                def rms(n):
                    return P.get(last, 'state', n, 'ocean', 'verif', p, state,
                                 'global', 'rms')
                r = rms(ref)
                cells = [_fmt(r, 4)]
                for n in others:
                    v = rms(n)
                    pc = _pct(v, r)
                    cells.append('%s%s' % (_fmt(v, 4),
                                           '' if pc is None
                                           else ' (%+.1f%%)' % pc))
                if any(c != 'n/a' for c in cells):
                    L.append('| %s | %s | %s |'
                             % (spec['label'],
                                'background' if state == 'bkg' else 'analysis',
                                ' | '.join(cells)))
        L.append('')

    # ---- state space --------------------------------------------------------
    L.append('## 5. State space')
    L.append('')
    levels = cfg.get('map_levels', [0])
    L.append('| field | level | %s |' % ' | '.join(names))
    L.append('|---|---|' + '---|' * len(names))
    for realm in ('ocean', 'ice'):
        for var in cfg.get('state_vars', {}).get(realm, []):
            prof = {n: P.get(last, 'state', n, realm, 'incr_rms', var,
                             default=None) for n in names}
            if not any(prof.values()):
                continue
            avail = [p for p in prof.values() if p]
            ks = levels if max(len(p) for p in avail) > 1 else [0]
            for k in ks:
                cells = []
                for n in names:
                    p = prof[n]
                    cells.append(_fmt(p[k], 5)
                                 if p and k < len(p) else 'n/a')
                L.append('| RMS increment %s | %s | %s |'
                         % (var, PS.level_label(last, k), ' | '.join(cells)))
    for realm, var in (('ocean', 'Temp'), ('ocean', 'Salt')):
        for k in levels:
            cells = []
            for n in names:
                r = P.get(last, 'state', n, realm, 'spread_ratio', var,
                          default=None)
                cells.append(_fmt(r[k]) if r and k < len(r) else 'n/a')
            if any(c != 'n/a' for c in cells):
                L.append('| spread ratio %s | %s | %s |'
                         % (var, PS.level_label(last, k), ' | '.join(cells)))
    L.append('')

    cl_keys = sorted({k for n in names
                      for k in P.get(last, 'state', n, 'ocean', 'corr_length',
                                     default={})})
    if cl_keys:
        boxes = list(cfg.get('corr_regions') or [])
        L.append('### Increment horizontal correlation length (km)')
        L.append('')
        L.append('| field | box | %s |' % ' | '.join(names))
        L.append('|---|---|' + '---|' * len(names))
        for key in cl_keys:
            for b in boxes:
                cells = [_fmt(P.get(last, 'state', n, 'ocean', 'corr_length',
                                    key, b), 0) for n in names]
                if any(c != '-' for c in cells):
                    L.append('| %s | %s | %s |'
                             % (key.replace('_k', ' level '), b,
                                ' | '.join(cells)))
        L.append('')

    return '\n'.join(L) + '\n'


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
    md = build(cfg, cycles)
    out = a.out or cfg['scorecard']
    # The output directory is the user's, not the code's, so it may not
    # exist yet.
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        f.write(md)
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
