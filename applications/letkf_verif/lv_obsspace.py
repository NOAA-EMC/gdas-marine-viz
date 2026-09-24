"""Observation-space metrics for the LETKF verification suite.

Every metric is computed on two samples: the *common* sample (observations
present and passing QC in every experiment being compared) and each
experiment's *own* sample. The common sample is the one that can be compared
across experiments; the own sample says what each configuration actually did.
Keeping both visible is the only honest way to report a system that thins its
observations.
"""

import numpy as np

from lv_common import in_region, mean, rms, signed_sqrt

# obs types whose statistics are meaningful per depth bin rather than per
# latitude band
PROFILE_TYPES = ('insitu_temp_profile', 'insitu_salt_profile')


def is_profile(obstype):
    return any(k in obstype for k in PROFILE_TYPES)


def is_ice(obstype):
    return 'icec' in obstype


# --------------------------------------------------------------------------

def core_metrics(s, sel):
    """Scalar metrics for one ObsSet restricted to boolean mask ``sel``.

    ``s`` is an ObsSet; ``sel`` selects the observations to score.
    """
    n = int(np.count_nonzero(sel))
    out = {'n': n}
    if n == 0:
        return out

    ombg = s.ombg[sel]
    oman = s.oman[sel]
    Reff = s.Reff[sel] if s.Reff is not None else None
    Rraw = s.R[sel] if s.R is not None else None

    out['ombg_mean'] = mean(ombg)
    out['ombg_rms'] = rms(ombg)
    out['oman_mean'] = mean(oman)
    out['oman_rms'] = rms(oman)
    if out['ombg_rms']:
        out['fit_ratio'] = out['oman_rms'] / out['ombg_rms']

    out['R_assigned'] = rms(Reff) if Reff is not None else np.nan
    out['R_raw'] = rms(Rraw) if Rraw is not None else np.nan

    # Desroziers. E[d_ob . d_oa] = R and E[d_ab . d_ob] = HBH^T, valid when the
    # gain is the one implied by the assumed statistics -- the ratios below are
    # exactly the statement "is my assumed R (or my ensemble spread) right?".
    dab = ombg - oman
    out['desroziers_R'] = signed_sqrt(mean(ombg * oman))
    out['desroziers_HBHt'] = signed_sqrt(mean(dab * ombg))
    if out['R_assigned']:
        out['desroziers_R_ratio'] = out['desroziers_R'] / out['R_assigned']

    if s.spread_b is not None:
        sb = rms(s.spread_b[sel])
        out['spread_b'] = sb
        if s.spread_a is not None:
            sa = rms(s.spread_a[sel])
            out['spread_a'] = sa
            if sb:
                out['spread_ratio'] = sa / sb
        if sb and out['desroziers_HBHt'] == out['desroziers_HBHt']:
            out['desroziers_HBHt_ratio'] = out['desroziers_HBHt'] / sb
        # Consistency ratio: total prior variance in obs space against the
        # variance actually observed. 1 is right, <1 under-dispersive.
        if out['ombg_rms']:
            R2 = out['R_assigned'] ** 2 if out['R_assigned'] == out['R_assigned'] else 0.0
            out['consistency_ratio'] = (sb ** 2 + R2) / out['ombg_rms'] ** 2
            out['spread_skill'] = sb / out['ombg_rms']
    if s.crps is not None:
        out['crps'] = mean(s.crps[sel])
    return out


def rank_histogram(s, sel):
    """Talagrand histogram of the observation within the prior ensemble."""
    if s.rank is None:
        return None
    r = s.rank[sel]
    r = r[r >= 0]
    if r.size == 0:
        return None
    h = np.bincount(r, minlength=s.nens + 1)[:s.nens + 1]
    flat = r.size / (s.nens + 1.0)
    # Reliability index: L1 distance of the normalised histogram from flat.
    ri = float(np.abs(h / r.size - 1.0 / (s.nens + 1)).sum())
    return {'counts': [int(x) for x in h],
            'n': int(r.size),
            'flat': float(flat),
            'ends_ratio': float((h[0] + h[-1]) / (2.0 * flat)) if flat else np.nan,
            'reliability_index': ri}


def spread_skill_curve(s, sel, nbin=10):
    """RMS(OmB) against ensemble spread, binned by spread quantile.

    A perfectly calibrated ensemble puts these points on the 1:1 line once the
    observation error is included, so the fitted slope is a direct read on
    whether inflation is doing enough.
    """
    if s.spread_b is None:
        return None
    sb = s.spread_b[sel]
    ob = s.ombg[sel]
    R = s.Reff[sel] if s.Reff is not None else np.zeros_like(sb)
    ok = np.isfinite(sb) & np.isfinite(ob) & np.isfinite(R)
    sb, ob, R = sb[ok], ob[ok], R[ok]
    if sb.size < 10 * nbin:
        return None
    edges = np.quantile(sb, np.linspace(0, 1, nbin + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    idx = np.clip(np.searchsorted(edges, sb, side='right') - 1, 0, nbin - 1)
    xs, ys, ns = [], [], []
    for b in range(nbin):
        m = idx == b
        if np.count_nonzero(m) < 10:
            continue
        # expected departure spread includes the observation error
        xs.append(float(np.sqrt(np.mean(sb[m] ** 2 + R[m] ** 2))))
        ys.append(rms(ob[m]))
        ns.append(int(np.count_nonzero(m)))
    if len(xs) < 3:
        return None
    slope = float(np.polyfit(xs, ys, 1)[0])
    return {'spread': xs, 'rmse': ys, 'n': ns, 'slope': slope}


# --------------------------------------------------------------------------

def strata(s, obstype, cfg):
    """Yield (stratum name, boolean mask) for one ObsSet."""
    yield 'all', np.ones(s.n, dtype=bool)

    if is_profile(obstype) and 'depth' in s.meta:
        d = s.meta['depth']
        bins = cfg.get('depth_bins', [])
        lat = s.meta.get('latitude')
        # Depth bins globally, and again within each region, so profile
        # departures can be read by region rather than only as a global mean.
        # Regions are the same entries the state-space profiles use, and are
        # applied in longitude too where they carry it -- what `lat_bands:`
        # could not do.
        bands = [('', np.ones(s.n, dtype=bool))]
        if lat is not None:
            lon = s.meta.get('longitude')
            bands += [(r['name'] + '/', in_region(r, lat, lon))
                      for r in cfg.get('regions', [])]
        for prefix, bsel in bands:
            if prefix:
                yield prefix.rstrip('/'), bsel
            for lo, hi in zip(bins[:-1], bins[1:]):
                yield ('%sdepth_%g_%g' % (prefix, lo, hi),
                       bsel & (d >= lo) & (d < hi))
    elif is_ice(obstype) and 'latitude' in s.meta:
        lat = s.meta['latitude']
        yield 'NH', lat > 0
        yield 'SH', lat <= 0
    elif 'latitude' in s.meta:
        lat, lon = s.meta['latitude'], s.meta.get('longitude')
        for r in cfg.get('regions', []):
            yield r['name'], in_region(r, lat, lon)


def compute(obstype, aligned, counts, common_pass, own, cfg):
    """Assemble every obs-space diagnostic for one obs type.

    ``aligned`` maps experiment -> ObsSet on the common (joined) sample,
    ``own`` maps experiment -> ObsSet on that experiment's full sample.
    """
    any_set = next(iter(aligned.values()))
    res = {
        'var': any_set.var,
        'is_profile': is_profile(obstype),
        'is_ice': is_ice(obstype),
        'counts': counts,
        'qc': {n: s.qc_counts() for n, s in own.items()},
        'nens': {n: int(s.nens) for n, s in own.items()},
        # the experiments actually joined for the common sample; a merge of
        # separately-computed caches uses this to tell whether 'common' is
        # still a common sample at all
        'common_experiments': sorted(aligned),
        'common': {},
        'own': {},
        'rank': {},
        'spread_skill': {},
    }

    for name, s in aligned.items():
        res['common'][name] = {
            st: core_metrics(s, sel & common_pass)
            for st, sel in strata(s, obstype, cfg)
        }
    for name, s in own.items():
        p = s.passed()
        res['own'][name] = {
            st: core_metrics(s, sel & p)
            for st, sel in strata(s, obstype, cfg)
        }
        rh = rank_histogram(s, p)
        if rh:
            res['rank'][name] = rh
        ss = spread_skill_curve(s, p)
        if ss:
            res['spread_skill'][name] = ss
    return res
