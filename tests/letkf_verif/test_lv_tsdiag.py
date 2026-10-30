"""Tests for the T-S (water-mass) pairing and its reductions.

Synthetic ObsSet stand-ins only: the CI environment has no sample IODA files.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import lv_tsdiag as TS


def _set(meta, y, ombg, oman):
    return SimpleNamespace(n=len(y), meta=meta, y=np.asarray(y, 'f8'),
                           ombg=np.asarray(ombg, 'f8'),
                           oman=np.asarray(oman, 'f8'))


def _meta(depth, station=1):
    n = len(depth)
    return {'dateTime': np.full(n, 1.7e9), 'stationID': np.full(n, station),
            'latitude': np.full(n, 30.0), 'longitude': np.full(n, -40.0),
            'depth': np.asarray(depth, 'f8')}


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------

def test_pairs_matches_temperature_with_salinity_per_platform():
    got = TS.pairs(['insitu_temp_profile_argo', 'insitu_salt_profile_argo',
                    'insitu_temp_profile_xbt',          # no salinity partner
                    'insitu_salt_profile_glider', 'insitu_temp_profile_glider',
                    'sst_viirs_n20_l3u'])
    assert got == {
        'argo': ('insitu_temp_profile_argo', 'insitu_salt_profile_argo'),
        'glider': ('insitu_temp_profile_glider',
                   'insitu_salt_profile_glider')}


def test_fig_name_round_trips_through_the_report_slugs():
    assert (TS.fig_name('density', 'argo', 'e_pacific', TS.depth_slug(0, 500),
                        'all')
            == 'ts_density_argo_e_pacific_d0-500m_all.png')


def test_depth_ranges_default_and_override():
    assert TS.depth_ranges({}) == [[0.0, 6000.0], [0.0, 500.0],
                                   [500.0, 2000.0]]
    assert TS.depth_ranges({'ts_diagram': {'depth_ranges': [[0, 100]]}}) == \
        [[0.0, 100.0]]


# --------------------------------------------------------------------------
# seawater
# --------------------------------------------------------------------------

@pytest.mark.parametrize('s,t,rho', [
    (0.0, 5.0, 999.96675),       # UNESCO (1983) check values, p = 0
    (35.0, 5.0, 1027.67547),
    (35.0, 25.0, 1023.34306),
])
def test_sigma0_matches_the_unesco_check_values(s, t, rho):
    assert TS.sigma0(t, s) == pytest.approx(rho - 1000.0, abs=1e-4)


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------

def test_compute_pairs_only_levels_present_and_passing_in_both():
    # T at 4 depths, S at 3 of them (no 30 m) plus one T does not have
    tm, sm = _meta([10, 20, 30, 40]), _meta([10, 20, 40, 50])
    t_obs, s_obs = [20.0, 18.0, 15.0, 10.0], [35.0, 35.1, 35.2, 35.3]
    names = ('a', 'b')
    tsets = {n: _set(tm, t_obs, [0.1] * 4, [0.01] * 4) for n in names}
    ssets = {n: _set(sm, s_obs, [-0.02] * 4, [0.0] * 4) for n in names}
    tpass = np.array([True, True, True, True])
    spass = np.array([True, False, True, True])     # 20 m salinity rejected
    out = TS.compute('argo', TS.extract(tsets, tpass),
                     TS.extract(ssets, spass))
    depth = out['tsdiag/argo/obs/depth']
    assert sorted(depth.tolist()) == [10.0, 40.0]
    k = int(np.argmax(depth))
    assert out['tsdiag/argo/obs/salt'][k] == pytest.approx(35.2)
    # model = obs - departure, on the same pressure, so theta(model) is
    # within the adiabatic term of t - 0.1
    assert out['tsdiag/argo/a/salt_b'][k] == pytest.approx(35.22)
    assert (out['tsdiag/argo/obs/theta'][k] - out['tsdiag/argo/a/theta_b'][k]
            == pytest.approx(0.1, abs=1e-3))
    assert set(out) >= {'tsdiag/argo/b/theta_a', 'tsdiag/argo/b/salt_a'}


def test_compute_keeps_the_sample_common_across_experiments():
    m = _meta([10, 20])
    tsets = {'a': _set(m, [20, 18], [0.1, 0.1], [0, 0]),
             'b': _set(m, [20, 18], [0.1, np.nan], [0, 0])}
    ssets = {n: _set(m, [35, 35], [0, 0], [0, 0]) for n in 'ab'}
    both = np.ones(2, bool)
    out = TS.compute('argo', TS.extract(tsets, both), TS.extract(ssets, both))
    assert out['tsdiag/argo/obs/depth'].tolist() == [10.0]


def test_compute_with_a_missing_partner_is_empty():
    m = _meta([10])
    part = TS.extract({'a': _set(m, [20], [0], [0])}, np.ones(1, bool))
    assert TS.compute('argo', part, None) == {}


def test_potential_temperature_is_in_situ_at_the_surface():
    m = _meta([0.0])
    t = TS.extract({'a': _set(m, [12.3], [0], [0])}, np.ones(1, bool))
    s = TS.extract({'a': _set(m, [35.0], [0], [0])}, np.ones(1, bool))
    out = TS.compute('argo', t, s)
    assert out['tsdiag/argo/obs/theta'][0] == pytest.approx(12.3, abs=1e-5)


# --------------------------------------------------------------------------
# reductions
# --------------------------------------------------------------------------

def _arrays(theta, salt, depth, dtheta, dsalt):
    n = len(theta)
    a = {'tsdiag/argo/obs/latitude': np.full(n, 10.0, 'f4'),
         'tsdiag/argo/obs/longitude': np.full(n, 0.0, 'f4'),
         'tsdiag/argo/obs/depth': np.asarray(depth, 'f4'),
         'tsdiag/argo/obs/theta': np.asarray(theta, 'f4'),
         'tsdiag/argo/obs/salt': np.asarray(salt, 'f4'),
         'tsdiag/version': np.array(TS.VERSION)}
    for side in 'ba':
        a['tsdiag/argo/x/theta_%s' % side] = (np.asarray(theta, 'f4')
                                              + dtheta)
        a['tsdiag/argo/x/salt_%s' % side] = np.asarray(salt, 'f4') + dsalt
    return a


def test_reduce_and_accumulate_sum_over_cycles():
    n = 50
    arr = _arrays(np.full(n, 10.0), np.full(n, 35.0), np.full(n, 800.0),
                  0.5, -0.1)
    ranges = [[0, 6000], [0, 500]]
    total = {}
    for _ in range(2):
        TS.accumulate(total, TS.reduce_cycle(arr, {}, ranges))
    # nothing in 0-500 m, so no key for it
    assert set(total) == {('argo', 'global', 'd0-6000m')}
    acc = total[('argo', 'global', 'd0-6000m')]
    assert acc['obs/hist'].sum() == 2 * n
    assert acc['x/hist_b'].sum() == 2 * n
    assert acc['n'].sum() == 2 * n
    groups = TS.merge_classes(acc['n'], 1)
    assert len(groups) == 1
    n, mean, sd = TS.class_stats(acc, 'x', 'theta', 'b', groups)
    assert n[0] == 2 * n_levels(arr)
    assert mean[0] == pytest.approx(0.5, abs=1e-5)
    assert sd[0] == pytest.approx(0.0, abs=1e-3)
    _n, smean, _sd = TS.class_stats(acc, 'x', 'salt', 'a', groups)
    assert smean[0] == pytest.approx(-0.1, abs=1e-5)


def n_levels(arr):
    return arr['tsdiag/argo/obs/theta'].size


def test_merge_classes_pools_until_each_class_is_full():
    n = np.array([0, 0, 3, 1, 0, 6, 2, 2, 1, 0, 0])
    # 3+1+0+6 = 10, 2+2+1 = 5 is short and joins the last full class
    assert TS.merge_classes(n, 10) == [(2, 9)]
    # an empty class does not open a new one
    assert TS.merge_classes(n, 4) == [(2, 4), (5, 6), (6, 9)]
    # nothing reaches the threshold: no class at all
    assert TS.merge_classes(n, 100) == []


def test_class_stats_pools_the_sums_of_merged_classes():
    acc = {'n': np.array([2, 2]),
           'x/dtheta_b': np.array([2.0, 6.0]),         # means 1 and 3
           'x/dtheta2_b': np.array([2.0, 18.0])}
    n, mean, sd = TS.class_stats(acc, 'x', 'theta', 'b', [(0, 2)])
    assert n.tolist() == [4]
    assert mean[0] == pytest.approx(2.0)
    assert sd[0] == pytest.approx(1.0)
    _n, m, _s = TS.class_stats(acc, 'y', 'theta', 'b', [(0, 2)])
    assert np.isnan(m[0])


def test_regions_follow_the_configured_boxes():
    arr = _arrays([10.0] * 3, [35.0] * 3, [800.0] * 3, 0.0, 0.0)
    cfg = {'regions': [{'name': 'Tropics', 'lat0': -20, 'lat1': 20},
                       {'name': 'Arctic', 'lat0': 66, 'lat1': 90}]}
    keys = set(TS.reduce_cycle(arr, cfg, [[0, 6000]]))
    assert ('argo', 'Tropics', 'd0-6000m') in keys
    assert ('argo', 'Arctic', 'd0-6000m') not in keys


def test_hist2d_drops_points_off_the_grid():
    h = TS.hist2d(np.array([10.0, 99.0]), np.array([35.0, 35.0]))
    assert h.sum() == 1
    assert h.shape == (TS.THETA_EDGES.size - 1, TS.SALT_EDGES.size - 1)
