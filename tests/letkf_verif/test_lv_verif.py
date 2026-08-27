"""Tests for the gridded-analysis verification helpers.

Deliberately limited to what runs under the CI environment, which installs only
requirements.txt: no cartopy, no scipy, and no sample NetCDF. That covers the
config handling and the sampler, which is where the fiddly logic lives.
"""

import numpy as np
import pytest

import lv_verif as V


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def test_configured_accepts_a_bare_path():
    """The common form: a product name mapped to the directory holding it."""
    got = V.configured({'verification': {'products': {'sst': '/data/ostia'}}})
    assert got['sst']['path'] == '/data/ostia'
    # falls back to the built-in glob for that product
    assert got['sst']['pattern'] == V.PRODUCTS['sst']['pattern']


def test_configured_accepts_a_pattern_override():
    """For an archive that names its files differently."""
    got = V.configured({'verification': {'products': {
        'sst': {'path': '/data/ostia', 'pattern': '{Y}/{m}/{Ymd}-custom.nc'}}}})
    assert got['sst']['pattern'] == '{Y}/{m}/{Ymd}-custom.nc'


@pytest.mark.parametrize('cfg', [
    {},                                              # no section at all
    {'verification': {}},                            # section, no products
    {'verification': {'products': {}}},              # products, none named
    {'verification': {'products': {'sst': ''}}},     # named, empty path
    {'verification': {'products': {'sst': None}}},   # named, no value
    {'verification': {'products': {'nope': '/x'}}},  # not a known product
])
def test_configured_degrades_rather_than_raising(cfg):
    """A half-written config must switch the section off, not crash the run."""
    assert V.configured(cfg) == {}


# --------------------------------------------------------------------------
# grid spacing
# --------------------------------------------------------------------------

def test_spacing_survives_float32_axes():
    """OSTIA's axes are float32 and are NOT uniform to an equality test.

    They drift up to ~1.2e-5 degrees from a perfect 0.05 degree grid, which is
    about 1.4 m at the equator against a 5.5 km cell. Deriving the spacing from
    the endpoints is exact for a uniform axis and immune to that rounding;
    a[1] - a[0] is not.
    """
    axis = np.linspace(-89.975, 89.975, 3600).astype('f4').astype('f8')
    assert not np.allclose(np.diff(axis), axis[1] - axis[0], atol=0, rtol=0)
    assert V._spacing(axis) == pytest.approx(0.05, abs=1e-9)


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------

class _StubGrid:
    """Just what sample_to_grid touches: lat, lon180 and shape."""

    def __init__(self, lat, lon180):
        self.lat = np.asarray(lat, dtype='f8')
        self.lon180 = np.asarray(lon180, dtype='f8')
        self.shape = self.lat.shape


def _product(nlat=181, nlon=360):
    """A 1-degree product whose value encodes its own (row, column)."""
    lat = np.linspace(-90.0, 90.0, nlat)
    lon = np.linspace(-180.0, 179.0, nlon)
    arr = (np.arange(nlat)[:, None] * 1000.0 + np.arange(nlon)[None, :])
    return arr, lat, lon


def test_sample_picks_the_containing_cell():
    arr, lat, lon = _product()
    grid = _StubGrid([[0.0, 45.0]], [[0.0, 90.0]])
    got = V.sample_to_grid(grid, arr, lat, lon, half=0)
    # lat 0 -> row 90, lon 0 -> column 180; lat 45 -> row 135, lon 90 -> 270
    assert got[0, 0] == 90 * 1000.0 + 180
    assert got[0, 1] == 135 * 1000.0 + 270


def test_sample_wraps_longitude_rather_than_clamping():
    """Past the last cell centre, longitude wraps to the first cell.

    The product's cells are centred at whole degrees from -180 to 179, so the
    cell holding the antimeridian is column 0: it spans 179.5 to 180.5, which
    is the same as -180.5 to -179.5. A query at 179.9 therefore belongs in
    column 0. Clamping the index instead would land it in column 359, a whole
    cell away, and would do so silently.
    """
    arr, lat, lon = _product()
    grid = _StubGrid([[0.0, 0.0, 0.0]], [[179.9, -179.9, 178.9]])
    got = V.sample_to_grid(grid, arr, lat, lon, half=0)
    east, west, inside = got[0, 0], got[0, 1], got[0, 2]
    assert east == 90 * 1000.0 + 0, 'east of the last centre must wrap, not clamp'
    assert west == 90 * 1000.0 + 0, 'its neighbour across the seam is the same cell'
    # and the mapping is not simply collapsing everything to column 0
    assert inside == 90 * 1000.0 + 359


def test_sample_returns_nan_outside_the_product_latitudes():
    """Beyond the product's range the answer is 'no data', not the edge row.

    Clamping would fabricate polar values out of the highest latitude the
    product happens to reach.
    """
    arr, lat, lon = _product()
    lat = lat[(lat >= -60.0) & (lat <= 60.0)]
    arr = arr[:lat.size]
    grid = _StubGrid([[0.0, 80.0, -80.0]], [[0.0, 0.0, 0.0]])
    got = V.sample_to_grid(grid, arr, lat, lon, half=0)
    assert np.isfinite(got[0, 0])
    assert np.isnan(got[0, 1])
    assert np.isnan(got[0, 2])


def test_block_mean_averages_the_window_and_ignores_gaps():
    """Averaging a finer product onto a coarser cell must skip missing cells.

    Treating a NaN as zero would drag the mean toward zero next to any gap --
    a coastline, or the edge of the product's coverage.
    """
    arr, lat, lon = _product()
    arr[90, 179] = np.nan                     # one hole inside the window
    grid = _StubGrid([[0.0]], [[0.0]])
    got = V.sample_to_grid(grid, arr, lat, lon, half=1)
    window = arr[89:92, 179:182]
    assert got[0, 0] == pytest.approx(np.nanmean(window))
    assert np.isfinite(got[0, 0])


# --------------------------------------------------------------------------
# weighted statistics
# --------------------------------------------------------------------------

class _WeightedGrid:
    """Enough of Grid for wpercentile: the weights and the mask."""

    from lv_common import Grid as _G
    wpercentile = _G.wpercentile

    def __init__(self, wgt):
        self.wgt = np.asarray(wgt, dtype='f8')
        self.shape = self.wgt.shape


def test_wpercentile_is_area_weighted():
    """Weighting must actually change the answer, in the expected direction.

    bias and rms are area-weighted; a percentile counting cells equally is not
    comparable with them. Here nine cells hold a large value at tiny weight and
    one holds a small value at large weight -- unweighted the 90th percentile
    sits among the large values, weighted it must not.
    """
    vals = np.array([[0.0] + [10.0] * 9])
    wgt = np.array([[100.0] + [1.0] * 9])
    g = _WeightedGrid(wgt)
    assert g.wpercentile(vals, 90) < 10.0
    assert np.percentile(vals, 90) == pytest.approx(10.0)


def test_wpercentile_ignores_unselected_and_nonfinite():
    vals = np.array([[1.0, 2.0, 3.0, np.nan]])
    g = _WeightedGrid(np.ones((1, 4)))
    sel = np.array([[True, True, False, True]])
    # only 1 and 2 contribute; 3 is deselected and the NaN is dropped
    assert 1.0 <= g.wpercentile(vals, 50, sel) <= 2.0
    assert np.isnan(g.wpercentile(vals, 50, np.zeros((1, 4), dtype=bool)))


# --------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------

def test_in_region_bands_and_boxes():
    from lv_common import in_region
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([-100.0, 20.0, 160.0])
    band = {'name': 'tropics', 'lat0': -30, 'lat1': 30}
    assert list(in_region(band, lat, lon)) == [False, True, False]
    box = {'name': 'b', 'lat0': -10, 'lat1': 10, 'lon0': 0, 'lon1': 90}
    assert list(in_region(box, lat, lon)) == [False, True, False]
    # a longitude-bounded region degrades to its latitude band when the
    # observations carry no longitude, rather than selecting nothing
    assert list(in_region(box, lat)) == [False, True, False]


def test_in_region_straddles_the_dateline():
    from lv_common import in_region
    lat = np.zeros(3)
    lon = np.array([170.0, 0.0, -170.0])
    box = {'name': 'pac', 'lat0': -10, 'lat1': 10, 'lon0': 140, 'lon1': -140}
    assert list(in_region(box, lat, lon)) == [True, False, True]


def test_retired_region_keys_are_refused():
    """Silently ignoring a region the user asked for is worse than stopping."""
    from lv_common import check_retired
    for key in ('lat_bands', 'corr_boxes'):
        with pytest.raises(SystemExit) as e:
            check_retired({key: [{'name': 'x'}]})
        assert 'regions' in str(e.value)
    check_retired({'regions': [{'name': 'x'}]})      # the new key is fine


def test_corr_regions_must_name_known_regions():
    from lv_common import corr_regions
    cfg = {'regions': [{'name': 'Kuroshio', 'lat0': 25, 'lat1': 45}],
           'corr_regions': ['Kuroshio']}
    assert [r['name'] for r in corr_regions(cfg)] == ['Kuroshio']
    with pytest.raises(SystemExit):
        corr_regions({'regions': [], 'corr_regions': ['Atlantis']})
