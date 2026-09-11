"""Shared plumbing for the marine LETKF verification suite.

Handles the things every downstream script needs: reading the experiment
registry, loading the MOM6 grid and a depth axis, opening IODA obs diag files
(tarred or loose), and joining observations across N experiments that do not
share a row ordering -- or even a sample, once thinning is in play.

Run ``python3 lv_common.py --selftest`` to check the readers against the
identities the JEDI output is supposed to satisfy.
"""

import glob
import os
import shutil
import sys
import tarfile
from datetime import datetime, timedelta
from functools import reduce

import numpy as np
import yaml
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))

# These scripts are normally run with output redirected to a log file or piped
# through tee, where python block-buffers stdout in 8 KB chunks -- so a long
# run prints nothing for minutes and then everything at once, looking hung.
# Line buffering makes progress appear as it happens. Every module imports
# lv_common, so doing it here covers the whole suite.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):      # not a regular stream; nothing to do
    pass

# Mirrors ufo/src/ufo/filters/QCflags.h. Only the codes we can actually see in
# marine diags are worth naming; anything else falls back to "qc<N>".
QC_NAMES = {
    0: 'pass', 1: 'passive', 10: 'missing', 11: 'preQC', 12: 'bounds',
    13: 'domain', 14: 'black', 15: 'Hfailed', 16: 'thinned', 17: 'diffref',
    18: 'clw', 19: 'fguess', 20: 'seaice', 21: 'track', 22: 'buddy',
    23: 'derivative', 24: 'profile', 25: 'onedvar', 26: 'bayesianQC',
    27: 'modelobthresh', 28: 'history', 29: 'processed',
    30: 'superrefraction', 31: 'superob', 33: 'percentile',
}


def qc_name(code):
    return QC_NAMES.get(int(code), 'qc%d' % code)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def expand(p):
    """Expand ~ and $VARS so a config can be written machine-independently."""
    return os.path.expanduser(os.path.expandvars(str(p)))


CYCLE_FMT = '%Y%m%d%H'


def parse_cycle(c):
    """Strict YYYYMMDDHH -> datetime.

    strptime('%Y%m%d%H') is lenient about field widths, so an 8-digit date
    silently parses as a completely different time rather than failing. Every
    cycle string goes through this instead.
    """
    s = str(c).strip()
    if len(s) != 10 or not s.isdigit():
        raise ValueError('cycle %r is not a 10-digit YYYYMMDDHH string' % c)
    return datetime.strptime(s, CYCLE_FMT)


def cycle_range(start, stop, step_hours=6):
    """Inclusive list of YYYYMMDDHH cycle strings.

    A week of 6-hourly cycles is 29 strings nobody should have to type, so the
    config accepts {start, stop, step} as well as an explicit list.
    """
    t = parse_cycle(start)
    end = parse_cycle(stop)
    if end < t:
        raise ValueError('cycles: stop %s is before start %s' % (stop, start))
    if step_hours <= 0:
        raise ValueError('cycles: step must be positive, got %r' % step_hours)
    step = timedelta(hours=float(step_hours))
    out = []
    while t <= end:
        out.append(t.strftime(CYCLE_FMT))
        t += step
    return out


def resolve_cycles(spec):
    """Accept either a list of cycles or a {start, stop, step} mapping."""
    if isinstance(spec, dict):
        missing = {'start', 'stop'} - set(spec)
        if missing:
            raise ValueError('cycles: mapping needs %s'
                             % ', '.join(sorted(missing)))
        return cycle_range(spec['start'], spec['stop'], spec.get('step', 6))
    out = []
    for c in spec:
        parse_cycle(c)                      # validate, discard the datetime
        out.append(str(c).strip())
    return out


# The two experiment kinds store their background under different conventions,
# verified against the files: the 3DVar first guess is an f006 forecast sitting
# in the PREVIOUS cycle's directory, while the LETKF prior sits in its own
# cycle's directory. Applying one offset to both silently compares states 6 h
# apart, so each kind carries its own default.
BKG_DEFAULTS = {
    'var': dict(stem='gdas.{Ymd}/{HH}/model/{realm}/history',
                pattern='*inst.f006.nc', offset_hours=-6),
    'letkf': dict(stem=None,               # None -> the experiment's own stem
                  pattern='*ensmean_prior.nc', offset_hours=0),
}

# CICE history carries grid-cell-mean thickness; the analysis files carry
# thickness per unit ice area. Map the former onto the latter so every
# downstream consumer sees one set of names.
#   canonical name -> (source var, divide-by var or None)
BKG_VARMAP = {
    'var': {'aice_h': ('aice_h', None),
            'hi_div_aice_h': ('hi_h', 'aice_h'),
            'hs_div_aice_h': ('hs_h', 'aice_h')},
    'letkf': {},                            # already canonical
}

# The three stages of the ensemble spread. Note the post-inflation file does
# NOT follow the ensvar_<stage> naming, hence a table rather than a format
# string; a run that names it differently sets `ensvar_patterns:` in its
# experiment entry.
#   prior -> sigma_b            background
#   post  -> sigma_a            analysis, before inflation
#   an    -> sigma_an           analysis, after inflation
ENSVAR_PATTERNS = {
    'prior': '*ensvar_prior.nc',
    'post': '*ensvar_post.nc',
    'an': '*an_ensvar.nc',
}


class Experiment:
    """One entry in experiments.yaml."""

    def __init__(self, d, cfg_dir, root_override=None):
        self.name = d['name']
        self.kind = d.get('kind', 'var')          # 'var' or 'letkf'
        self.label = d.get('label', self.name)
        root = expand(root_override or d.get('root', cfg_dir))
        self.root = root if os.path.isabs(root) else os.path.join(cfg_dir, root)
        self.stem = d['stem']
        if self.kind not in ('var', 'letkf'):
            raise ValueError('%s: kind must be var or letkf' % self.name)
        bkg = dict(BKG_DEFAULTS[self.kind])
        bkg.update(d.get('background') or {})
        self.bkg = bkg
        self.varmap = BKG_VARMAP[self.kind]
        ev = dict(ENSVAR_PATTERNS)
        ev.update(d.get('ensvar_patterns') or {})
        self.ensvar_patterns = ev
        # Older archives name the analysis increment differently (e.g.
        # ocn.incr.nc / ice.incr.nc rather than jedi_increment*.nc); override
        # per experiment rather than hardcoding every convention ever used.
        self.incr_pattern = d.get('increment_pattern')
        # The verification suite normally reconstructs an analysis as the
        # background plus its increment.  A few diagnostics, including the
        # frontal-current maps, need the written analysis state itself.  Keep
        # its archive-specific name in the registry rather than duplicating
        # path conventions in those diagnostics.
        self.analysis_pattern = d.get('analysis_pattern')

    def dir_for(self, cycle, stem=None, realm=''):
        """cycle is a 10-character YYYYMMDDHH string."""
        return os.path.join(self.root,
                            (stem or self.stem).format(
                                Ymd=cycle[:8], HH=cycle[8:10],
                                YYYYMMDDHH=cycle, realm=realm))

    def _glob1(self, cycle, pattern, required=True):
        hits = sorted(glob.glob(os.path.join(self.dir_for(cycle), pattern)))
        if not hits:
            if required:
                raise FileNotFoundError('%s / %s: no match for %s' %
                                        (self.name, cycle, pattern))
            return None
        return hits[0]

    # -- observation diagnostics -------------------------------------------
    def obs_files(self, cycle, workdir):
        """Return {obstype: path}, extracting the LETKF tarball if needed."""
        if self.kind == 'letkf':
            tar = self._glob1(cycle, 'ocean/*ioda_hofx*.tar.gz', required=False)
            if tar is None:
                tar = self._glob1(cycle, 'ocean/*ioda_hofx*.tar')
            dest = os.path.join(workdir, self.name, cycle)
            if not os.path.isdir(dest):
                os.makedirs(dest, exist_ok=True)
                with tarfile.open(tar) as tf:
                    tf.extractall(dest)
            paths = sorted(glob.glob(os.path.join(dest, '*.nc')))
        else:
            paths = sorted(glob.glob(
                os.path.join(self.dir_for(cycle), 'ocean/diags/*.nc')))
        return {os.path.basename(p)[:-3]: p for p in paths}

    # -- background / first guess -------------------------------------------
    def background_cycle(self, cycle):
        """The directory cycle holding the background valid at ``cycle``."""
        off = float(self.bkg.get('offset_hours', 0))
        return (parse_cycle(cycle) + timedelta(hours=off)).strftime(CYCLE_FMT)

    def background(self, cycle, realm):
        """Path to the background valid at ``cycle`` for ``realm``, or None.

        No time check is applied: the offset and filename pattern are taken as
        given, per the configured convention.
        """
        bc = self.background_cycle(cycle)
        stem = self.bkg.get('stem')
        if stem:                       # stem carries {realm} itself (3DVar)
            d = self.dir_for(bc, stem, realm)
        else:                          # fall back to the analysis stem (LETKF)
            d = os.path.join(self.dir_for(bc), realm)
        hits = sorted(glob.glob(os.path.join(d, self.bkg['pattern'])))
        return hits[0] if hits else None

    # -- state space --------------------------------------------------------
    def increment(self, cycle, realm):
        default = ('*ensmean_incr.nc' if self.kind == 'letkf'
                   else '*jedi_increment*.nc')
        pat = '%s/%s' % (realm, self.incr_pattern or default)
        return self._glob1(cycle, pat, required=False)

    def analysis(self, cycle, realm='ocean'):
        """Written analysis state for an optional archive-specific pattern.

        The pattern is relative to the realm directory.  Returning ``None``
        when it is not configured lets an optional diagnostic report a clear
        unavailable-cycle reason without affecting the standard suite.
        """
        if not self.analysis_pattern:
            return None
        return self._glob1(cycle, '%s/%s' % (realm, self.analysis_pattern),
                           required=False)

    def ensvar(self, cycle, realm, when):
        """Path to the 'prior' / 'post' / 'an' ensemble variance, or None."""
        pat = self.ensvar_patterns.get(when)
        if self.kind != 'letkf' or pat is None:
            return None
        return self._glob1(cycle, '%s/%s' % (realm, pat), required=False)


def load_config(path=None, root_override=None, outdir_override=None,
                cache_override=None):
    """Read experiments.yaml.

    ``root_override`` (or $LETKF_VERIF_ROOT) replaces every experiment's
    ``root``, which is how the same config is moved between machines.
    ``outdir_override`` (or $LETKF_VERIF_OUTDIR) relocates every output.
    ``cache_override`` prepends extra cache directories to read and merge.

    Every relative path in the config -- ``grid``, ``outdir``, an experiment
    ``root`` -- resolves against the directory holding the config file, so a
    checkout can sit anywhere.
    """
    # No default inside the code directory. Installed as a shared application
    # that directory is effectively read-only, and two runs started in parallel
    # must not contend over one implicit config.
    path = expand(path or os.environ.get('LETKF_VERIF_CONFIG') or '')
    example = os.path.join(HERE, 'experiments.example.yaml')
    if not path:
        raise SystemExit(
            'no config given: pass one as the first argument (or --config, or '
            '$LETKF_VERIF_CONFIG).\n  cp %s my_experiments.yaml\nthen edit it '
            'for your machine.' % example)
    if not os.path.exists(path):
        raise FileNotFoundError(
            'no config at %s\n  cp %s %s\nthen edit it for your machine.'
            % (path, example, path))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(path))
    root_override = root_override or os.environ.get('LETKF_VERIF_ROOT')

    def _abs(p):
        """Relative paths in the config mean "relative to the config file"."""
        p = expand(p)
        return p if os.path.isabs(p) else os.path.join(cfg_dir, p)

    cfg['grid'] = _abs(cfg['grid'])
    cfg['experiments'] = [Experiment(d, cfg_dir, root_override)
                          for d in cfg['experiments']]
    # Output locations. `outdir` (or --outdir / $LETKF_VERIF_OUTDIR) moves all
    # four together; any of them can still be pinned individually.
    #
    # The default is the CONFIG's directory, never the code's: output must not
    # land inside a shared application checkout, and this makes one rule cover
    # everything -- relative means relative to your config, and nothing is ever
    # relative to the code.
    outdir = expand(outdir_override or cfg.get('outdir')
                    or os.environ.get('LETKF_VERIF_OUTDIR') or cfg_dir)
    outdir = outdir if os.path.isabs(outdir) else os.path.join(cfg_dir, outdir)
    cfg['outdir'] = outdir
    # `cache` may be a single directory or a list. The first is where new
    # results are written; all of them are read and merged, which is how two
    # experiments verified in separate runs get compared without copying.
    src = cfg.get('cache') or os.path.join(outdir, 'cache')
    src = [src] if isinstance(src, str) else list(src)
    if cache_override:
        src = list(cache_override) + [p for p in src if p not in cache_override]
    cfg['caches'] = [_abs(p) for p in src]
    cfg['cache'] = cfg['caches'][0]
    cfg['figs'] = _abs(cfg.get('figs') or os.path.join(outdir, 'figs'))
    cfg['report'] = _abs(cfg.get('report')
                         or os.path.join(outdir, 'letkf_verification.html'))
    cfg['scorecard'] = _abs(cfg.get('scorecard')
                            or os.path.join(outdir, 'scorecard.md'))
    # Gridded verification products, if configured. Each names its own
    # directory -- the products come from unrelated archives -- given either as
    # a bare path or as a mapping with `path` and an optional `pattern`.
    if cfg.get('verification'):
        v = dict(cfg['verification'])
        prods = {}
        for name, entry in (v.get('products') or {}).items():
            if isinstance(entry, str):
                entry = {'path': entry}
            entry = dict(entry or {})
            if entry.get('path'):
                entry['path'] = _abs(entry['path'])
            prods[name] = entry
        v['products'] = prods
        cfg['verification'] = v
    # WOA23 climatology directory. Same rule as `verification:` above, and
    # deliberately NOT nested inside it -- every consumer of that block assumes
    # a same-day L4 product with an lv_verif.PRODUCTS spec, which a climatology
    # is not.
    if cfg.get('woa'):
        w = cfg['woa']
        w = {'path': w} if isinstance(w, str) else dict(w or {})
        if w.get('path'):
            w['path'] = _abs(w['path'])
        cfg['woa'] = w
    names = [e.name for e in cfg['experiments']]
    if len(set(names)) != len(names):
        raise ValueError('duplicate experiment names: %s' % names)
    if cfg.get('reference') and cfg['reference'] not in names:
        raise ValueError('reference %r not among %s' % (cfg['reference'], names))
    check_retired(cfg)
    corr_regions(cfg)          # validate the names now, not mid-run
    cfg['cycles'] = resolve_cycles(cfg['cycles'])
    return cfg


def region_box(r):
    """(lat0, lat1, lon0, lon1) for one `regions:` entry; lon may be None.

    One schema for every consumer. It used to be three -- `regions:` with
    lat0/lat1, `lat_bands:` with lo/hi, and `corr_boxes:` with lat0/lat1 plus
    longitude -- which listed the same bands twice under different key names
    and, worse, let one name mean two different boxes: `N_Atlantic` was
    30-60N/70-10W for the profiles and 30-50N/60-20W for the correlation
    lengths, so the report showed both under one heading.
    """
    lat0, lat1 = r.get('lat0', -90.0), r.get('lat1', 90.0)
    if 'lon0' in r:
        return lat0, lat1, r['lon0'], r['lon1']
    return lat0, lat1, None, None


def in_region(r, lat, lon=None):
    """Boolean mask for one region over flat lat/lon arrays.

    Shared by the grid masks and the observation strata so a region means the
    same thing in state space and in observation space. ``lon`` is optional:
    where it is absent a longitude-bounded region degrades to its latitude
    band rather than selecting nothing.
    """
    lat0, lat1, lon0, lon1 = region_box(r)
    sel = (lat >= lat0) & (lat < lat1)
    if lon0 is None or lon is None:
        return sel
    if lon0 <= lon1:
        return sel & (lon >= lon0) & (lon < lon1)
    return sel & ((lon >= lon0) | (lon < lon1))    # straddles the dateline


# --------------------------------------------------------------------------
# ocean-basin mask (RECCAP2 open_ocean), the real-geography alternative to a
# hand-drawn `regions:` box for the broad basin breakdown. `corr_regions:`
# keeps using boxes -- the correlation-length fit needs an actual box extent
# to fit within, which a basin mask cannot give it.
# --------------------------------------------------------------------------

_BASIN_MASK_CACHE = {}


def _load_basin_mask(path):
    """(lat, lon, open_ocean codes, {code: name}), cached by path.

    Only open_ocean is read: it is the file's whole-basin classification
    (Atlantic/Pacific/Indian/Arctic/Southern, 0 = not open ocean), separate
    from its `coastal_marcats` sub-regions this suite has no use for. Codes
    and names both come from the variable's own `region_names` attribute --
    "1.Atlantic, 2.Pacific, ..." -- rather than being hardcoded here, so a
    differently-coded mask file still reads correctly.
    """
    path = expand(path)
    if path not in _BASIN_MASK_CACHE:
        with Dataset(path) as ds:
            lat = np.asarray(ds['lat'][:], dtype='f8')
            lon = np.asarray(ds['lon'][:], dtype='f8')
            codes = np.asarray(ds['open_ocean'][:], dtype='i4')
            names = dict(kv.strip().split('.', 1)
                        for kv in ds['open_ocean'].region_names.split(','))
            names = {int(k): v for k, v in names.items()}
        _BASIN_MASK_CACHE[path] = (lat, lon, codes, names)
    return _BASIN_MASK_CACHE[path]


def basin_regions(path):
    """[(code, name), ...] in ascending code order, from the mask file."""
    return sorted(_load_basin_mask(path)[3].items())


def basin_at(path, lat, lon):
    """RECCAP2 open_ocean code at each (lat, lon), any matching shape.

    Nearest cell on the mask's own 1-degree grid -- exact for a uniform axis,
    same approach as lv_verif.sample_to_grid. ``lon`` is wrapped into the
    mask's 0..360 convention (cell centres 0.5..359.5) first, since callers
    pass either that or -180..180.
    """
    mlat, mlon, codes, _names = _load_basin_mask(path)
    dla = (mlat[-1] - mlat[0]) / (mlat.size - 1)
    dlo = (mlon[-1] - mlon[0]) / (mlon.size - 1)
    j = np.clip(np.rint((np.asarray(lat) - mlat[0]) / dla).astype(int),
               0, mlat.size - 1)
    i = np.mod(np.rint((np.mod(lon, 360.0) - mlon[0]) / dlo).astype(int),
              mlon.size)
    return codes[j, i]


def region_list(cfg):
    """Every region name in display order: basins first (if configured),
    then the explicit `regions:` boxes -- shared so the grid masks, the
    observation strata and figure ordering all agree on one list."""
    names = []
    mask_path = cfg.get('ocean_basin_mask')
    if mask_path:
        names += [name for _code, name in basin_regions(mask_path)]
    names += [r['name'] for r in cfg.get('regions', []) if r['name'] not in names]
    return names


def corr_regions(cfg):
    """Regions the correlation-length fit runs in, as full entries.

    Opt-in by name, because the fit is per-box and meaningless globally.
    """
    want = cfg.get('corr_regions') or []
    by_name = {r['name']: r for r in cfg.get('regions', [])}
    unknown = [n for n in want if n not in by_name]
    if unknown:
        raise SystemExit(
            'corr_regions names %s, which is not in regions:.\n'
            '  regions: has %s'
            % (', '.join(unknown), ', '.join(sorted(by_name)) or '(none)'))
    return [by_name[n] for n in want]


# Replaced by the single `regions:` list; refuse them rather than ignore them,
# because silently dropping a region the user asked for is worse than stopping.
_RETIRED_KEYS = {
    'lat_bands': ('regions:', 'the same entries, with lat0/lat1 instead of '
                              'lo/hi -- observations are now stratified by the '
                              'same regions as the state'),
    'corr_boxes': ('regions: plus corr_regions:',
                   'move the boxes into regions: and name them in '
                   'corr_regions: [..]'),
}


def check_retired(cfg):
    for key, (repl, how) in _RETIRED_KEYS.items():
        if cfg.get(key):
            raise SystemExit(
                '`%s:` has been replaced by `%s`.\n  %s' % (key, repl, how))


def select_experiments(cfg, names):
    """Restrict the registry to ``names``, for a per-experiment precompute.

    Everything downstream iterates cfg['experiments'] -- the observation join,
    the state loop, the presence check -- so filtering that one list is a
    complete restriction. A cache built this way honestly records that it
    joined only these experiments, which is what lets a later merge detect that
    its common sample is not a real cross-experiment one.
    """
    if not names:
        return cfg
    known = {e.name for e in cfg['experiments']}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit('unknown experiment(s) %s; the config has %s'
                         % (', '.join(unknown), ', '.join(sorted(known))))
    cfg = dict(cfg)
    cfg['experiments'] = [e for e in cfg['experiments'] if e.name in set(names)]
    return cfg


def experiment(cfg, name):
    for e in cfg['experiments']:
        if e.name == name:
            return e
    raise KeyError(name)


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

class Grid:
    """MOM6 tripolar grid plus a nominal depth axis.

    The increment and ensemble-variance files carry index axes only, so every
    area-weighted number and every map needs this.
    """

    @staticmethod
    def _f2d(ds, name):
        """Read a 2-D field whether or not it carries a leading Time axis."""
        v = ds[name]
        return np.asarray(v[0] if v.ndim == 3 else v[:])

    def __init__(self, gridspec):
        with Dataset(gridspec) as g:
            self.lon = self._f2d(g, 'lon')
            self.lat = self._f2d(g, 'lat')
            self.area = self._f2d(g, 'area')
            self.mask = self._f2d(g, 'mask2d') > 0
        self.shape = self.mask.shape
        self.wgt = np.where(self.mask, self.area, 0.0)
        # This gridspec runs -300..60 deg; box selection and plotting both want
        # a conventional [-180, 180) longitude regardless of what is on disk.
        self.lon180 = ((self.lon + 180.0) % 360.0) - 180.0

    def wmean(self, field, extra=None):
        """Area-weighted mean over wet points."""
        w = self.wgt if extra is None else np.where(extra, self.wgt, 0.0)
        f = np.asarray(field, dtype='f8')
        ok = np.isfinite(f) & (w > 0)
        if not ok.any():
            return np.nan
        return float(np.sum(f[ok] * w[ok]) / np.sum(w[ok]))

    def wrms(self, field, extra=None):
        m = self.wmean(np.asarray(field, dtype='f8') ** 2, extra)
        return float(np.sqrt(m)) if np.isfinite(m) else np.nan

    def wsd(self, field, extra=None):
        """Area-weighted standard deviation across the selected columns.

        How representative the regional mean is: a narrow spread means the
        region is behaving uniformly, a wide one means the mean is hiding
        structure.
        """
        m = self.wmean(field, extra)
        if not np.isfinite(m):
            return np.nan
        v = self.wmean((np.asarray(field, dtype='f8') - m) ** 2, extra)
        return float(np.sqrt(v)) if np.isfinite(v) else np.nan

    def wpercentile(self, field, q, extra=None):
        """Area-weighted percentile of ``field`` over the selected cells.

        The unweighted np.percentile is not comparable with wmean and wrms on
        this grid: cell areas span roughly a factor of 40 between the equator
        and the Arctic, so counting cells equally lets the small polar ones
        dominate. On the sample cycle's SSS differences the unweighted 90th
        percentile is 1.117 psu against 0.519 area-weighted -- a factor of two,
        in a number read alongside an area-weighted RMS.

        Uses the usual weighted-quantile convention, cumulative weight taken at
        the midpoint of each cell's interval. That is not numpy's type-7
        interpolation, so it does not reproduce np.percentile exactly even on
        equal weights (17.5 against 17.1 on 0..19); the two are different
        conventions, and what matters here is being on the same footing as
        wmean and wrms rather than agreeing with an unweighted function.
        """
        w = self.wgt if extra is None else np.where(extra, self.wgt, 0.0)
        f = np.asarray(field, dtype='f8')
        ok = np.isfinite(f) & (w > 0)
        if not ok.any():
            return np.nan
        v, wv = f[ok], w[ok]
        order = np.argsort(v)
        v, wv = v[order], wv[order]
        c = (np.cumsum(wv) - 0.5 * wv) / wv.sum()
        return float(np.interp(q / 100.0, c, v))

    def regions(self, cfg):
        """{name: bool mask} from ``ocean_basin_mask:`` and ``regions:``,
        always including global.

        Boxes are given in conventional longitude, so selection uses lon180
        rather than the gridspec's native -300..60 range.
        """
        out = {'global': self.mask.copy()}
        mask_path = cfg.get('ocean_basin_mask')
        if mask_path:
            codes = basin_at(mask_path, self.lat, self.lon180)
            for code, name in basin_regions(mask_path):
                out[name] = self.mask & (codes == code)
        for r in cfg.get('regions', []):
            name = r['name']
            if name == 'global' or name in out:
                continue
            out[name] = self.mask & in_region(r, self.lat, self.lon180)
        return out


# --------------------------------------------------------------------------
# observation diagnostics
# --------------------------------------------------------------------------

_META = ('dateTime', 'latitude', 'longitude', 'depth', 'stationID',
         'oceanBasin')

# Integer scaling used to build exact join keys. Float equality across files
# written by different runs is not safe; scaled integers are.
_KEY_SCALE = {'dateTime': 1, 'stationID': 1, 'latitude': 10 ** 4,
              'longitude': 10 ** 4, 'depth': 10 ** 3}
_KEY_ORDER = ('dateTime', 'stationID', 'latitude', 'longitude', 'depth',
              'obsvalue')
_MISSING_KEY = np.int64(-(2 ** 62))


def _get(ds, group, var):
    if group not in ds.groups or var not in ds[group].variables:
        return None
    return np.ma.filled(ds[group][var][:].astype('f8'), np.nan)


def _geti(ds, group, var, fill=-1):
    if group not in ds.groups or var not in ds[group].variables:
        return None
    return np.ma.filled(ds[group][var][:], fill).astype('i8')


def _scaled(x, scale):
    y = np.asarray(x, dtype='f8')
    out = np.full(y.shape, _MISSING_KEY, dtype='i8')
    ok = np.isfinite(y)
    out[ok] = np.rint(y[ok] * scale).astype('i8')
    return out


def _occurrence(sorted_cols):
    """Index of each row within its run of identical rows (input sorted)."""
    n = sorted_cols.shape[0]
    if n == 0:
        return np.zeros(0, dtype='i8')
    new = np.ones(n, dtype=bool)
    new[1:] = np.any(sorted_cols[1:] != sorted_cols[:-1], axis=1)
    start = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    return (np.arange(n) - start).astype('i8')


def _make_key(cols):
    """Pack integer columns into a unique, hashable per-row void key.

    Genuine duplicate observations exist (Argo has ~25k rows whose full
    metadata *and* value coincide). Appending an occurrence counter within each
    duplicate run makes every row distinct, so the N-way intersection below
    keeps duplicates instead of collapsing them.
    """
    M = np.stack(cols, axis=1)
    order = np.lexsort(M.T[::-1])
    occ = np.empty(M.shape[0], dtype='i8')
    occ[order] = _occurrence(M[order])
    full = np.ascontiguousarray(np.concatenate([M, occ[:, None]], axis=1))
    return full.view(np.dtype((np.void, full.shape[1] * 8))).ravel()


class ObsSet:
    """Per-observation arrays for one obs type in one experiment.

    Ensemble members are reduced to spread / rank / CRPS on read and then
    dropped -- keeping 30 x N members alive for every obs type at once is not
    worth the memory.
    """

    def __init__(self, path, name):
        self.path = path
        self.name = name
        with Dataset(path) as ds:
            self.var = sorted(ds['ObsValue'].variables)[0]
            v = self.var
            self.n = ds.dimensions['Location'].size
            self.meta = {m: np.ma.filled(ds['MetaData'][m][:].astype('f8'),
                                         np.nan)
                         for m in _META if m in ds['MetaData'].variables}
            self.y = _get(ds, 'ObsValue', v)
            self.ombg = _get(ds, 'ombg', v)
            self.oman = _get(ds, 'oman', v)
            self.R = _get(ds, 'ObsError', v)
            self.Reff = _get(ds, 'EffectiveError1', v)
            if self.Reff is None:
                self.Reff = _get(ds, 'EffectiveError0', v)
            self.qc = _geti(ds, 'EffectiveQC1', v)
            if self.qc is None:
                self.qc = _geti(ds, 'EffectiveQC0', v)
            self.preqc = _geti(ds, 'PreQC', v)

            mem0 = sorted(g for g in ds.groups if g.startswith('hofx0_'))
            mem1 = sorted(g for g in ds.groups if g.startswith('hofx1_'))
            self.nens = len(mem0)
            if self.nens:
                Xb = np.stack([_get(ds, g, v) for g in mem0])
                self.spread_b = Xb.std(axis=0, ddof=1)
                self.rank = self._rank(Xb, self.y)
                self.crps = self._crps(Xb, self.y)
                del Xb
                if mem1:
                    Xa = np.stack([_get(ds, g, v) for g in mem1])
                    self.spread_a = Xa.std(axis=0, ddof=1)
                    del Xa
                else:
                    self.spread_a = None
            else:
                self.spread_b = self.spread_a = self.rank = self.crps = None
        self.key = self._build_key()

    def _build_key(self):
        cols = []
        for m in _KEY_ORDER:
            if m == 'obsvalue':
                cols.append(_scaled(self.y, 10 ** 6))
            elif m in self.meta:
                cols.append(_scaled(self.meta[m], _KEY_SCALE[m]))
        return _make_key(cols)

    @staticmethod
    def _rank(X, y):
        """Rank of the observation among the members, 0..nens."""
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=0)
        r = np.full(y.shape, -1, dtype='i4')
        r[ok] = (X[:, ok] < y[ok]).sum(axis=0)
        return r

    @staticmethod
    def _crps(X, y):
        """Fair (unbiased) CRPS of the ensemble against y, per observation.

        CRPS = mean|x_i - y| - 1/(2 m (m-1)) * sum_ij |x_i - x_j|, the m/(m-1)
        correction removing the finite-ensemble low bias.
        """
        m = X.shape[0]
        out = np.full(y.shape, np.nan)
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=0)
        if not ok.any():
            return out
        Xs = np.sort(X[:, ok], axis=0)
        term1 = np.abs(Xs - y[ok]).mean(axis=0)
        # sum_ij |x_i - x_j| = 2 * sum_i (2i - m + 1) * x_(i)  for sorted x
        coef = (2 * np.arange(m) - m + 1).reshape(-1, 1)
        pair = 2.0 * (coef * Xs).sum(axis=0)
        out[ok] = term1 - pair / (2.0 * m * (m - 1))
        return out

    def passed(self):
        return self.qc == 0

    def qc_counts(self):
        if self.qc is None:
            return {}
        vals, cnt = np.unique(self.qc, return_counts=True)
        return {qc_name(v): int(c) for v, c in zip(vals, cnt)}

    def take(self, idx):
        """Reindexed shallow copy (used after joining)."""
        out = ObsSet.__new__(ObsSet)
        out.path, out.name, out.var = self.path, self.name, self.var
        out.nens, out.n = self.nens, len(idx)
        out.meta = {k: v[idx] for k, v in self.meta.items()}
        for a in ('y', 'ombg', 'oman', 'R', 'Reff', 'qc', 'preqc',
                  'spread_b', 'spread_a', 'rank', 'crps', 'key'):
            v = getattr(self, a)
            setattr(out, a, None if v is None else v[idx])
        return out


def join_obs(sets):
    """Align N ObsSets of the same obs type onto their common observations.

    ``sets`` maps experiment name -> ObsSet. Returns (aligned, counts) where
    ``aligned`` maps name -> reindexed ObsSet, all of equal length and in the
    same order, and ``counts`` records what the intersection cost.
    """
    common = reduce(np.intersect1d, [s.key for s in sets.values()])
    aligned, counts = {}, {}
    for name, s in sets.items():
        order = np.argsort(s.key)
        pos = np.searchsorted(s.key, common, sorter=order)
        idx = order[pos]
        aligned[name] = s.take(idx)
        counts[name] = {
            'n_own': int(s.n),
            'n_pass_own': int(np.count_nonzero(s.passed())),
            'n_matched': int(len(common)),
        }
    if aligned:
        both = np.ones(len(common), dtype=bool)
        for a in aligned.values():
            both &= a.passed()
        for name in counts:
            counts[name]['n_common_pass'] = int(np.count_nonzero(both))
        return aligned, counts, both
    return aligned, counts, np.zeros(0, dtype=bool)


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------

def rms(x):
    x = np.asarray(x, dtype='f8')
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan


def mean(x):
    x = np.asarray(x, dtype='f8')
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def signed_sqrt(x):
    """sqrt that survives the negative expectations Desroziers can produce."""
    return float(np.sign(x) * np.sqrt(abs(x))) if np.isfinite(x) else np.nan


def open_workdir(cfg):
    """Scratch space for the extracted LETKF hofx tarballs.

    This used to be a fresh mkdtemp per run, which leaked ~1 GB of extracted
    observations every time any script was invoked and filled /tmp. It now
    defaults to a stable directory beside the cache, so extraction is reused
    rather than repeated, and compute_cycle releases each cycle when it is
    done with it.
    """
    d = expand(cfg.get('workdir') or os.path.join(cfg['outdir'], 'work'))
    os.makedirs(d, exist_ok=True)
    return d


def release_workdir(work, cycle, names):
    """Drop the extracted observations for one cycle once it is cached."""
    freed = 0
    for name in names:
        d = os.path.join(work, name, cycle)
        if os.path.isdir(d):
            for root, _dirs, files in os.walk(d):
                for f in files:
                    try:
                        freed += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            shutil.rmtree(d, ignore_errors=True)
    return freed


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _selftest(cfg_path=None):
    cfg = load_config(cfg_path)
    ok = True

    def check(label, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print('%-58s %s %s' % (label, 'PASS' if cond else 'FAIL', detail))

    grid = Grid(cfg['grid'])
    # Shape is whatever the configured grid is -- 1080x1440 for the 1/4 degree
    # tripolar grid this was built against, but the suite is not tied to it.
    # What must hold is that lon/lat/area/mask agree and cover a globe.
    check('grid is 2-D and non-trivial',
          len(grid.shape) == 2 and min(grid.shape) > 1, str(grid.shape))
    ocean_area = float(grid.wgt.sum())
    check('global ocean area ~3.6e14 m2', 3.0e14 < ocean_area < 4.0e14,
          '%.3e' % ocean_area)

    cycle = cfg['cycles'][0]
    work = open_workdir(cfg)
    per_type = {}
    for e in cfg['experiments']:
        files = e.obs_files(cycle, work)
        check('%s: obs files found' % e.name, len(files) > 0, str(len(files)))
        for t, p in files.items():
            per_type.setdefault(t, {})[e.name] = p

    # identities on one ensemble file and one deterministic file
    for t in ('insitu_temp_surface_drifter', 'icec_amsr2_north'):
        if t not in per_type:
            continue
        for name, p in per_type[t].items():
            with Dataset(p) as ds:
                v = sorted(ds['ObsValue'].variables)[0]
                mem = sorted(g for g in ds.groups if g.startswith('hofx0_'))
                if not mem:
                    continue
                y = _get(ds, 'ObsValue', v)
                ombg = _get(ds, 'ombg', v)
                oman = _get(ds, 'oman', v)
                Xb = np.stack([_get(ds, g, v) for g in mem])
                Xa = np.stack([_get(ds, g, v) for g in
                               sorted(g for g in ds.groups
                                      if g.startswith('hofx1_'))])
                mb = _get(ds, 'hofx_y_mean_xb0', v)
                d0 = np.nanmax(np.abs(ombg - (y - Xb.mean(0))))
                d1 = np.nanmax(np.abs(oman - (y - Xa.mean(0))))
                d2 = np.nanmax(np.abs(mb - Xb.mean(0)))
            check('%s/%s: ombg == y - mean(hofx0_*)' % (name, t), d0 < 1e-5,
                  '%.2e' % d0)
            check('%s/%s: oman == y - mean(hofx1_*)' % (name, t), d1 < 1e-5,
                  '%.2e' % d1)
            check('%s/%s: hofx_y_mean_xb0 == mean(hofx0_*)' % (name, t),
                  d2 < 1e-5, '%.2e' % d2)

    # join behaviour
    for t in ('insitu_temp_profile_argo', 'icec_amsr2_north'):
        if t not in per_type or len(per_type[t]) < 2:
            continue
        sets = {n: ObsSet(p, t) for n, p in per_type[t].items()}
        _, counts, both = join_obs(sets)
        sizes = {n: s.n for n, s in sets.items()}
        matched = list(counts.values())[0]['n_matched']
        smallest = min(sizes.values())
        check('%s: join matched all of the smaller sample' % t,
              matched == smallest, '%d of %d %s' % (matched, smallest, sizes))
        check('%s: common-pass sample non-empty' % t, both.sum() > 0,
              '%d' % both.sum())

    # background: both path conventions, the ice mapping, and the depth axis
    import lv_statespace as LS
    for e in cfg['experiments']:
        bc = e.background_cycle(cycle)
        paths = {r: e.background(cycle, r) for r in ('ocean', 'ice')}
        check('%s: background found (dir cycle %s)' % (e.name, bc),
              all(paths.values()),
              ', '.join('%s=%s' % (r, os.path.basename(p) if p else 'MISSING')
                        for r, p in paths.items()))
        if paths['ocean']:
            d = LS.depth_from_h(grid, paths['ocean'])
            mono = d is not None and all(b > a for a, b in zip(d, d[1:]))
            check('%s: depth from h is monotonic' % e.name, mono)
            # the old anlgeom nominal axis put the bottom level at 348 m; the
            # real column reaches the abyss, so anything shallow means the
            # wrong file or a bad h
            check('%s: depth reaches the abyss' % e.name,
                  d is not None and d[-1] > 4000,
                  'bottom %.0f m' % (d[-1] if d else float('nan')))
        if paths['ice']:
            tot = LS.ice_totals(grid, paths['ice'], e.varmap)
            sane = (tot and 1.0 < tot.get('extent_nh', 0) < 20.0
                    and 0.0 < tot.get('extent_sh', 0) < 25.0)
            check('%s: ice extent physically plausible' % e.name, sane,
                  'NH %.1f SH %.1f 1e6 km2' % (tot.get('extent_nh', float('nan')),
                                               tot.get('extent_sh', float('nan'))))
            # 3DVar ice history stores cell-mean thickness; the ratio mapping
            # must turn it into per-ice thickness in a physical range
            with Dataset(paths['ice']) as ds:
                hi = LS._mapped_level(ds, e.varmap, 'hi_div_aice_h', 0)
                a = LS._mapped_level(ds, e.varmap, 'aice_h', 0)
            if hi is not None and a is not None:
                w = np.isfinite(hi) & np.isfinite(a) & (a > 0.15)
                med = float(np.median(hi[w])) if w.any() else float('nan')
                check('%s: ice thickness in a physical range' % e.name,
                      0.1 < med < 6.0, 'median %.2f m' % med)

        # the three spread stages. `bg_ensvar` turned out to be a byte copy of
        # `ensvar_prior`, so a new variance file is not trusted until it has
        # been shown to differ from the others and to satisfy the relation its
        # name implies: RTPS relaxes the analysis back TOWARD the prior, so
        # sigma_a <= sigma_an <= sigma_b at essentially every wet cell.
        ev = {w: e.ensvar(cycle, 'ocean', w) for w in ('prior', 'post', 'an')}
        if all(ev.values()):
            with Dataset(ev['prior']) as dp, Dataset(ev['post']) as dq, \
                    Dataset(ev['an']) as da:
                var = 'Temp' if 'Temp' in da.variables else None
                if var:
                    p = LS._read_level(dp, var, 0)
                    q = LS._read_level(dq, var, 0)
                    s = LS._read_level(da, var, 0)
                    w = np.isfinite(p) & np.isfinite(q) & np.isfinite(s) & (p > 0)
                    sb, sa, sn = np.sqrt(p[w]), np.sqrt(q[w]), np.sqrt(s[w])
                    tol = 1e-6 * np.maximum(sb, 1e-12)
                    frac = float(np.mean((sn >= sa - tol) & (sn <= sb + tol)))
                    check('%s: an_ensvar differs from ensvar_post' % e.name,
                          not np.array_equal(q, s))
                    check('%s: RTPS bracket sigma_a <= sigma_an <= sigma_b'
                          % e.name, frac > 0.99, '%.4f of wet cells' % frac)

    print('\n%s' % ('all checks passed' if ok else 'SOME CHECKS FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    raise SystemExit(_selftest(a.config) if a.selftest else 0)
