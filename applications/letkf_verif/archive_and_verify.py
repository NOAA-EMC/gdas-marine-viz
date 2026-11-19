#!/usr/bin/env python3
"""Retrieve HPSS archives and submit the comparisons using their tracked cycles."""

import argparse
import fcntl
import os
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/scratch3/NCEPDEV/da/Andrew.Eichmann/vrfy/comparisons')
DEFAULT_ARCHIVE = Path('/scratch3/NCEPDEV/da/Andrew.Eichmann/hpss/get_hpss_arch.yaml')


def read_yaml(path):
    with path.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def cycle(value):
    value = str(value)
    if len(value) != 10 or not value.isdigit():
        raise ValueError('invalid cycle: %r (expected YYYYMMDDHH)' % value)
    return datetime.strptime(value, '%Y%m%d%H')


def resolved(path, base):
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return (base / path).resolve()


def advance_stop(config, archive):
    """Advance the stop to the latest tracked cycle among its experiments."""
    data = read_yaml(config)
    spec = data['cycles']
    if not isinstance(spec, dict):
        raise ValueError('%s: cycles must be a start/stop/step mapping' % config)
    start, stop = cycle(spec['start']), cycle(spec['stop'])
    step = timedelta(hours=float(spec.get('step', 6)))
    if step <= timedelta(0) or stop < start or (stop - start) % step:
        raise ValueError('%s: invalid cycle range or step' % config)

    archive_root = resolved(archive['local hpss root'], Path.cwd())
    tracked = {resolved(e['pslot'], archive_root) for e in archive['experiments']}
    found = []
    for experiment in data['experiments']:
        root = resolved(os.environ.get('LETKF_VERIF_ROOT') or experiment['root'],
                        config.parent)
        if root not in tracked:
            raise ValueError('%s: experiment root %s is not in the archive config'
                             % (config, root))
        tracker = root / 'cycles.yaml'
        latest = cycle(read_yaml(tracker)['last_extracted_cycle'])
        print('%s: %s tracked through %s' %
              (config.parent.name, experiment['name'], latest.strftime('%Y%m%d%H')),
              flush=True)
        found.append(latest)
    if not found:
        raise ValueError('%s: no experiments configured' % config)
    latest = max(found)
    if latest <= stop:
        return stop.strftime('%Y%m%d%H')
    if (latest - start) % step:
        raise ValueError('%s: tracked cycle is not on the configured cycle grid' % config)

    text = config.read_text(encoding='utf-8')
    document = yaml.compose(text)
    cycles_node = next(value for key, value in document.value if key.value == 'cycles')
    stop_node = next(value for key, value in cycles_node.value if key.value == 'stop')
    new_stop = latest.strftime('%Y%m%d%H')
    replacement = new_stop if stop_node.style not in ("'", '"') else (
        stop_node.style + new_stop + stop_node.style)
    text = text[:stop_node.start_mark.index] + replacement + text[stop_node.end_mark.index:]
    mode = stat.S_IMODE(config.stat().st_mode)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=config.parent,
                                         delete=False) as stream:
            tmp = Path(stream.name)
            os.fchmod(stream.fileno(), mode)
            stream.write(text)
        os.replace(tmp, config)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return new_stop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-config', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--archive-script', type=Path,
                        default=HERE.parent / 'display_diags/get_hpss_arch.py')
    parser.add_argument('--comparison', type=Path, action='append',
                        help='comparison directory; repeat to select multiple')
    parser.add_argument('--lock', type=Path, default=DEFAULT_ROOT / '.archive-verif.lock')
    parser.add_argument('--sbatch', default='sbatch', help='sbatch executable')
    args = parser.parse_args(argv)
    config = args.archive_config.resolve()
    comparisons = args.comparison or [DEFAULT_ROOT / name
                                      for name in ('baseline', 'realtime', 'stream2')]

    with args.lock.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another archive and verification run holds %s; skipping.' % args.lock)
            return 0
        archive = read_yaml(config)
        root = resolved(archive['local hpss root'], Path.cwd())
        print('Retrieving archives using %s' % config, flush=True)
        subprocess.run([sys.executable, str(args.archive_script.resolve()),
                        '--yaml_file', str(config)], cwd=root, check=True)
        failures = []
        for directory in comparisons:
            directory = directory.resolve()
            try:
                stop = advance_stop(directory / 'experiments.yaml', archive)
                print('Submitting %s through %s' % (directory, stop), flush=True)
                env = os.environ.copy()
                env.pop('CFG', None)
                env.pop('APP', None)
                subprocess.run([args.sbatch, '--wait', 'run_verif.sh'],
                               cwd=directory, env=env, check=True)
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError,
                    subprocess.CalledProcessError) as exc:
                print('%s: verification failed: %s' % (directory, exc),
                      file=sys.stderr, flush=True)
                failures.append(directory)
        return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
