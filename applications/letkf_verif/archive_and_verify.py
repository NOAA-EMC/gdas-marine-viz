#!/usr/bin/env python3
"""Retrieve HPSS archives and submit the comparisons using their tracked cycles."""

import argparse
import fcntl
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from jinja2 import Template, TemplateError

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/scratch3/NCEPDEV/da/Andrew.Eichmann/vrfy/comparisons')
DEFAULT_ARCHIVE = Path('/scratch3/NCEPDEV/da/Andrew.Eichmann/hpss/get_hpss_arch.yaml')
DEFAULT_CYCLE_ROOT = DEFAULT_ROOT.parent
ACTIVE_STATES = ('pending,running,suspended,completing,configuring,expediting,'
                 'resizing,resv_del_hold,requeued,requeue_fed,requeue_hold,'
                 'signaling,special_exit,stage_out,stopped')


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


def job_name(directory):
    digest = hashlib.sha256(os.fsencode(str(directory))).hexdigest()[:16]
    return 'archiveverif-' + digest


def save_cycle_state(path, state):
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False) as stream:
            tmp = Path(stream.name)
            yaml.safe_dump(state, stream)
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def initialize_cycle_state(archive, state_path):
    root = resolved(archive['local hpss root'], Path.cwd())
    state = read_yaml(state_path) if state_path.exists() else {}
    if not isinstance(state, dict):
        raise ValueError('%s: invalid cycle state' % state_path)
    for experiment in archive['experiments']:
        pslot = experiment['pslot']
        if not isinstance(pslot, str) or Path(pslot).name != pslot or pslot in ('', '.', '..'):
            raise ValueError('invalid pslot: %r' % pslot)
        archive_dir = resolved(pslot, root)
        key = str(archive_dir)
        latest = cycle(read_yaml(archive_dir / 'cycles.yaml')['last_extracted_cycle'])
        if key not in state:
            state[key] = latest.strftime('%Y%m%d%H')
        else:
            cycle(state[key])
    save_cycle_state(state_path, state)
    return root, state


def validate_cycle_config(config_path, pslot, archive_dir, output_dir):
    text = config_path.read_text(encoding='utf-8')
    raw = yaml.safe_load(text)
    config = yaml.safe_load(Template(text).render(pslot=raw['pslot']))
    if config['pslot'] != pslot:
        raise ValueError('%s: pslot does not match %s' % (config_path, pslot))
    for field, expected in (('base_exp_path', archive_dir), ('vrfyout', output_dir)):
        if resolved(config[field], config_path.parent) != expected.resolve():
            raise ValueError('%s: %s must resolve to %s' % (config_path, field, expected))


def submit_cycles(archive, archive_root, state, state_path, args):
    failures = []
    submitted = []
    for experiment in archive['experiments']:
        pslot = experiment['pslot']
        archive_dir = resolved(pslot, archive_root)
        output_dir = args.cycle_root / pslot
        run_dir = output_dir / 'run'
        config = run_dir / 'vrfy_config.yaml'
        try:
            validate_cycle_config(config, pslot, archive_dir, output_dir)
            previous = cycle(state[str(archive_dir)])
            latest = cycle(read_yaml(archive_dir / 'cycles.yaml')['last_extracted_cycle'])
            if latest < previous:
                raise ValueError('archive tracker moved backwards from %s to %s' %
                                 (previous, latest))
            process = None
            if latest > previous:
                command = [sys.executable, str(args.cycle_script.resolve()), str(config),
                           '--after-cycle', previous.strftime('%Y%m%d%H'),
                           '--through-cycle', latest.strftime('%Y%m%d%H'), '--wait',
                           '--job-name', job_name(output_dir), '--sbatch', args.sbatch]
                print('%s: plotting cycles after %s through %s' %
                      (pslot, previous.strftime('%Y%m%d%H'), latest.strftime('%Y%m%d%H')),
                      flush=True)
                process = subprocess.Popen(command, cwd=run_dir)
            submitted.append((pslot, archive_dir, output_dir, latest, process))
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, TemplateError) as exc:
            print('%s: cycle verification failed: %s' % (pslot, exc),
                  file=sys.stderr, flush=True)
            failures.append(pslot)
    for pslot, archive_dir, output_dir, latest, process in submitted:
        if process is not None and process.wait():
            print('%s: cycle verification failed (exit %s)' % (pslot, process.returncode),
                  file=sys.stderr, flush=True)
            failures.append(pslot)
            continue
        try:
            subprocess.run([sys.executable, str(args.site_script.resolve()), str(output_dir)],
                           check=True)
            updated = dict(state)
            updated[str(archive_dir)] = latest.strftime('%Y%m%d%H')
            save_cycle_state(state_path, updated)
            state.update(updated)
        except (OSError, subprocess.CalledProcessError) as exc:
            print('%s: gallery update failed: %s' % (pslot, exc), file=sys.stderr, flush=True)
            failures.append(pslot)
    return failures


def advance_stop(config, archive):
    """Advance the stop to the latest tracked cycle among its experiments."""
    data = read_yaml(config)
    spec = data['cycles']
    if not isinstance(spec, dict):
        raise ValueError('%s: cycles must be a start/stop/step mapping' % config)
    start, stop = cycle(spec['start']), cycle(spec['stop'])
    step = timedelta(hours=float(spec.get('step', 6)))
    if step <= timedelta(0) or stop < start:
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
    parser.add_argument('--squeue', default='squeue', help='squeue executable')
    parser.add_argument('--cycle-root', type=Path, default=DEFAULT_CYCLE_ROOT,
                        help='parent directory of experiment verification outputs')
    parser.add_argument('--cycle-script', type=Path,
                        default=HERE.parent / 'cycle_diagnostics/run_vrfy.py')
    parser.add_argument('--site-script', type=Path,
                        default=HERE.parent / 'display_diags/update_site.py')
    parser.add_argument('--cycle-state', type=Path,
                        help='persistent cycle plotting checkpoint (default: beside lock)')
    args = parser.parse_args(argv)
    config = args.archive_config.resolve()
    args.cycle_root = args.cycle_root.resolve()
    state_path = (args.cycle_state or args.lock.with_name(args.lock.name + '.cycles.yaml')).resolve()
    comparisons = [directory.resolve() for directory in (
        args.comparison or [DEFAULT_ROOT / name
                            for name in ('baseline', 'realtime', 'stream2')])]
    if len(set(comparisons)) != len(comparisons):
        parser.error('comparison directories must be distinct')

    with args.lock.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another archive and verification run holds %s; skipping.' % args.lock)
            return 0
        active = subprocess.run(
            [args.squeue, '--all', '--noheader', '--user', str(os.getuid()),
             '--states', ACTIVE_STATES, '--format', '%i %j %T'],
            check=True, text=True, stdout=subprocess.PIPE)
        active_jobs = []
        for line in active.stdout.splitlines():
            fields = line.split(maxsplit=2)
            if len(fields) > 1 and fields[1].startswith('archiveverif-'):
                active_jobs.append(line)
        if active_jobs:
            print('Previous verification jobs are still active; skipping:\n%s'
                  % '\n'.join(active_jobs), flush=True)
            return 0
        archive = read_yaml(config)
        archive_root, cycle_state = initialize_cycle_state(archive, state_path)
        print('Retrieving archives using %s' % config, flush=True)
        subprocess.run([sys.executable, str(args.archive_script.resolve()),
                        '--yaml_file', str(config)], check=True)
        failures = []
        submitted = []
        for directory in comparisons:
            try:
                stop = advance_stop(directory / 'experiments.yaml', archive)
                print('Submitting %s through %s' % (directory, stop), flush=True)
                env = os.environ.copy()
                env.pop('CFG', None)
                env.pop('APP', None)
                env.pop('SBATCH_WAIT', None)
                process = subprocess.Popen(
                    [args.sbatch, '--parsable', '--job-name=' + job_name(directory),
                     'run_verif.sh'], cwd=directory, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                submitted.append((directory, process))
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
                print('%s: submission failed: %s' % (directory, exc),
                      file=sys.stderr, flush=True)
                failures.append(directory)
        for directory, process in submitted:
            stdout, stderr = process.communicate()
            if process.returncode:
                print('%s: submission failed (exit %s): %s' %
                      (directory, process.returncode, stderr.strip()),
                      file=sys.stderr, flush=True)
                failures.append(directory)
            else:
                print('%s: submitted job %s' % (directory, stdout.strip()), flush=True)
                if stderr:
                    print(stderr, file=sys.stderr, end='', flush=True)
        failures.extend(submit_cycles(archive, archive_root, cycle_state, state_path, args))
        return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
