import fcntl
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'applications/letkf_verif/archive_and_verify.py'
sys.path.insert(0, str(ROOT / 'applications/display_diags'))
import get_hpss_arch  # noqa: E402


def write(path, text):
    path.write_text(text, encoding='utf-8')
    return path


def workflow(tmp_path):
    archive_root = tmp_path / 'archive'
    archive_root.mkdir()
    for name, last in [('one', '2024010112'), ('two', '2024010118')]:
        directory = archive_root / name
        directory.mkdir()
        write(directory / 'cycles.yaml', 'last_extracted_cycle: "%s"\n' % last)
    archive = write(tmp_path / 'archive.yaml', yaml.safe_dump({
        'local hpss root': str(archive_root),
        'experiments': [{'pslot': name} for name in ('one', 'two')],
    }))
    comparisons = []
    for name, experiments in [('first', ['one', 'two']), ('second', ['one'])]:
        directory = tmp_path / name
        directory.mkdir()
        roots = ''.join('  - name: %s\n    root: %s\n'
                        % (e, archive_root / e) for e in experiments)
        write(directory / 'experiments.yaml',
              'other:\n  stop: untouched\ncycles:\n  start: 2024010100\n'
              '  stop: "2024010106" # retain comment\n  step: 6\nexperiments:\n' + roots)
        write(directory / 'run_verif.sh', '')
        comparisons.append(directory)
    fake_archive = write(tmp_path / 'archive.py', 'print("archive complete")\n')
    fake_sbatch = write(tmp_path / 'sbatch',
                        '#!/bin/sh\nprintf "%s\\n" "$PWD $*" >> "$CALLS"\n')
    fake_sbatch.chmod(0o755)
    args = [sys.executable, str(DRIVER), '--archive-config', str(archive),
            '--archive-script', str(fake_archive), '--sbatch', str(fake_sbatch),
            '--lock', str(tmp_path / 'lock')]
    for directory in comparisons:
        args += ['--comparison', str(directory)]
    return args, comparisons, fake_archive


def run(args, tmp_path):
    env = os.environ.copy()
    env.update(CALLS=str(tmp_path / 'calls'))
    env.pop('LETKF_VERIF_ROOT', None)
    return subprocess.run(args, text=True, capture_output=True, env=env)


def test_full_workflow_and_retry(tmp_path):
    args, comparisons, _ = workflow(tmp_path)
    result = run(args, tmp_path)
    assert result.returncode == 0, result.stderr
    for directory, expected in zip(comparisons, ['2024010118', '2024010112']):
        text = (directory / 'experiments.yaml').read_text()
        assert 'stop: untouched' in text
        assert 'stop: "%s" # retain comment' % expected in text
    calls = (tmp_path / 'calls').read_text().splitlines()
    assert calls == ['%s --wait run_verif.sh' % d for d in comparisons]
    # A previous failed build, or a lagging experiment, still gets a rerun.
    assert run(args, tmp_path).returncode == 0
    assert len((tmp_path / 'calls').read_text().splitlines()) == 4


def test_archive_failure_never_submits(tmp_path):
    args, comparisons, fake_archive = workflow(tmp_path)
    write(fake_archive, 'raise SystemExit(1)\n')
    assert run(args, tmp_path).returncode != 0
    assert not (tmp_path / 'calls').exists()
    assert '2024010106' in (comparisons[0] / 'experiments.yaml').read_text()


def test_untracked_root_fails_only_that_comparison(tmp_path):
    args, comparisons, _ = workflow(tmp_path)
    config = comparisons[0] / 'experiments.yaml'
    config.write_text(config.read_text().replace('/archive/two', '/archive/missing'))
    result = run(args, tmp_path)
    assert result.returncode != 0
    assert 'not in the archive config' in result.stderr
    assert (tmp_path / 'calls').read_text().splitlines() == [
        '%s --wait run_verif.sh' % comparisons[1]]


def test_submission_failure_continues_and_never_moves_stop_back(tmp_path):
    args, comparisons, _ = workflow(tmp_path)
    config = comparisons[0] / 'experiments.yaml'
    config.write_text(config.read_text().replace('2024010106', '2024010200'))
    sbatch = tmp_path / 'sbatch'
    write(sbatch, '#!/bin/sh\nprintf "%s\\n" "$PWD" >> "$CALLS"\n'
                  '[ "$(basename "$PWD")" != first ]\n')
    result = run(args, tmp_path)
    assert result.returncode != 0
    assert '2024010200' in config.read_text()
    assert (tmp_path / 'calls').read_text().splitlines() == [
        str(d) for d in comparisons]


def test_overlapping_run_does_not_retrieve_or_submit(tmp_path):
    args, _, fake_archive = workflow(tmp_path)
    write(fake_archive, 'raise SystemExit(1)\n')
    with (tmp_path / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = run(args, tmp_path)
    assert result.returncode == 0
    assert 'skipping' in result.stdout
    assert not (tmp_path / 'calls').exists()


def test_failed_extraction_preserves_tracker(tmp_path, monkeypatch):
    tracker = write(tmp_path / 'cycles.yaml', "last_extracted_cycle: '2024010100'\n")
    listing = tmp_path / 'listing'

    def command(cmd, check):
        if cmd[0] == 'hsi':
            write(listing, '/archive/2024010106\n')
        else:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(get_hpss_arch.subprocess, 'run', command)
    with pytest.raises(subprocess.CalledProcessError):
        get_hpss_arch.extract_last_dir('/archive', listing, tracker)
    assert yaml.safe_load(tracker.read_text())['last_extracted_cycle'] == '2024010100'
