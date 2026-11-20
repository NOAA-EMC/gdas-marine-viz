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
                        '#!/bin/sh\nprintf "%s\\n" "$PWD $*" >> "$CALLS"\n'
                        'echo 12345\n')
    fake_sbatch.chmod(0o755)
    fake_squeue = write(tmp_path / 'squeue', '#!/bin/sh\nexit 0\n')
    fake_squeue.chmod(0o755)
    args = [sys.executable, str(DRIVER), '--archive-config', str(archive),
            '--archive-script', str(fake_archive), '--sbatch', str(fake_sbatch),
            '--squeue', str(fake_squeue),
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
    assert len(calls) == 2
    for directory in comparisons:
        assert any(line.startswith('%s --parsable --job-name=archiveverif-' % directory)
                   and line.endswith(' run_verif.sh') for line in calls)
    assert '--wait' not in '\n'.join(calls)
    assert result.stdout.count('submitted job 12345') == 2
    # A previous failed build, or a lagging experiment, still gets a rerun.
    assert run(args, tmp_path).returncode == 0
    assert len((tmp_path / 'calls').read_text().splitlines()) == 4


def test_non_aligned_cycles_are_accepted(tmp_path):
    args, comparisons, _ = workflow(tmp_path)
    for directory in comparisons:
        config = directory / 'experiments.yaml'
        config.write_text(config.read_text().replace('step: 6', 'step: 24'))

    result = run(args, tmp_path)

    assert result.returncode == 0, result.stderr
    assert 'stop: "2024010118"' in (comparisons[0] / 'experiments.yaml').read_text()
    assert 'stop: "2024010112"' in (comparisons[1] / 'experiments.yaml').read_text()


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
    calls = (tmp_path / 'calls').read_text().splitlines()
    assert len(calls) == 1
    assert calls[0].startswith('%s --parsable ' % comparisons[1])


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
    assert sorted((tmp_path / 'calls').read_text().splitlines()) == sorted(
        str(d) for d in comparisons)


def test_submissions_run_concurrently(tmp_path):
    args, comparisons, _ = workflow(tmp_path)
    sbatch = tmp_path / 'sbatch'
    write(sbatch, '#!%s\n'
          'import os, pathlib, sys, time\n'
          'assert "SBATCH_WAIT" not in os.environ\n'
          'root = pathlib.Path(os.environ["CALLS"]).parent\n'
          '(root / pathlib.Path.cwd().name).with_suffix(".started").touch()\n'
          'deadline = time.monotonic() + 5\n'
          'while len(list(root.glob("*.started"))) < 2:\n'
          '    if time.monotonic() > deadline: sys.exit(1)\n'
          '    time.sleep(0.01)\n'
          'print("456")\n' % sys.executable)
    env = os.environ.copy()
    env.update(CALLS=str(tmp_path / 'calls'), SBATCH_WAIT='1')
    env.pop('LETKF_VERIF_ROOT', None)
    result = subprocess.run(args, text=True, capture_output=True, env=env, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('submitted job 456') == len(comparisons)


def test_active_jobs_prevent_retrieval_and_resubmission(tmp_path):
    args, comparisons, fake_archive = workflow(tmp_path)
    write(fake_archive, 'raise SystemExit(1)\n')
    write(tmp_path / 'squeue', '#!/bin/sh\necho "123 archiveverif-test RUNNING"\n')
    result = run(args, tmp_path)
    assert result.returncode == 0
    assert 'still active' in result.stdout
    assert not (tmp_path / 'calls').exists()
    assert '2024010106' in (comparisons[0] / 'experiments.yaml').read_text()


def test_unselected_driver_job_prevents_retrieval(tmp_path):
    args, comparisons, fake_archive = workflow(tmp_path)
    write(fake_archive, 'raise SystemExit(1)\n')
    second_comparison = args.index('--comparison', args.index('--comparison') + 1)
    del args[second_comparison:second_comparison + 2]
    write(tmp_path / 'squeue', '#!/bin/sh\necho "123 archiveverif-other RUNNING"\n')
    result = run(args, tmp_path)
    assert result.returncode == 0
    assert 'archiveverif-other RUNNING' in result.stdout
    assert not (tmp_path / 'calls').exists()


def test_squeue_failure_prevents_retrieval(tmp_path):
    args, _, _ = workflow(tmp_path)
    write(tmp_path / 'squeue', '#!/bin/sh\necho unavailable >&2\nexit 1\n')
    result = run(args, tmp_path)
    assert result.returncode != 0
    assert 'unavailable' in result.stderr
    assert not (tmp_path / 'calls').exists()


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


@pytest.mark.parametrize('root', ['archive', '~/archive', '$ARCHIVE_ROOT'])
def test_archive_root_is_expanded_before_retrieval(tmp_path, monkeypatch, root):
    archive_root = tmp_path / 'archive'
    destination = archive_root / 'one'
    destination.mkdir(parents=True)
    config = write(tmp_path / 'archive.yaml', yaml.safe_dump({
        'local hpss root': root,
        'experiments': [{'pslot': 'one', 'HPSS root': '/hpss'}],
    }))
    retrieved = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('ARCHIVE_ROOT', str(archive_root))
    monkeypatch.setattr(get_hpss_arch, 'extract_last_dir',
                        lambda hpss_root, **kwargs: retrieved.append(
                            (Path.cwd(), hpss_root)))

    get_hpss_arch.main(str(config))

    assert retrieved == [(destination, '/hpss/one')]
