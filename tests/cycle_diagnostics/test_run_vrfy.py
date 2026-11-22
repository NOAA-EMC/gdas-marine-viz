import pytest
import os
import subprocess
import sys
from pathlib import Path
import yaml
from unittest.mock import patch
from applications.cycle_diagnostics.run_vrfy import (
    render_html,
    iterate_pdy_range,
    generate_jobcard,
)


@pytest.fixture
def sample_template(tmp_path):
    """Creates a temporary Jinja2 template file for testing."""
    src_template_path = os.path.join(os.path.dirname(__file__), '..', '..', 'templates', 'index_vrfy_marine.html.j2')
    # Copy the template to the temporary directory
    template_path = tmp_path / "index_vrfy_marine.html.j2"
    os.system(f"cp {src_template_path} {template_path}")
    return template_path


@pytest.fixture
def sample_job_template(tmp_path):
    """Creates a temporary job script Jinja2 template file."""
    src_template_path = os.path.join(os.path.dirname(__file__), '..', '..', 'templates', 'vrfy_jobcard.sh.j2')
    # Copy the template to the temporary directory
    template_path = tmp_path / "vrfy_jobcard.sh.j2"
    os.system(f"cp {src_template_path} {template_path}")
    return template_path


@pytest.fixture
def temp_output_file(tmp_path):
    """Creates a temporary output file path for testing."""
    return str(tmp_path / "output.html")


def test_render_html(sample_template, temp_output_file):
    """Test rendering an HTML file from a Jinja2 template."""
    context = {"pslot": "test_slot", "year_list": ["2023", "2024"]}
    render_html(sample_template, temp_output_file, context)

    # Ensure output file exists
    assert os.path.exists(temp_output_file)


def test_iterate_pdy_range():
    """Test the generation of date ranges in YYYYMMDD format."""
    start_pdy = "20230101"
    end_pdy = "20230105"

    expected_dates = ["20230101", "20230102", "20230103", "20230104", "20230105"]
    generated_dates = list(iterate_pdy_range(start_pdy, end_pdy))

    assert generated_dates == expected_dates


def test_generate_jobcard(sample_job_template, temp_output_file):
    """Test generating job script from Jinja2 template."""
    context = {
        "pslot": "test_pslot",
        "cyc": "00",
        "pdy": "20230101",
        "base_exp_path": "/path/to/experiments",
        "homegdasmarineviz": "/path/to/gdasmarineviz"
    }

    # Call the function with real file paths
    generate_jobcard(sample_job_template, temp_output_file, context)

    # Verify output file exists
    assert os.path.exists(temp_output_file)

    # Read and verify output job script content
    with open(temp_output_file, "r") as f:
        content = f.read()

    # TODO: Add more assertions
    assert "#SBATCH" in content
    assert "Execute Marine Verify Analysis" in content


@patch("subprocess.run")
def test_job_submission(mock_subprocess, sample_job_template, temp_output_file):
    """Test if job scripts are submitted correctly using subprocess."""
    context = {"pslot": "test_pslot", "cyc": "00", "pdy": "20230101"}

    generate_jobcard(sample_job_template, temp_output_file, context)

    # Simulate job submission command
    subprocess.run(f"sbatch {temp_output_file}", shell=True)

    # Verify subprocess was called with correct command
    mock_subprocess.assert_called_with(f"sbatch {temp_output_file}", shell=True)


@patch("logging.error")
def test_render_html_missing_template(mock_logging_error, temp_output_file):
    """Test that render_html handles missing templates gracefully."""
    render_html("non_existent_template.j2", temp_output_file, {"pslot": "test"})


def test_selected_cycles_wait_and_report_job_failure(tmp_path):
    root = Path(__file__).resolve().parents[2]
    sbatch = tmp_path / 'sbatch'
    sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n'
                      'case "$*" in *20240102.00*) exit 1;; esac\n')
    sbatch.chmod(0o755)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'pslot': 'test', 'start_pdy': '20240101', 'end_pdy': '20240101',
        'cycs': ['00', '06', '12', '18'], 'homegdasmarineviz': str(root),
        'base_exp_path': str(tmp_path), 'vrfyout': str(tmp_path / 'out'),
    }))
    env = os.environ.copy()
    env.update(CALLS=str(tmp_path / 'calls'))
    result = subprocess.run(
        [sys.executable, str(root / 'applications/cycle_diagnostics/run_vrfy.py'),
         str(config), '--after-cycle', '2024010112', '--through-cycle', '2024010200',
         '--wait', '--job-name', 'archiveverif-test', '--sbatch', str(sbatch)],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'failed with exit code' in result.stderr
    calls = (tmp_path / 'calls').read_text().splitlines()
    assert len(calls) == 2
    assert all('--wait --job-name=archiveverif-test' in line for line in calls)
    assert any('20240101.18.sh' in line for line in calls)
    assert any('20240102.00.sh' in line for line in calls)


def test_update_site_uses_checkout_scripts_for_external_root(tmp_path):
    root = Path(__file__).resolve().parents[2]
    image = tmp_path / 'gdas.20240101.00/vrfy/ana/sst/test.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'png')
    result = subprocess.run(
        [sys.executable, str(root / 'applications/display_diags/update_site.py'), str(tmp_path)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'gdas.20240101.00/vrfy/ana/sst/test.png' in (
        tmp_path / 'images_manifest.json').read_text()
    assert 'gdas.20240101.00/vrfy/ana/sst/test.png' in (
        tmp_path / 'gallery_static.html').read_text()
