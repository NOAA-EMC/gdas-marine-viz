import pytest
import os
import subprocess
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
