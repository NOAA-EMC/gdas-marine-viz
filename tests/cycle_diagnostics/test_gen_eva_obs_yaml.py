import pytest
import yaml
import os
from unittest.mock import patch
from applications.cycle_diagnostics.gen_eva_obs_yaml import gen_eva_obs_yaml


@pytest.fixture
def sample_input_yaml(tmp_path):
    """Creates a temporary sample JEDI YAML input file for testing."""
    yaml_content = {
        "cost function": {
            "observations": {
                "observers": [
                    {
                        "obs space": {
                            "name": "obs_test",
                            "obsdataout": {"engine": {"obsfile": "test_diag_2023010100.nc"}},
                            "simulated variables": ["temperature", "humidity"],
                            "channels": [1, 2, 3]
                        }
                    }
                ]
            }
        }
    }
    yaml_path = tmp_path / "sample_input.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    return str(yaml_path)


@pytest.fixture
def sample_template_yaml(tmp_path):
    """Creates a temporary sample template YAML file for testing."""
    template_content = """---
observations:
  - name: @NAME@
    filename: @FILENAME@
    variables: @VARIABLES@
    @CHANNELSKEY@
"""
    template_path = tmp_path / "template.yaml"
    with open(template_path, "w") as f:
        f.write(template_content)
    return str(template_path)


def test_gen_eva_obs_yaml(sample_input_yaml, sample_template_yaml, tmp_path):
    """Test the YAML generation function."""

    # Call function to generate EVA YAML
    output_dir = str(tmp_path / "output")
    gen_eva_obs_yaml(sample_input_yaml, sample_template_yaml, output_dir)

    # Ensure output directory is created
    assert os.path.exists(output_dir)

    # Get generated YAML file
    output_files = os.listdir(output_dir)
    assert len(output_files) == 1  # Only one YAML file should be generated

    output_yaml = os.path.join(output_dir, output_files[0])

    # Ensure the output YAML file exists
    assert os.path.isfile(output_yaml)

    # Load and verify output YAML content
    with open(output_yaml, "r") as f:
        output_yaml_content = f.read()

    assert "obs_test" in output_yaml_content  # Check that the name was replaced
    assert "test_diag_2023010100.nc" in output_yaml_content  # Ensure file name replacement
    assert "temperature" in output_yaml_content  # Variable substitution
    assert "channels: [1, 2, 3]" in output_yaml_content  # Channels replacement


@patch("logging.error")
def test_gen_eva_obs_yaml_missing_input(mock_logging_error, sample_template_yaml, tmp_path):
    """Test that function handles missing input YAML properly."""
    output_dir = str(tmp_path / "output")

    # Call with a non-existent input YAML
    gen_eva_obs_yaml("non_existent.yaml", sample_template_yaml, output_dir)

    # Ensure an error was logged
    mock_logging_error.assert_called()


@patch("logging.error")
def test_gen_eva_obs_yaml_missing_template(mock_logging_error, sample_input_yaml, tmp_path):
    """Test that function handles missing template YAML properly."""
    output_dir = str(tmp_path / "output")

    # Call with a non-existent template YAML
    gen_eva_obs_yaml(sample_input_yaml, "non_existent.yaml", output_dir)

    # Ensure an error was logged
    mock_logging_error.assert_called()


def test_gen_eva_obs_yaml_creates_directory(sample_input_yaml, sample_template_yaml, tmp_path):
    """Test that the function creates the output directory if it does not exist."""
    output_dir = str(tmp_path / "new_output_dir")

    assert not os.path.exists(output_dir)  # Ensure directory does not exist before running

    # Call function
    gen_eva_obs_yaml(sample_input_yaml, sample_template_yaml, output_dir)

    # Ensure output directory was created
    assert os.path.exists(output_dir)
