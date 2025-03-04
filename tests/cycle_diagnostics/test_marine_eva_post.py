import pytest
import yaml
import os
from unittest.mock import patch
from applications.cycle_diagnostics.marine_eva_post import marine_eva_post, vminmax


@pytest.fixture
def sample_input_yaml(tmp_path):
    """Creates a temporary sample input YAML file for testing."""
    yaml_content = {
        "datasets": [
            {"name": "test_dataset", "filenames": ["test1.nc", "test2.nc"]}
        ],
        "graphics": {
            "figure_list": [
                {
                    "batch figure": {"variables": ["seaSurfaceTemperature"]},
                    "plots": [
                        {
                            "layers": [
                                {"type": "MapScatter"}
                            ]
                        }
                    ]
                }
            ]
        }
    }
    yaml_path = tmp_path / "sample_input.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    return str(yaml_path)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Creates a temporary output directory for testing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def temp_diag_dir(tmp_path):
    """Creates a temporary diagnostics directory for testing."""
    diag_dir = tmp_path / "diag"
    diag_dir.mkdir()
    return str(diag_dir)


def test_marine_eva_post(sample_input_yaml, temp_output_dir, temp_diag_dir):
    """Test that marine_eva_post correctly modifies and generates a YAML file."""

    # Run the function
    marine_eva_post(sample_input_yaml, temp_output_dir, temp_diag_dir)

    # Ensure output file is created
    output_yaml_path = os.path.join(temp_output_dir, os.path.basename(sample_input_yaml))
    assert os.path.isfile(output_yaml_path)

    # Load and verify output YAML content
    with open(output_yaml_path, "r") as f:
        output_yaml_content = yaml.safe_load(f)

    # Check that filenames are updated correctly
    expected_filenames = [
        os.path.join(temp_diag_dir, "test1.nc"),
        os.path.join(temp_diag_dir, "test2.nc"),
    ]
    assert output_yaml_content["datasets"][0]["filenames"] == expected_filenames

    # Check that vmin and vmax are updated for MapScatter layers
    scatter_layer = output_yaml_content["graphics"]["figure_list"][0]["plots"][0]["layers"][0]
    assert scatter_layer["vmin"] == vminmax["seaSurfaceTemperature"]["vmin"]
    assert scatter_layer["vmax"] == vminmax["seaSurfaceTemperature"]["vmax"]


@patch("logging.error")
def test_marine_eva_post_missing_input(mock_logging_error, temp_output_dir, temp_diag_dir):
    """Test that function handles missing input YAML properly."""

    marine_eva_post("non_existent.yaml", temp_output_dir, temp_diag_dir)

    # Ensure an error was logged
    mock_logging_error.assert_called()


@patch("logging.error")
def test_marine_eva_post_unwritable_output(mock_logging_error, sample_input_yaml, temp_diag_dir):
    """Test that function handles errors when output YAML cannot be written."""

    # Use a non-existent directory for output
    non_writable_dir = "/non_existent_directory"

    marine_eva_post(sample_input_yaml, non_writable_dir, temp_diag_dir)

    # Ensure an error was logged
    mock_logging_error.assert_called()
