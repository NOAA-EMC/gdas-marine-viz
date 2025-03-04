import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from applications.oceanview.oceanview import Instrument, ioda, observation_space


@pytest.fixture
def mock_nc_file():
    """Creates a mock NetCDF file structure with realistic groups and variables."""

    mock_nc = MagicMock()

    # Define the dimension sizes
    num_locations = 90

    # Mock the groups and variables
    mock_nc.groups = {
        "MetaData": MagicMock(variables={
            "dateTime": np.random.randint(1625184000, 1625270400, size=num_locations),  # Random timestamps
            "latitude": np.random.uniform(-90, 90, size=num_locations),
            "longitude": np.random.uniform(-180, 180, size=num_locations),
            "oceanBasin": np.random.randint(1, 10, size=num_locations),
            "sequenceNumber": np.random.randint(1, 10, size=num_locations),
        }),
        "ObsValue": MagicMock(variables={
            "waterTemperature": np.random.uniform(270, 310, size=num_locations),  # SST in Kelvin
        }),
        "ObsError": MagicMock(variables={
            "waterTemperature": np.random.uniform(0.1, 2.0, size=num_locations),  # Example error values
        }),
        "hofx0": MagicMock(variables={
            "waterTemperature": np.random.uniform(270, 310, size=num_locations),  # Simulated SST
        }),
        "ombg": MagicMock(variables={
            "waterTemperature": np.random.uniform(-2, 2, size=num_locations),  # Observation-minus-background
        }),
        "oman": MagicMock(variables={
            "waterTemperature": np.random.uniform(-2, 2, size=num_locations),  # Observation-minus-analysis
        }),
        "EffectiveQC0": MagicMock(variables={
            "waterTemperature": np.random.randint(0, 3, size=num_locations),  # QC flags
        }),
        "EffectiveError0": MagicMock(variables={
            "waterTemperature": np.random.randint(0, 3, size=num_locations),  # QC flags
        }),
    }

    return mock_nc


def test_instrument():
    """Test the Instrument class initialization."""
    inst = Instrument(name='Argo', instid=508, varid=np.array([101, 102]), zmin=0, zmax=2000)

    assert inst.name == 'Argo'
    assert inst.instid == 508
    assert np.array_equal(inst.varid, np.array([101, 102]))
    assert inst.zmin == 0
    assert inst.zmax == 2000


@patch("applications.oceanview.oceanview.Dataset")
def test_ioda(mock_dataset, mock_nc_file):
    """Test the ioda class and its ability to read mock NetCDF data."""
    mock_dataset.return_value = mock_nc_file

    test_files = ["insitu_profile_argo.2021070200.nc4", "insitu_profile_argo.2021070100.nc4"]
    ioda_instance = ioda(test_files, varname="waterTemperature")

    # Check that the expected properties are populated
    assert len(ioda_instance.obs) > 0
    assert len(ioda_instance.lon) > 0
    assert len(ioda_instance.lat) > 0


@patch("applications.oceanview.oceanview.Dataset")
def test_observation_space(mock_dataset, mock_nc_file):
    """Test the observation_space class instantiation with mock NetCDF files."""
    mock_dataset.return_value = mock_nc_file
    test_files = ["insitu_profile_argo.2021070200.nc4"]

    obs_space_instance = observation_space(test_files, varname="waterTemperature")

    assert isinstance(obs_space_instance, observation_space)
    assert obs_space_instance.iodafname == test_files
    assert len(obs_space_instance.ioda.lon) > 0  # Check if mock data is loaded properly
