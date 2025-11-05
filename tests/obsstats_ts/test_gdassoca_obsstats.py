import pytest
import pandas as pd
from unittest.mock import patch
from applications.obsstats_ts.csv_ts/gdassoca_obsstats import get_inst, ObsStats


@pytest.fixture
def sample_csv(tmp_path):
    """Creates a temporary sample CSV file for testing."""
    csv_content = """date,Ocean,Variable,Exp,RMSE,Bias,Count
2023010100,Global,ombg_noqc,exp1,1.2,-0.5,100
2023010200,Global,ombg_noqc,exp1,1.1,-0.4,110
2023010300,Global,ombg_qc,exp2,0.9,-0.3,95
2023010100,Atlantic,ombg_noqc,exp1,1.3,-0.6,90
2023010200,Atlantic,ombg_qc,exp2,1.0,-0.2,85
"""
    csv_file = tmp_path / "test_obsstats.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


def test_get_inst():
    """Test the extraction of instrument names from file names."""
    filename = "gdas.t00z.ocn.sst_ahi_h08_l3c.stats.csv"
    assert get_inst(filename) == "sst_ahi_h08_l3c"


def test_read_csv(sample_csv):
    """Test that the CSV file is read correctly into the ObsStats object."""
    obs_stats = ObsStats()
    obs_stats.read_csv([sample_csv])

    # Ensure data is loaded
    assert not obs_stats.data.empty
    assert len(obs_stats.data) == 5  # 5 rows in the test file
    assert list(obs_stats.data.columns) == ["date", "Ocean", "Variable", "Exp", "RMSE", "Bias", "Count"]

    # Check date conversion
    assert pd.api.types.is_datetime64_any_dtype(obs_stats.data["date"])


@patch("matplotlib.pyplot.savefig")
def test_plot_timeseries(mock_savefig, sample_csv, tmp_path):
    """Test the plot_timeseries function ensuring correct filtering and plotting."""
    obs_stats = ObsStats()
    obs_stats.read_csv([sample_csv])

    # Call plot_timeseries for 'Global' and 'ombg_noqc'
    experiments = obs_stats.plot_timeseries("Global", "ombg_noqc", inst="sst", dirout=tmp_path)

    # Verify that experiments are returned
    assert "exp1" in experiments
    assert len(experiments) == 1  # Only one experiment in the sample data for this case

    # Verify that a plot was saved
    mock_savefig.assert_called_once()
    expected_filename = f"{tmp_path}/sst_ombg_noqc_Global.png"
    assert mock_savefig.call_args[0][0] == expected_filename
