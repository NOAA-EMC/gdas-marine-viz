import subprocess


def test_flake8():
    """Run flake8 to check coding style."""
    result = subprocess.run(["flake8", "applications", "tests"], capture_output=True, text=True)
    assert result.returncode == 0, f"Flake8 found issues:\n{result.stdout}"
