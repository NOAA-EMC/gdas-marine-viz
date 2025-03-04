def pytest_ignore_collect(path, config):
    if "tests/oceanview" in str(path):  # Adjust directory name
        return True  # Prevent pytest from even collecting these tests
