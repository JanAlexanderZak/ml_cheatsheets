import os
import pytest

PATH_TO_TEST_DATA_DIR: str = os.path.join(os.path.dirname(__file__), 'src/data/', )


@pytest.fixture
def my_fixture():
    yield 42


@pytest.fixture
def raw_and_processed_file(tmp_path):
    """ tmpdir sets up and closes files 
    Call this fixture to access vairables """
    raw_file_path = tmp_path.join("raw.txt")
    processed_file_path = tmp_path.join("processed.txt")
    with open(raw_file_path) as f:
        f.write("X")
    yield raw_file_path
