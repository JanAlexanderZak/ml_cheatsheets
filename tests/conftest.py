import os
import pytest

PATH_TO_TEST_DATA_DIR: str = os.path.join(os.path.dirname(__file__), 'src/data/', )


@pytest.fixture
def my_fixture():
    yield 42
