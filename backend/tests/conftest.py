import os
import sys

import pytest

# Get the directory of the current script (the directory where conftest.py resides)
current_dir = os.path.dirname(__file__)

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Add the project root directory to sys.path
sys.path.append(project_root)


@pytest.fixture(scope="module")
def mlruns_dir(tmp_path_factory):
    mlruns_dir = tmp_path_factory.mktemp("mlruns")
    return mlruns_dir
