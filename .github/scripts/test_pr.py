#import pytest
from typing import Dict
from pr import main
import os

#@pytest.fixture(scope="session", autouse=True)
def test_build_test_skip():
        os.environ["pull_request"] = {"draft": "false"}
        os.environ["changed_files"] =  { "all_changed_and_modified_files": ".github/scripts/pr.py docs/src/build.md" } 
        main()
    
test_build_test_skip()