import pytest
from typing import Dict
from .pr import pr



@pytest.mark.parametrize(
    "changed_files, pull_request",
    [
        ({ "all_changed_and_modified_files": ".github/scripts/pr.py docs/src/build.md" }, {"draft": "false"}),
    ],
)
def test_build_test_skip(changed_files: Dict, pull_request: Dict):
    pr(changed_files, pull_request)
