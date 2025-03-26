import subprocess
import glob
import os
import re

import pytest

filenames = glob.glob("./ttnn/*.ttnn")

@pytest.mark.parametrize("path", filenames, ids=os.path.basename)
def test_flatbuffers(path: str):
    result = subprocess.run(f"ttrt run {path}", shell=True, capture_output=True, text=True)
    code = result.returncode

    if code == 42:
        pcc_search = re.compile(r'actual_pcc=(\d\.\d+)')
        matches = pcc_search.findall(result.stderr)
        assert len(matches) > 0
        assert False, f"PCC failure: {matches[0]}"
    elif code == 1:
        assert False, "Test run failure"
    else:
        assert code == 0
