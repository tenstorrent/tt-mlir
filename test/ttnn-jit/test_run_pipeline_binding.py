# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from ttnn_jit.ttmlir.passes import run_pipeline
from ttnn_jit.ttmlir.ir import Context, Module


def test_run_pipeline_rejects_unknown_name():
    with Context():
        module = Module.parse("module {}")
        with pytest.raises(RuntimeError, match="Unknown pipeline"):
            run_pipeline(module, "no-such-pipeline")
