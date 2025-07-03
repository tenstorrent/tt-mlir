# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path
from chisel.core.ops import IRModule, attr_type_set
from ttmlir.ir import Context
from chisel.core.enums import ExecutionType



# Base folder
BASE = Path("/proj_sw/user_dev/ndrakulic/chisel")

# Define test configs: model -> list of main_fn
TEST_CONFIGS = {
    "dare": ["forward"],
    "ffe_llama_l1": ["forward", "backward"],
    "ffe_mnist1": ["forward", "backward"],
    "ffe_mnist3": ["forward", "backward"],
    "xla_bert": ["main"],
    "xla_bert_l1": ["main"],
    "xla_mnist": ["main"],
}

# Create parameter list
params = []
for model, main_fns in TEST_CONFIGS.items():
    for ext, execution in [("ttir.mlir", ExecutionType.GOLDEN), ("ttnn.mlir", ExecutionType.DEVICE)]:
        for fn in main_fns:
            path = BASE / model / ext
            params.append((path, execution, fn))

@pytest.mark.parametrize("path, execution_type, main_fn", params)
def test_ir_module(path: Path, execution_type: ExecutionType, main_fn: str):
    ir_module = IRModule(
        mlir_text=path.read_text(),
        context=Context(),
        execution_type=execution_type,
        main_op_name=main_fn,
    )