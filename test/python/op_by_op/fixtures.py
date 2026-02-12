# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from ttmlir.compile_and_run import stablehlo_to_ttir, ttir_to_ttnn


@pytest.fixture
def shlo_module_str() -> str:
    lit_test = (
        "test/ttmlir/Dialect/StableHLO/op_by_op_infra_examples/add_op_with_reshape.mlir"
    )

    with open(lit_test, "r") as f:
        return f.read()


@pytest.fixture
def expected_shlo_module_count() -> int:
    return 3


@pytest.fixture
def ttir_module_str(shlo_module_str: str) -> str:
    return stablehlo_to_ttir(shlo_module_str)


@pytest.fixture
def expected_ttir_module_count() -> int:
    return 4


@pytest.fixture
def ttnn_module_str(ttir_module_str: str) -> str:
    return ttir_to_ttnn(ttir_module_str)


@pytest.fixture
def expected_ttnn_module_count() -> int:
    return 2


@pytest.fixture
def multi_func_shlo_module_str() -> str:
    lit_test = "test/ttmlir/Dialect/StableHLO/op_by_op_infra_examples/multi_func_shlo_module.mlir"

    with open(lit_test, "r") as f:
        return f.read()


@pytest.fixture
def expected_multi_func_shlo_module_count() -> int:
    return 5
