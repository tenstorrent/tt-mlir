# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from op_by_op_infra.mlir_module_splitter import MLIRModuleSplitter

from .fixtures import *


def test_shlo_module_split(shlo_module_str: str, expected_shlo_module_count: int):
    splitter = MLIRModuleSplitter()
    sub_ops = splitter.split(shlo_module_str)
    sub_modules = splitter.sub_modules

    assert len(sub_ops) == len(sub_modules) == expected_shlo_module_count


def test_multi_func_shlo_module_split(
    multi_func_shlo_module_str: str, expected_multi_func_shlo_module_count: int
):
    splitter = MLIRModuleSplitter()
    sub_ops = splitter.split(multi_func_shlo_module_str)
    sub_modules = splitter.sub_modules

    assert len(sub_ops) == len(sub_modules) == expected_multi_func_shlo_module_count


def test_ttir_module_split(ttir_module_str: str, expected_ttir_module_count: int):
    splitter = MLIRModuleSplitter()
    sub_ops = splitter.split(ttir_module_str)
    sub_modules = splitter.sub_modules

    assert len(sub_ops) == len(sub_modules) == expected_ttir_module_count


def test_ttnn_module_split(ttnn_module_str: str, expected_ttnn_module_count: int):
    splitter = MLIRModuleSplitter()
    sub_ops = splitter.split(ttnn_module_str)
    sub_modules = splitter.sub_modules

    assert len(sub_ops) == len(sub_modules) == expected_ttnn_module_count
