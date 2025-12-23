# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from op_by_op_infra.mlir_module_splitter import MLIRModuleSplitter

from .fixtures import *


def test_shlo_module_split(shlo_module_str: str):
    splitter = MLIRModuleSplitter()
    sub_ops =  splitter.split(shlo_module_str)
    sub_modules = splitter.sub_modules

    assert len(sub_ops) == len(sub_modules) == 3


def test_multi_func_shlo_module_split(multi_func_shlo_module_str: str):
    splitter = MLIRModuleSplitter()
    sub_ops =  splitter.split(multi_func_shlo_module_str)
    sub_modules = splitter.sub_modules

    assert len(sub_ops) == len(sub_modules) == 7


def test_ttir_module_split(ttir_module_str: str):
    splitter = MLIRModuleSplitter()
    sub_ops =  splitter.split(ttir_module_str)
    sub_modules = splitter.sub_modules

    assert len(sub_ops) == len(sub_modules) == 4


def test_ttnn_module_split(ttnn_module_str: str):
    splitter = MLIRModuleSplitter()
    sub_ops =  splitter.split(ttnn_module_str)
    sub_modules = splitter.sub_modules

    assert len(sub_ops) == len(sub_modules) == 5
