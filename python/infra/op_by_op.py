# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from ttmlir.ir import Module

from .mlir_module_executor import ExecutionResult, MLIRModuleExecutor
from .mlir_module_splitter import MLIRModuleSplitter
from .stablehlo_executor import StableHLOExecutor
from .ttir_executor import TTIRExecutor
from .ttnn_executor import TTNNExecutor
from .utils import ModuleDialect


def _create_executor(dialect: ModuleDialect) -> MLIRModuleExecutor:
    if dialect == ModuleDialect.STABLE_HLO:
        return StableHLOExecutor()
    elif dialect == ModuleDialect.TTIR:
        return TTIRExecutor()
    elif dialect == ModuleDialect.TTNN:
        return TTNNExecutor()
    else:
        raise ValueError(f"Unkown dialect: {dialect.name}")


def split_and_execute(module: Module | str) -> List[ExecutionResult]:
    """
    Splits `module` into constituent ops, wraps each of them in a module, for each
    module it compiles, generates flatbuffer and runs it.

    Returns list of `ExecutionResult`s, one for each constituent op.
    """
    splitter = MLIRModuleSplitter()
    # TODO fetch this from module and assert they all have the same dialect
    dialect = ModuleDialect.TTIR
    executor = _create_executor(dialect)

    # TODO we should provide mapping constituent op : result.
    return [executor.execute(sub_module) for sub_module in splitter.split(module)]


def compile_split_and_execute(module: Module | str) -> List[ExecutionResult]:
    """
    Compiles `module` to generate TTNN module, splits it into constituent ops, wraps
    each of them in a module, for each module it compiles, generates flatbuffer and runs
    it.

    Returns list of `ExecutionResult`s, one for each constituent op.
    """
    # TODO extract it from given module
    dialect = ModuleDialect.STABLE_HLO

    executor = _create_executor(dialect)
    ttnn_module = executor.compile(module)

    splitter = MLIRModuleSplitter()
    ttnn_executor = TTNNExecutor()

    # TODO we should provide mapping constituent op : result.
    return [
        ttnn_executor.execute(sub_module) for sub_module in splitter.split(ttnn_module)
    ]
