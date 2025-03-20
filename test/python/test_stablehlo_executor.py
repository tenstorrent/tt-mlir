# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.mlir_module_executor import ExecutionPhase, ExecutionResult
from ttmlir.stablehlo_executor import StableHLOExecutor


def test_compile(print_results: bool = False):
    shlo_module_str = """
        module {
            func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
                %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
                %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
                %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
                return %2 : tensor<1x128xf32>
            }
        }
    """

    ex = StableHLOExecutor()
    result = ex.compile(shlo_module_str)

    if print_results:
        print(result)


def test_execute(print_results: bool = False):
    shlo_module_str = """
        module {
            func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
                %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
                %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
                %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
                return %2 : tensor<1x128xf32>
            }
        }
    """

    ex = StableHLOExecutor()
    result: ExecutionResult = ex.execute(shlo_module_str)

    assert result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER

    if print_results:
        result.flatbuffer.print()
        print("Run on device passed: ", result.device_run_passed)


if __name__ == "__main__":
    test_compile(True)
    test_execute(True)
