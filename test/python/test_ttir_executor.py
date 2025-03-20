# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.mlir_module_executor import ExecutionPhase, ExecutionResult
from ttmlir.ttir_executor import TTIRExecutor


def test_compile(print_results: bool = False):
    ttir_module_str = """
        module {
            func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
                %0 = tensor.empty() : tensor<1x128xf32>
                %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                %2 = tensor.empty() : tensor<1x128xf32>
                %3 = "ttir.reshape"(%arg1, %2) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                %4 = tensor.empty() : tensor<1x128xf32>
                %5 = "ttir.broadcast"(%3, %4) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                %6 = tensor.empty() : tensor<1x128xf32>
                %7 = "ttir.add"(%1, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                return %7 : tensor<1x128xf32>
            }
        }
    """

    ex = TTIRExecutor()
    result = ex.compile(ttir_module_str)

    if print_results:
        print(result)


def test_execute(print_results: bool = False):
    ttir_module_str = """
        module {
            func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
                %0 = tensor.empty() : tensor<1x128xf32>
                %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                %2 = tensor.empty() : tensor<1x128xf32>
                %3 = "ttir.reshape"(%arg1, %2) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                %4 = tensor.empty() : tensor<1x128xf32>
                %5 = "ttir.broadcast"(%3, %4) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                %6 = tensor.empty() : tensor<1x128xf32>
                %7 = "ttir.add"(%1, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
                return %7 : tensor<1x128xf32>
            }
        }
    """

    ex = TTIRExecutor()
    result: ExecutionResult = ex.execute(ttir_module_str)

    assert result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER

    if print_results:
        result.flatbuffer.print()
        print("Run on device passed: ", result.device_run_passed)


if __name__ == "__main__":
    test_compile(True)
    # test_execute(True)
