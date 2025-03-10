# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ttir_module_compiler import TTIRModuleCompiler


def test_compile_full_module(print_results: bool = False):
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

    compiler = TTIRModuleCompiler.create_from_module_str(ttir_module_str)
    fb = compiler.compile_full_module()

    if print_results:
        fb.print()


def test_compile_op_by_op(print_results: bool = False):
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

    compiler = TTIRModuleCompiler.create_from_module_str(ttir_module_str)
    fbs = compiler.compile_op_by_op()

    assert len(fbs) == 8, "Compiler isn't working as expected"

    if print_results:
        for fb in fbs:
            fb.print()


if __name__ == "__main__":
    test_compile_full_module(True)
    test_compile_op_by_op(True)
