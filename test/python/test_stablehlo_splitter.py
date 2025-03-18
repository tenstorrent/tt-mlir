# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.stablehlo_splitter import StableHLOSplitter


def test1(print_results: bool = False):
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

    splitter = StableHLOSplitter.create_from_module_str(shlo_module_str)
    sub_ops = splitter.get_sub_ops()
    sub_modules = splitter.get_sub_modules()

    assert len(sub_ops) == len(sub_modules) == 3, f"Splitter isn't working as expected"

    if print_results:
        for op in sub_ops:
            print(op)

        print()

        for m in sub_modules:
            print(str(m))


def test2(print_results: bool = False):
    shlo_module_str = """
        module {
            func.func public @main(%arg0: tensor<1x4xi32>, %arg1: tensor<512x768xf32>, %arg2: tensor<2x768xf32>, %arg3: tensor<30522x768xf32>) -> (tensor<1x4x768xf32>) {
                %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                %c = stablehlo.constant dense<1> : tensor<i32>
                %c_64 = stablehlo.constant dense<0> : tensor<i32>
                %0 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
                %1 = stablehlo.iota dim = 0 : tensor<4xi32>
                %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<4xi32>) -> tensor<1x4xi32>
                %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
                %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<12x12xf32>
                %5 = stablehlo.convert %4 : (tensor<12x12xf32>) -> tensor<12x12xi32>
                %6 = call @_take(%arg3, %arg0) : (tensor<30522x768xf32>, tensor<1x4xi32>) -> tensor<1x4x768xf32>
                %7 = call @_take_0(%arg1, %2) : (tensor<512x768xf32>, tensor<1x4xi32>) -> tensor<1x4x768xf32>
                %8 = call @_take_1(%arg2, %0) : (tensor<2x768xf32>, tensor<1x4xi32>) -> tensor<1x4x768xf32>
                %9 = stablehlo.add %6, %8 : tensor<1x4x768xf32>
                %10 = stablehlo.add %9, %7 : tensor<1x4x768xf32>
                return %10 : tensor<1x4x768xf32>
            }
            func.func private @_take(%arg0: tensor<30522x768xf32>, %arg1: tensor<1x4xi32>) -> tensor<1x4x768xf32> {
                %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
                %c = stablehlo.constant dense<true> : tensor<i1>
                %c_0 = stablehlo.constant dense<30521> : tensor<1xi32>
                %c_1 = stablehlo.constant dense<30522> : tensor<i32>
                %c_2 = stablehlo.constant dense<0> : tensor<i32>
                %0 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
                %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi1>
                %2 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
                %3 = stablehlo.add %arg1, %2 : tensor<1x4xi32>
                %4 = call @_where(%1, %3, %arg1) : (tensor<1x4xi1>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
                %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x4xi32>) -> tensor<1x4x1xi32>
                %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x4x1xi32>
                %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x4x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi1>
                %8 = stablehlo.broadcast_in_dim %c_0, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
                %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x4x1xi32>
                %10 = stablehlo.compare  LE, %5, %9,  SIGNED : (tensor<1x4x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi1>
                %11 = stablehlo.and %7, %10 : tensor<1x4x1xi1>
                %12 = stablehlo.reduce(%11 init: %c) applies stablehlo.and across dimensions = [2] : (tensor<1x4x1xi1>, tensor<i1>) -> tensor<1x4xi1>
                %13 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 768>}> : (tensor<30522x768xf32>, tensor<1x4x1xi32>) -> tensor<1x4x768xf32>
                %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x4xi1>) -> tensor<1x4x768xi1>
                %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x768xf32>
                %16 = stablehlo.select %14, %13, %15 : tensor<1x4x768xi1>, tensor<1x4x768xf32>
                return %16 : tensor<1x4x768xf32>
            }
            func.func private @_take_0(%arg0: tensor<512x768xf32>, %arg1: tensor<1x4xi32>) -> tensor<1x4x768xf32> {
                %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
                %c = stablehlo.constant dense<true> : tensor<i1>
                %c_0 = stablehlo.constant dense<511> : tensor<1xi32>
                %c_1 = stablehlo.constant dense<512> : tensor<i32>
                %c_2 = stablehlo.constant dense<0> : tensor<i32>
                %0 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
                %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi1>
                %2 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
                %3 = stablehlo.add %arg1, %2 : tensor<1x4xi32>
                %4 = call @_where(%1, %3, %arg1) : (tensor<1x4xi1>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
                %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x4xi32>) -> tensor<1x4x1xi32>
                %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x4x1xi32>
                %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x4x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi1>
                %8 = stablehlo.broadcast_in_dim %c_0, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
                %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x4x1xi32>
                %10 = stablehlo.compare  LE, %5, %9,  SIGNED : (tensor<1x4x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi1>
                %11 = stablehlo.and %7, %10 : tensor<1x4x1xi1>
                %12 = stablehlo.reduce(%11 init: %c) applies stablehlo.and across dimensions = [2] : (tensor<1x4x1xi1>, tensor<i1>) -> tensor<1x4xi1>
                %13 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 768>}> : (tensor<512x768xf32>, tensor<1x4x1xi32>) -> tensor<1x4x768xf32>
                %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x4xi1>) -> tensor<1x4x768xi1>
                %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x768xf32>
                %16 = stablehlo.select %14, %13, %15 : tensor<1x4x768xi1>, tensor<1x4x768xf32>
                return %16 : tensor<1x4x768xf32>
            }
            func.func private @_take_1(%arg0: tensor<2x768xf32>, %arg1: tensor<1x4xi32>) -> tensor<1x4x768xf32> {
                %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
                %c = stablehlo.constant dense<true> : tensor<i1>
                %c_0 = stablehlo.constant dense<1> : tensor<1xi32>
                %c_1 = stablehlo.constant dense<2> : tensor<i32>
                %c_2 = stablehlo.constant dense<0> : tensor<i32>
                %0 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
                %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi1>
                %2 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
                %3 = stablehlo.add %arg1, %2 : tensor<1x4xi32>
                %4 = call @_where(%1, %3, %arg1) : (tensor<1x4xi1>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi32>
                %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x4xi32>) -> tensor<1x4x1xi32>
                %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x4x1xi32>
                %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x4x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi1>
                %8 = stablehlo.broadcast_in_dim %c_0, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
                %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x4x1xi32>
                %10 = stablehlo.compare  LE, %5, %9,  SIGNED : (tensor<1x4x1xi32>, tensor<1x4x1xi32>) -> tensor<1x4x1xi1>
                %11 = stablehlo.and %7, %10 : tensor<1x4x1xi1>
                %12 = stablehlo.reduce(%11 init: %c) applies stablehlo.and across dimensions = [2] : (tensor<1x4x1xi1>, tensor<i1>) -> tensor<1x4xi1>
                %13 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 768>}> : (tensor<2x768xf32>, tensor<1x4x1xi32>) -> tensor<1x4x768xf32>
                %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x4xi1>) -> tensor<1x4x768xi1>
                %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x768xf32>
                %16 = stablehlo.select %14, %13, %15 : tensor<1x4x768xi1>, tensor<1x4x768xf32>
                return %16 : tensor<1x4x768xf32>
            }
            func.func private @_where(%arg0: tensor<1x4xi1>, %arg1: tensor<1x4xi32>, %arg2: tensor<1x4xi32>) -> tensor<1x4xi32> {
                %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x4xi1>, tensor<1x4xi32>
                return %0 : tensor<1x4xi32>
            }
        }
    """

    splitter = StableHLOSplitter.create_from_module_str(shlo_module_str)
    sub_ops = splitter.get_sub_ops()
    sub_modules = splitter.get_sub_modules()

    assert len(sub_ops) == len(sub_modules) == 77, f"Splitter isn't working as expected"

    if print_results:
        for op in sub_ops:
            print(op)

        print()

        for m in sub_modules:
            print(str(m))


if __name__ == "__main__":
    test1()
    test2()
