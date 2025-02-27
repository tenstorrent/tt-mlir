# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.mlir_module_splitter import MLIRModuleSplitter


def test1(print_results: bool = False):
    """Tests that splitter works with stablehlo dialect."""
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

    splitter = MLIRModuleSplitter()
    sub_modules = splitter.split(shlo_module_str)
    sub_ops = splitter.sub_ops

    assert len(sub_ops) == len(sub_modules) == 3, f"Splitter isn't working as expected"

    if print_results:
        for op in sub_ops:
            print(op)

        print()

        for m in sub_modules:
            print(str(m))


def test2(print_results: bool = False):
    """Tests that splitter works with stablehlo dialect and a multi-func module."""
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

    splitter = MLIRModuleSplitter()
    sub_modules = splitter.split(shlo_module_str)
    sub_ops = splitter.sub_ops

    assert len(sub_ops) == len(sub_modules) == 77, f"Splitter isn't working as expected"

    if print_results:
        for op in sub_ops:
            print(op)

        print()

        for m in sub_modules:
            print(str(m))


def test3(print_results: bool = False):
    """Tests that splitter works with ttir dialect."""
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

    splitter = MLIRModuleSplitter()
    sub_modules = splitter.split(ttir_module_str)
    sub_ops = splitter.sub_ops

    assert len(sub_ops) == len(sub_modules) == 8, f"Splitter isn't working as expected"

    if print_results:
        for op in sub_ops:
            print(op)

        print()

        for m in sub_modules:
            print(str(m))


def test4(print_results: bool = False):
    """Tests that splitter works with ttnn dialect."""
    ttnn_module_str = """
        #device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
        #system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
        #dram = #ttnn.buffer_type<dram>
        #ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
        #ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
        module attributes {tt.device = #device, tt.system_desc = #system_desc} {
            func.func @main(%arg0: tensor<1x128xf32, #ttnn_layout>, %arg1: tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout> {
                %0 = "ttnn.reshape"(%arg1) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout>
                %1 = "ttnn.add"(%arg0, %0) : (tensor<1x128xf32, #ttnn_layout>, tensor<1x128xf32, #ttnn_layout>) -> tensor<1x128xf32, #ttnn_layout>
                "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x128xf32, #ttnn_layout>) -> ()
                return %1 : tensor<1x128xf32, #ttnn_layout>
            }
        }
    """

    splitter = MLIRModuleSplitter()
    sub_modules = splitter.split(ttnn_module_str)
    sub_ops = splitter.sub_ops

    assert len(sub_ops) == len(sub_modules) == 3, f"Splitter isn't working as expected"

    if print_results:
        for op in sub_ops:
            print(op)

        print()

        for m in sub_modules:
            print(str(m))


if __name__ == "__main__":
    test1(True)
    # test2()
    # test3()
    # test4()
