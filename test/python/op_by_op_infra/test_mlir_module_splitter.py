# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from ttmlir.mlir_module_splitter import MLIRModuleSplitter


@pytest.fixture
def shlo_module_str() -> str:
    return """
        module {
            func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
                %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
                %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
                %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
                return %2 : tensor<1x128xf32>
            }
        }
    """


@pytest.fixture
def multi_func_shlo_module_str() -> str:
    return """
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


@pytest.fixture
def ttir_module_str() -> str:
    return """
        module {
        func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
            %0 = ttir.empty() : tensor<1x128xf32>
            %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
            %2 = ttir.empty() : tensor<1x128xf32>
            %3 = "ttir.reshape"(%arg1, %2) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
            %4 = ttir.empty() : tensor<1x128xf32>
            %5 = "ttir.broadcast"(%3, %4) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
            %6 = ttir.empty() : tensor<1x128xf32>
            %7 = "ttir.add"(%1, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
            return %7 : tensor<1x128xf32>
        }
        }
    """


@pytest.fixture
def ttnn_module_str() -> str:
    return """
        #dram = #ttnn.buffer_type<dram>
        #system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 97248, erisc_l1_unreserved_base = 104992, dram_unreserved_base = 32, dram_unreserved_end = 1073158336, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
        #ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
        #ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
        module {
            tt.device_module {
                builtin.module attributes {tt.system_desc = #system_desc} {
                    tt.device @default_device = <workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
                    func.func @main(%arg0: tensor<1x128xf32, #ttnn_layout>, %arg1: tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout> {
                        %0 = "ttnn.reshape"(%arg1) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout>
                        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<128xf32, #ttnn_layout1>) -> ()
                        %1 = "ttnn.add"(%arg0, %0) : (tensor<1x128xf32, #ttnn_layout>, tensor<1x128xf32, #ttnn_layout>) -> tensor<1x128xf32, #ttnn_layout>
                        "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x128xf32, #ttnn_layout>) -> ()
                        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x128xf32, #ttnn_layout>) -> ()
                        return %1 : tensor<1x128xf32, #ttnn_layout>
                    }
                }
            }
        }
    """


def test_shlo_module_split(shlo_module_str: str):
    splitter = MLIRModuleSplitter()
    sub_modules = splitter.split(shlo_module_str)
    sub_ops = splitter.sub_ops

    assert len(sub_ops) == len(sub_modules) == 3


def test_multi_func_shlo_module_split(multi_func_shlo_module_str: str):
    splitter = MLIRModuleSplitter()
    sub_modules = splitter.split(multi_func_shlo_module_str)
    sub_ops = splitter.sub_ops

    assert len(sub_ops) == len(sub_modules) == 77


def test_ttir_module_split(ttir_module_str: str):
    splitter = MLIRModuleSplitter()
    sub_modules = splitter.split(ttir_module_str)
    sub_ops = splitter.sub_ops

    assert len(sub_ops) == len(sub_modules) == 8


def test_ttnn_module_split(ttnn_module_str: str):
    splitter = MLIRModuleSplitter()
    sub_modules = splitter.split(ttnn_module_str)
    sub_ops = splitter.sub_ops

    assert len(sub_ops) == len(sub_modules) == 5
