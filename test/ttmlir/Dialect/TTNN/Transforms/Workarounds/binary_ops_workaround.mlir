// RUN: ttmlir-opt --ttnn-workaround --canonicalize %s | FileCheck %s

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!tt.tile<32x32, u16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!tt.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4xbf16, #dram>, <interleaved>>
module @jit_transpose attributes {tt.device = #device, tt.system_desc = #system_desc} {
  func.func public @test_add_workaround(%arg0: tensor<32x64xui16, #ttnn_layout>, %arg1: tensor<32x64xui16, #ttnn_layout>) -> tensor<32x64xui16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_add_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0,
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>,
    // CHECK-SAME: tensor<32x64xui16,
    // CHECK-SAME: -> tensor<32x64xbf16,
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"(%arg1,
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>,
    // CHECK-SAME: tensor<32x64xui16,
    // CHECK-SAME: -> tensor<32x64xbf16,
    // CHECK: %[[ADD:[0-9]+]] = "ttnn.add"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<32x64xbf16,
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x64xui16, #ttnn_layout>, tensor<32x64xui16, #ttnn_layout>) -> tensor<32x64xui16, #ttnn_layout>
    // CHECK: = "ttnn.to_layout"(%[[ADD]],
    // CHECK-SAME: dtype = #tt.supportedDataTypes<u16>,
    // CHECK-SAME:tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<32x64xui16,
    return %0 : tensor<32x64xui16, #ttnn_layout>
  }

  func.func public @test_multiply_workaround(%arg0: tensor<32x64xsi32, #ttnn_layout1>, %arg1: tensor<32x64xsi32, #ttnn_layout1>) -> tensor<32x64xsi32, #ttnn_layout1> {
    // CHECK-LABEL: func.func public @test_multiply_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0,
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>,
    // CHECK-SAME: tensor<32x64xsi32,
    // CHECK-SAME: -> tensor<32x64xbf16,
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"(%arg1,
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>,
    // CHECK-SAME: tensor<32x64xsi32,
    // CHECK-SAME: -> tensor<32x64xbf16,
    // CHECK: %[[MULTIPLY:[0-9]+]] = "ttnn.multiply"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<32x64xbf16,
    %0 = "ttnn.multiply"(%arg0, %arg1) : (tensor<32x64xsi32, #ttnn_layout1>, tensor<32x64xsi32, #ttnn_layout1>) -> tensor<32x64xsi32, #ttnn_layout1>
    // CHECK: = "ttnn.to_layout"(%[[MULTIPLY]]
    // CHECK-SAME: dtype = #tt.supportedDataTypes<si32>,
    // CHECK-SAME:tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<32x64xsi32,
    return %0 : tensor<32x64xsi32, #ttnn_layout1>
  }

  func.func public @test_bitwise_and_workaround(%arg0: tensor<32x64xf32, #ttnn_layout2>, %arg1: tensor<32x64xf32, #ttnn_layout2>) -> tensor<32x64xf32, #ttnn_layout2> {
    // CHECK-LABEL: func.func public @test_bitwise_and_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0,
    // CHECK-SAME: dtype = #tt.supportedDataTypes<si32>,
    // CHECK-SAME: tensor<32x64xf32,
    // CHECK-SAME: -> tensor<32x64xsi32
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"(%arg1,
    // CHECK-SAME: dtype = #tt.supportedDataTypes<si32>,
    // CHECK-SAME: tensor<32x64xf32,
    // CHECK-SAME: -> tensor<32x64xsi32
    // CHECK: %[[AND:[0-9]+]] = "ttnn.bitwise_and"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: tensor<32x64xsi32,
    // CHECK-SAME: tensor<32x64xsi32,
    // CHECK-SAME: -> tensor<32x64xsi32,
    %0 = "ttnn.bitwise_and"(%arg0, %arg1) : (tensor<32x64xf32, #ttnn_layout2>, tensor<32x64xf32, #ttnn_layout2>) -> tensor<32x64xf32, #ttnn_layout2>
    // CHECK: = "ttnn.to_layout"(%[[AND]],
    // CHECK-SAME: dtype = #tt.supportedDataTypes<f32>,
    // CHECK-SAME:tensor<32x64xsi32,
    // CHECK-SAME: -> tensor<32x64xf32,
    return %0 : tensor<32x64xf32, #ttnn_layout2>
  }

  func.func public @test_layout_workaround(%arg0: tensor<4x4xbf16, #ttnn_layout3>, %arg1: tensor<4x4xbf16, #ttnn_layout3>) -> tensor<4x4xbf16, #ttnn_layout3> {
    // CHECK-LABEL: func.func public @test_layout_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0,
    // CHECK-SAME: layout = #ttnn.layout<tile>,
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: -> tensor<4x4xbf16,
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"(%arg1,
    // CHECK-SAME: layout = #ttnn.layout<tile>,
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: -> tensor<4x4xbf16,
    // CHECK: %[[ADD:[0-9]+]] = "ttnn.add"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: -> tensor<4x4xbf16,
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<4x4xbf16, #ttnn_layout3>, tensor<4x4xbf16, #ttnn_layout3>) -> tensor<4x4xbf16, #ttnn_layout3>
    // CHECK: = "ttnn.to_layout"(%[[ADD]],
    // CHECK-SAME: layout = #ttnn.layout<row_major>,
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: -> tensor<4x4xbf16,
    return %0 : tensor<4x4xbf16, #ttnn_layout3>
  }
}
