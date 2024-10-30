// RUN: ttmlir-opt --ttnn-optimizer="memory-layout-analysis-enabled=true memreconfig-enabled=true insert-reshard=add_0_1_2=0" %s | FileCheck %s
// UNSUPPORTED: true
#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #tt.memory_space<dram>
#system = #tt.memory_space<system>
#system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32]}], [0], [3 : i32], [ 0x0x0x0]>
#layout = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <1x1>, memref<32x32xf32, #system>>
#layout1 = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <1x1>, memref<32x32xf32, #dram>, interleaved>
module attributes {tt.device = #device, tt.system_desc = #system_desc} {
  func.func @main(%arg0: tensor<1x32x32xf32, #layout>, %arg1: tensor<1x32x32xf32, #layout>, %arg2: tensor<1x32x32xf32, #layout>) -> tensor<1x32x32xf32, #layout> {
    // CHECK: #[[LAYOUT_1:.*]] = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <1x1>, memref<32x32xf32, #dram>, interleaved>
    // CHECK: #[[LAYOUT_2:.*]] = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <1x1>, memref<32x32xf32, #l1_>, block_sharded>
    // CHECK: #[[LAYOUT_3:.*]] = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <8x8>, memref<4x4xf32, #dram>, interleaved>
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_layout"(%arg0, %0) <{layout = #ttnn.layout<tile>}> : (tensor<1x32x32xf32, #layout>, !tt.device<#device>) -> tensor<1x32x32xf32, #layout1>
    %2 = "ttnn.to_device"(%1, %0) <{memory_config = #ttnn.memory_config<<interleaved>, <dram>>}> : (tensor<1x32x32xf32, #layout1>, !tt.device<#device>) -> tensor<1x32x32xf32, #layout1>
    %3 = "ttnn.to_layout"(%arg1, %0) <{layout = #ttnn.layout<tile>}> : (tensor<1x32x32xf32, #layout>, !tt.device<#device>) -> tensor<1x32x32xf32, #layout1>
    %4 = "ttnn.to_device"(%3, %0) <{memory_config = #ttnn.memory_config<<interleaved>, <dram>>}> : (tensor<1x32x32xf32, #layout1>, !tt.device<#device>) -> tensor<1x32x32xf32, #layout1>
    %5 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>>, shape = #ttnn.shape<1x32x32>}> : (!tt.device<#device>) -> tensor<1x32x32xf32, #layout1> loc(#loc1)
    // CHECK: %[[C:.*]] = "ttnn.add"{{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_2]]>
    %6 = "ttnn.add"(%2, %4, %5) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32, #layout1>, tensor<1x32x32xf32, #layout1>, tensor<1x32x32xf32, #layout1>) -> tensor<1x32x32xf32, #layout1> loc(#loc1)
    %7 = "ttnn.to_layout"(%arg0, %0) <{layout = #ttnn.layout<tile>}> : (tensor<1x32x32xf32, #layout>, !tt.device<#device>) -> tensor<1x32x32xf32, #layout1>
    %8 = "ttnn.to_device"(%7, %0) <{memory_config = #ttnn.memory_config<<interleaved>, <dram>>}> : (tensor<1x32x32xf32, #layout1>, !tt.device<#device>) -> tensor<1x32x32xf32, #layout1>
    %9 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>>, shape = #ttnn.shape<1x32x32>}> : (!tt.device<#device>) -> tensor<1x32x32xf32, #layout1> loc(#loc2)
    // CHECK: %{{.*}} = "ttnn.to_layout"(%[[C]], %0) {{.*}} -> tensor<1x32x32xf32, #[[LAYOUT_2]]>
    %10 = "ttnn.add"(%6, %8, %9) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x32xf32, #layout1>, tensor<1x32x32xf32, #layout1>, tensor<1x32x32xf32, #layout1>) -> tensor<1x32x32xf32, #layout1> loc(#loc2)
    %11 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>>, shape = #ttnn.shape<1x32x32>}> : (!tt.device<#device>) -> tensor<1x32x32xf32, #layout1> loc(#loc3)
    %12 = "ttnn.relu"(%10, %11) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x32x32xf32, #layout1>, tensor<1x32x32xf32, #layout1>) -> tensor<1x32x32xf32, #layout1> loc(#loc3)
    %13 = "ttnn.to_memory_config"(%12, %0) : (tensor<1x32x32xf32, #layout1>, !tt.device<#device>) -> tensor<1x32x32xf32, #layout>
    return %13 : tensor<1x32x32xf32, #layout>
  }
}
#loc1 = loc("add_1_2")
#loc2 = loc("add_0_1_2")
#loc3 = loc("relu")
