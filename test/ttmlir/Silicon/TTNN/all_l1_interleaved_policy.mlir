// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttnn-optimizer="memory-layout-analysis-enabled=true memory-layout-analysis-policy=L1Interleaved" --ttnn-decompose-layouts --ttnn-deallocate %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #tt.memory_space<dram>
#system = #tt.memory_space<system>
#system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xbf16, #system>>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<128x96xbf16, #system>>
#layout2 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x96xbf16, #system>>
#layout3 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<96x32xbf16, #system>>
#layout4 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x32xbf16, #system>>
#layout5 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, interleaved>
#layout6 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x96xbf16, #dram>, interleaved>
#layout7 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x32xbf16, #dram>, interleaved>
module attributes {tt.device = #device, tt.system_desc = #system_desc} {
  func.func @forward(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x96xbf16, #layout1>, %arg2: tensor<64x96xbf16, #layout2>, %arg3: tensor<96x32xbf16, #layout3>, %arg4: tensor<64x32xbf16, #layout4>) -> tensor<64x32xbf16, #layout4> {
    // CHECK: #[[L1_:.*]] = #tt.memory_space<l1>
    // CHECK: #[[LAYOUT_6:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x12xbf16, #l1_>, interleaved>
    // CHECK: #[[LAYOUT_7:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x4xbf16, #l1_>, interleaved>
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<1x1>>>}> : (tensor<64x128xbf16, #layout>, !tt.device<#device>) -> tensor<64x128xbf16, #layout5>
    %2 = "ttnn.to_layout"(%arg1, %0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<1x1>>>}> : (tensor<128x96xbf16, #layout1>, !tt.device<#device>) -> tensor<128x96xbf16, #layout5>
    %3 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x96>>>, shape = #ttnn.shape<64x96>}> : (!tt.device<#device>) -> tensor<64x96xbf16, #layout6>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_6]]>
    %4 = "ttnn.matmul"(%1, %2, %3) : (tensor<64x128xbf16, #layout5>, tensor<128x96xbf16, #layout5>, tensor<64x96xbf16, #layout6>) -> tensor<64x96xbf16, #layout6>
    %5 = "ttnn.to_layout"(%arg2, %0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<1x1>>>}> : (tensor<64x96xbf16, #layout2>, !tt.device<#device>) -> tensor<64x96xbf16, #layout5>
    %6 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x96>>>, shape = #ttnn.shape<64x96>}> : (!tt.device<#device>) -> tensor<64x96xbf16, #layout6>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_6]]>
    %7 = "ttnn.add"(%4, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x96xbf16, #layout6>, tensor<64x96xbf16, #layout5>, tensor<64x96xbf16, #layout6>) -> tensor<64x96xbf16, #layout6>
    %8 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x96>>>, shape = #ttnn.shape<64x96>}> : (!tt.device<#device>) -> tensor<64x96xbf16, #layout6>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x96xbf16, #[[LAYOUT_6]]>
    %9 = "ttnn.relu"(%7, %8) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x96xbf16, #layout6>, tensor<64x96xbf16, #layout6>) -> tensor<64x96xbf16, #layout6>
    %10 = "ttnn.to_layout"(%arg3, %0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<1x1>>>}> : (tensor<96x32xbf16, #layout3>, !tt.device<#device>) -> tensor<96x32xbf16, #layout5>
    %11 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x32>>>, shape = #ttnn.shape<64x32>}> : (!tt.device<#device>) -> tensor<64x32xbf16, #layout7>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_7]]>
    %12 = "ttnn.matmul"(%9, %10, %11) : (tensor<64x96xbf16, #layout6>, tensor<96x32xbf16, #layout5>, tensor<64x32xbf16, #layout7>) -> tensor<64x32xbf16, #layout7>
    %13 = "ttnn.to_layout"(%arg4, %0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<1x1>>>}> : (tensor<64x32xbf16, #layout4>, !tt.device<#device>) -> tensor<64x32xbf16, #layout5>
    %14 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x32>>>, shape = #ttnn.shape<64x32>}> : (!tt.device<#device>) -> tensor<64x32xbf16, #layout7>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_7]]>
    %15 = "ttnn.add"(%12, %13, %14) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x32xbf16, #layout7>, tensor<64x32xbf16, #layout5>, tensor<64x32xbf16, #layout7>) -> tensor<64x32xbf16, #layout7>
    %16 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x32>>>, shape = #ttnn.shape<64x32>}> : (!tt.device<#device>) -> tensor<64x32xbf16, #layout7>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_7]]>
    %17 = "ttnn.relu"(%15, %16) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x32xbf16, #layout7>, tensor<64x32xbf16, #layout7>) -> tensor<64x32xbf16, #layout7>
    %18 = "ttnn.to_layout"(%17) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<none>, <system_memory>, <<64x32>>>}> : (tensor<64x32xbf16, #layout7>) -> tensor<64x32xbf16, #layout4>
    return %18 : tensor<64x32xbf16, #layout4>
  }
}
