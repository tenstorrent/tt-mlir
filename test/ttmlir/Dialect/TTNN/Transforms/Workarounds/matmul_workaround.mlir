// RUN: ttmlir-opt --ttnn-workaround --canonicalize %s | FileCheck %s
#device = #tt.device<workerGrid = #tt.grid<7x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 7x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99104, erisc_l1_unreserved_base = 102400, dram_unreserved_base = 32, dram_unreserved_end = 1073196736, physical_cores = {worker = [ 18x18,  18x19,  18x20,  18x21,  18x22,  18x23,  18x24,  18x25,  19x18,  19x19,  19x20,  19x21,  19x22,  19x23,  19x24,  19x25,  20x18,  20x19,  20x20,  20x21,  20x22,  20x23,  20x24,  20x25,  21x18,  21x19,  21x20,  21x21,  21x22,  21x23,  21x24,  21x25,  22x18,  22x19,  22x20,  22x21,  22x22,  22x23,  22x24,  22x25,  23x18,  23x19,  23x20,  23x21,  23x22,  23x23,  23x24,  23x25,  24x18,  24x19,  24x20,  24x21,  24x22,  24x23,  24x24,  24x25] dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth = [ 17x25] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}, {arch = <wormhole_b0>, grid = 7x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99104, erisc_l1_unreserved_base = 102400, dram_unreserved_base = 32, dram_unreserved_end = 1073196736, physical_cores = {worker = [ 18x18,  18x19,  18x20,  18x21,  18x22,  18x23,  18x24,  18x25,  19x18,  19x19,  19x20,  19x21,  19x22,  19x23,  19x24,  19x25,  20x18,  20x19,  20x20,  20x21,  20x22,  20x23,  20x24,  20x25,  21x18,  21x19,  21x20,  21x21,  21x22,  21x23,  21x24,  21x25,  22x18,  22x19,  22x20,  22x21,  22x22,  22x23,  22x24,  22x25,  23x18,  23x19,  23x20,  23x21,  23x22,  23x23,  23x24,  23x25,  24x18,  24x19,  24x20,  24x21,  24x22,  24x23,  24x24,  24x25] dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth = [ 16x25] eth_inactive = [ 16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  17x18,  17x19,  17x20,  17x21,  17x22,  17x23,  17x24,  17x25]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0, 1], [3 : i32, 0 : i32], [ 0x0x0x0], [<[0, 8, 0], [1, 0, 0]>]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x96xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x96xbf16, #system_memory>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x96xbf16, #dram>, <interleaved>>
module attributes {tt.device = #device, tt.system_desc = #system_desc} {
  func.func @forward(%arg0: tensor<64x128xbf16, #ttnn_layout>, %arg1: tensor<128x96xbf16, #ttnn_layout1>) -> tensor<64x96xbf16, #ttnn_layout2> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %1 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <<2x3>>, <interleaved>>, shape = #ttnn.shape<64x96>}> : (!tt.device<#device>) -> tensor<64x96xbf16, #ttnn_layout3>
    // CHECK: %[[EMPTY_OP:.*]] = "ttnn.empty"(%[[DEVICE_OP]])
    // Check that the input operands are transformed into the tile layout.
    // CHECK-NEXT: %[[TO_LAYOUT_INPUT_A:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory, <<2x4>>>
    // CHECK-SAME: -> tensor<64x128xbf16
    // CHECK-NEXT: %[[TO_LAYOUT_INPUT_B:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory, <<4x3>>>
    // CHECK-SAME: -> tensor<128x96xbf16
    // Check that the output operand is transformed into the tile layout.
    // CHECK-NEXT: %[[TO_LAYOUT_OUTPUT_DPS:.*]] = "ttnn.to_layout"(%[[EMPTY_OP]], %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <<2x3>>, <interleaved>>
    // CHECK-SAME: -> tensor<64x96xbf16
    %2 = "ttnn.matmul"(%arg0, %arg1, %1) : (tensor<64x128xbf16, #ttnn_layout>, tensor<128x96xbf16, #ttnn_layout1>, tensor<64x96xbf16, #ttnn_layout3>) -> tensor<64x96xbf16, #ttnn_layout3>
    // CHECK-NEXT: %[[MATMUL_OP:.*]] = "ttnn.matmul"(%[[TO_LAYOUT_INPUT_A]], %[[TO_LAYOUT_INPUT_B]], %[[TO_LAYOUT_OUTPUT_DPS]])
    %3 = "ttnn.to_layout"(%2) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory, <<64x96>>>}> : (tensor<64x96xbf16, #ttnn_layout3>) -> tensor<64x96xbf16, #ttnn_layout2>
    // Check that the output operand is transformed back into the row_major layout.
    // CHECK-NEXT: %[[TO_LAYOUT_OUTPUT:.*]] = "ttnn.to_layout"(%[[MATMUL_OP]])
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory, <<64x96>>>
    // CHECK-SAME: -> tensor<64x96xbf16
    return %3 : tensor<64x96xbf16, #ttnn_layout2>
  }
}
