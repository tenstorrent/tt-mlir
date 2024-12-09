// RUN: ttmlir-opt --ttnn-create-input-gens %s > %t.mlir

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 98816, erisc_l1_unreserved_base = 102624, dram_unreserved_base = 32, dram_unreserved_end = 1073083040, physical_cores = {worker = [ 1x1,  1x2,  1x3,  1x4,  1x6,  1x7,  1x8,  1x9,  2x1,  2x2,  2x3,  2x4,  2x6,  2x7,  2x8,  2x9,  3x1,  3x2,  3x3,  3x4,  3x6,  3x7,  3x8,  3x9,  4x1,  4x2,  4x3,  4x4,  4x6,  4x7,  4x8,  4x9,  5x1,  5x2,  5x3,  5x4,  5x6,  5x7,  5x8,  5x9,  7x1,  7x2,  7x3,  7x4,  7x6,  7x7,  7x8,  7x9,  8x1,  8x2,  8x3,  8x4,  8x6,  8x7,  8x8,  8x9,  9x1,  9x2,  9x3,  9x4,  9x6,  9x7,  9x8,  9x9] dram = [ 1x0,  1x5,  2x5,  3x5,  5x0,  5x5,  7x0,  7x5,  8x5,  9x5,  11x0,  11x5] eth_inactive = [ 0x1,  0x2,  0x3,  0x4,  0x6,  0x7,  0x8,  0x9,  6x2,  6x3,  6x6,  6x7,  6x8]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {tt.device = #device, tt.system_desc = #system_desc} {

// CHECK: func.func @add(%arg0: [[TENSOR_A:.*]], %arg1: [[TENSOR_B:.*]]) -> [[TENSOR_OUT:.*]] {
  func.func @add(%arg0: tensor<32x32xbf16, #ttnn_layout>, %arg1: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout>, !tt.device<#device>) -> tensor<32x32xbf16, #ttnn_layout1>
    %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32x32xbf16, #ttnn_layout1>) -> tensor<32x32xbf16, #ttnn_layout1>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<32x32xbf16, #ttnn_layout1>) -> ()
    %3 = "ttnn.to_device"(%arg1, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<32x32xbf16, #ttnn_layout>, !tt.device<#device>) -> tensor<32x32xbf16, #ttnn_layout1>
    %4 = "ttnn.to_layout"(%3) <{layout = #ttnn.layout<tile>}> : (tensor<32x32xbf16, #ttnn_layout1>) -> tensor<32x32xbf16, #ttnn_layout1>
    "ttnn.deallocate"(%3) <{force = false}> : (tensor<32x32xbf16, #ttnn_layout1>) -> ()
    %5 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!tt.device<#device>) -> tensor<32x32xbf16, #ttnn_layout1>
    %6 = "ttnn.add"(%2, %4, %5) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16, #ttnn_layout1>, tensor<32x32xbf16, #ttnn_layout1>, tensor<32x32xbf16, #ttnn_layout1>) -> tensor<32x32xbf16, #ttnn_layout1>
    "ttnn.deallocate"(%4) <{force = false}> : (tensor<32x32xbf16, #ttnn_layout1>) -> ()
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x32xbf16, #ttnn_layout1>) -> ()
    %7 = "ttnn.from_device"(%6) : (tensor<32x32xbf16, #ttnn_layout1>) -> tensor<32x32xbf16, #ttnn_layout>
    "ttnn.deallocate"(%5) <{force = false}> : (tensor<32x32xbf16, #ttnn_layout1>) -> ()
    %8 = "ttnn.to_layout"(%7) <{layout = #ttnn.layout<row_major>}> : (tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    "ttnn.deallocate"(%7) <{force = false}> : (tensor<32x32xbf16, #ttnn_layout>) -> ()
    return %8 : tensor<32x32xbf16, #ttnn_layout>
  }

// Confirm that the generator func is generated, and that the tensor attrs match:
//
// CHECK: func.func @createInputsFor_add() -> ([[TENSOR_A]], [[TENSOR_B]]) {
// CHECK: {{.*}} -> [[TENSOR_A]]
// CHECK: {{.*}} -> [[TENSOR_B]]

// Confirm that the main func is generated, and that the tensor attrs match:
//
// CHECK: func.func @main() -> i32 {
// CHECK: %0:2 = call @createInputsFor_add() : () -> ([[TENSOR_A]], [[TENSOR_B]])
// CHECK: %1 = call @add(%0#0, %0#1) : ([[TENSOR_A]], [[TENSOR_B]]) -> [[TENSOR_OUT]]
}
