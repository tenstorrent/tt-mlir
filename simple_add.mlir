#device = #tt.device<workerGrid = #tt.grid<7x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #tt.memory_space<dram>
#system = #tt.memory_space<system>
#system_desc = #tt.system_desc<[{arch = <wormhole_b0>, grid = 7x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 98816, erisc_l1_unreserved_base = 102624, dram_unreserved_base = 32, dram_unreserved_end = 1073198336, physical_cores = {worker = [ 2x1,  2x2,  2x3,  2x4,  2x6,  2x7,  2x8,  2x9,  3x1,  3x2,  3x3,  3x4,  3x6,  3x7,  3x8,  3x9,  4x1,  4x2,  4x3,  4x4,  4x6,  4x7,  4x8,  4x9,  5x1,  5x2,  5x3,  5x4,  5x6,  5x7,  5x8,  5x9,  7x1,  7x2,  7x3,  7x4,  7x6,  7x7,  7x8,  7x9,  8x1,  8x2,  8x3,  8x4,  8x6,  8x7,  8x8,  8x9,  9x1,  9x2,  9x3,  9x4,  9x6,  9x7,  9x8,  9x9] dram = [ 1x0,  1x5,  2x5,  3x5,  5x0,  5x5,  7x0,  7x5,  8x5,  9x5,  11x0,  11x5] eth = [ 6x9] eth_inactive = [ 0x1,  0x2,  0x3,  0x4,  0x6,  0x7,  0x8,  0x9,  6x2,  6x3,  6x6,  6x7,  6x8]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}, {arch = <wormhole_b0>, grid = 7x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 98816, erisc_l1_unreserved_base = 102624, dram_unreserved_base = 32, dram_unreserved_end = 1073198336, physical_cores = {worker = [ 1x1,  1x2,  1x3,  1x4,  1x6,  1x7,  1x8,  1x9,  2x1,  2x2,  2x3,  2x4,  2x6,  2x7,  2x8,  2x9,  3x1,  3x2,  3x3,  3x4,  3x6,  3x7,  3x8,  3x9,  4x1,  4x2,  4x3,  4x4,  4x6,  4x7,  4x8,  4x9,  5x1,  5x2,  5x3,  5x4,  5x6,  5x7,  5x8,  5x9,  7x1,  7x2,  7x3,  7x4,  7x6,  7x7,  7x8,  7x9,  8x1,  8x2,  8x3,  8x4,  8x6,  8x7,  8x8,  8x9] dram = [ 1x0,  1x5,  2x5,  3x5,  5x0,  5x5,  7x0,  7x5,  8x5,  9x5,  11x0,  11x5] eth = [ 0x9] eth_inactive = [ 0x2,  0x3,  0x4,  0x6,  0x7,  0x8,  6x1,  6x2,  6x3,  6x4,  6x6,  6x7,  6x8,  6x9]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0, 1], [3 : i32, 0 : i32], [ 0x0x0x0], [<[0, 8, 0], [1, 0, 0]>]>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #system>>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, interleaved>
#layout2 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #dram>, interleaved>
module attributes {tt.device = #device, tt.system_desc = #system_desc} {
  func.func @add(%arg0: tensor<64x128xf32, #layout>, %arg1: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !tt.device<#device>
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout1>
    %2 = "ttnn.to_device"(%1, %0) <{memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<1x1>>>}> : (tensor<64x128xf32, #layout1>, !tt.device<#device>) -> tensor<64x128xf32, #layout1>
    "ttnn.dealloc"(%1) : (tensor<64x128xf32, #layout1>) -> ()
    %3 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout1>
    %4 = "ttnn.to_device"(%3, %0) <{memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<1x1>>>}> : (tensor<64x128xf32, #layout1>, !tt.device<#device>) -> tensor<64x128xf32, #layout1>
    "ttnn.dealloc"(%3) : (tensor<64x128xf32, #layout1>) -> ()
    %5 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<interleaved>, <dram>, <<64x128>>>, shape = #ttnn.shape<64x128>}> : (!tt.device<#device>) -> tensor<64x128xf32, #layout2>
    %6 = "ttnn.add"(%2, %4, %5) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout2>) -> tensor<64x128xf32, #layout2>
    "ttnn.dealloc"(%4) : (tensor<64x128xf32, #layout1>) -> ()
    "ttnn.dealloc"(%2) : (tensor<64x128xf32, #layout1>) -> ()
    %7 = "ttnn.from_device"(%6) : (tensor<64x128xf32, #layout2>) -> tensor<64x128xf32, #layout>
    "ttnn.dealloc"(%5) : (tensor<64x128xf32, #layout2>) -> ()
    %8 = "ttnn.to_layout"(%7) <{layout = #ttnn.layout<row_major>}> : (tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout>
    "ttnn.dealloc"(%7) : (tensor<64x128xf32, #layout>) -> ()
    return %8 : tensor<64x128xf32, #layout>
  }
}
