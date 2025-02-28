// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device="force-reload=true" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99104, erisc_l1_unreserved_base = 104480, dram_unreserved_base = 32, dram_unreserved_end = 1073147200, physical_cores = {worker = [ 18x18,  18x19,  18x20,  18x21,  18x22,  18x23,  18x24,  18x25,  19x18,  19x19,  19x20,  19x21,  19x22,  19x23,  19x24,  19x25,  20x18,  20x19,  20x20,  20x21,  20x22,  20x23,  20x24,  20x25,  21x18,  21x19,  21x20,  21x21,  21x22,  21x23,  21x24,  21x25,  22x18,  22x19,  22x20,  22x21,  22x22,  22x23,  22x24,  22x25,  23x18,  23x19,  23x20,  23x21,  23x22,  23x23,  23x24,  23x25,  24x18,  24x19,  24x20,  24x21,  24x22,  24x23,  24x24,  24x25,  25x18,  25x19,  25x20,  25x21,  25x22,  25x23,  25x24,  25x25] dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<1024x64xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 3 + d2, d3), <1x1>, memref<12288x3xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x64xbf16, #system_memory>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 900 + d1 * 30 + d2, d3), <1x1>, memref<900x64xbf16, #system_memory>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 900 + d1 * 30 + d2, d3), <1x1>, memref<29x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 900 + d1 * 30 + d2, d3), <1x1>, memref<29x2x!tt.tile<32x32, bf16>, #system_memory>>

#conv2d_config = #ttnn.conv2d_config<
  dtype = bf16,
  weightsDtype = bf16,
  activation = "",
  inputChannelsAlignment = 32 : i32,
  deallocateActivation = false,
  reallocateHaloOutput = true,
  actBlockHOverride = 0 : i32,
  actBlockWDiv = 1 : i32,
  reshardIfNotOptimal = false,
  overrideShardingConfig = false,
  shardLayout = #ttnn.tensor_memory_layout<height_sharded>,
  transposeShards = true,
  outputLayout = tile,
  enableActDoubleBuffer = false,
  enableWeightsDoubleBuffer = false,
  enableSplitReader = false,
  enableSubblockPadding = false
>

module attributes {tt.device = #device, tt.system_desc = #system_desc} {
  func.func @forward(%arg0: tensor<1x32x32x64xbf16, #ttnn_layout>, %arg1: tensor<64x64x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x64xbf16, #ttnn_layout2>) -> tensor<1x30x30x64xbf16, #ttnn_layout3> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<29x2>>, <interleaved>>, shape = #ttnn.shape<1x1x900x64>}> : (!tt.device<#device>) -> tensor<1x1x900x64xbf16, #ttnn_layout4>
    %2 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %1, %0) <{batch_size = 1 : i32, conv2d_config = #conv2d_config, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x32x32x64xbf16, #ttnn_layout>, tensor<64x64x3x3xbf16, #ttnn_layout1>, tensor<1x1x1x64xbf16, #ttnn_layout2>, tensor<1x1x900x64xbf16, #ttnn_layout4>, !tt.device<#device>) -> tensor<1x1x900x64xbf16, #ttnn_layout4>
    %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16, #ttnn_layout4>) -> tensor<1x30x30x64xbf16, #ttnn_layout4>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x900x64xbf16, #ttnn_layout4>) -> ()
    %4 = "ttnn.from_device"(%3) : (tensor<1x30x30x64xbf16, #ttnn_layout4>) -> tensor<1x30x30x64xbf16, #ttnn_layout5>
    "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x30x30x64xbf16, #ttnn_layout4>) -> ()
    %5 = "ttnn.to_layout"(%4) <{layout = #ttnn.layout<row_major>}> : (tensor<1x30x30x64xbf16, #ttnn_layout5>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x30x30x64xbf16, #ttnn_layout5>) -> ()
    return %5 : tensor<1x30x30x64xbf16, #ttnn_layout3>
  }
}
