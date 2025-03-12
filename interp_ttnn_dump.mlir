#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#loc1 = loc("arg0_1")
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102112, erisc_l1_unreserved_base = 104992, dram_unreserved_base = 32, dram_unreserved_end = 1073129152, physical_cores = {worker = [ 18x18,  18x19,  18x20,  18x21,  18x22,  18x23,  18x24,  18x25,  19x18,  19x19,  19x20,  19x21,  19x22,  19x23,  19x24,  19x25,  20x18,  20x19,  20x20,  20x21,  20x22,  20x23,  20x24,  20x25,  21x18,  21x19,  21x20,  21x21,  21x22,  21x23,  21x24,  21x25,  22x18,  22x19,  22x20,  22x21,  22x22,  22x23,  22x24,  22x25,  23x18,  23x19,  23x20,  23x21,  23x22,  23x23,  23x24,  23x25,  24x18,  24x19,  24x20,  24x21,  24x22,  24x23,  24x24,  24x25,  25x18,  25x19,  25x20,  25x21,  25x22,  25x23,  25x24,  25x25] dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 50 + d1 * 50 + d2, d3), <1x1>, memref<2x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 50 + d1, d2), <1x1>, memref<2x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 25 + d1 * 25 + d2, d3), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 50 + d1, d2), <1x1>, memref<2x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 50 + d1 * 50 + d2, d3), <1x1>, memref<2x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 25 + d1 * 25 + d2, d3), <1x1>, memref<1x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 25 + d1, d2), <1x1>, memref<1x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 25 + d1, d2), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  tt.device_module {
    builtin.module attributes {tt.device = #device, tt.system_desc = #system_desc} {
      func.func @main(%arg0: tensor<1x1x50x50xbf16, #ttnn_layout> loc("arg0_1"), %arg1: tensor<1x50x25xbf16, #ttnn_layout1> loc("arg0_1"), %arg2: tensor<1x50x25xbf16, #ttnn_layout1> loc("arg0_1")) -> tensor<1x1x25x25xbf16, #ttnn_layout2> {
        %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x1x50x50xbf16, #ttnn_layout>) -> tensor<1x1x50x50xbf16, #ttnn_layout> loc(#loc2)
        %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 50 : i32, 50 : i32]}> : (tensor<1x1x50x50xbf16, #ttnn_layout>) -> tensor<1x50x50xbf16, #ttnn_layout3> loc(#loc3)
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x1x50x50xbf16, #ttnn_layout>) -> () loc(#loc3)
        %2 = "ttnn.matmul"(%1, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x50x50xbf16, #ttnn_layout3>, tensor<1x50x25xbf16, #ttnn_layout1>) -> tensor<1x50x25xbf16, #ttnn_layout1> loc(#loc4)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x50x50xbf16, #ttnn_layout3>) -> () loc(#loc4)
        %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 1 : i32, 50 : i32, 25 : i32]}> : (tensor<1x50x25xbf16, #ttnn_layout1>) -> tensor<1x1x50x25xbf16, #ttnn_layout4> loc(#loc5)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x50x25xbf16, #ttnn_layout1>) -> () loc(#loc5)
        %4 = "ttnn.permute"(%3) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x1x50x25xbf16, #ttnn_layout4>) -> tensor<1x1x25x50xbf16, #ttnn_layout5> loc(#loc6)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x50x25xbf16, #ttnn_layout4>) -> () loc(#loc6)
        %5 = "ttnn.reshape"(%4) <{shape = [1 : i32, 25 : i32, 50 : i32]}> : (tensor<1x1x25x50xbf16, #ttnn_layout5>) -> tensor<1x25x50xbf16, #ttnn_layout6> loc(#loc7)
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x25x50xbf16, #ttnn_layout5>) -> () loc(#loc7)
        %6 = "ttnn.matmul"(%5, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x25x50xbf16, #ttnn_layout6>, tensor<1x50x25xbf16, #ttnn_layout1>) -> tensor<1x25x25xbf16, #ttnn_layout7> loc(#loc8)
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x25x50xbf16, #ttnn_layout6>) -> () loc(#loc8)
        %7 = "ttnn.reshape"(%6) <{shape = [1 : i32, 1 : i32, 25 : i32, 25 : i32]}> : (tensor<1x25x25xbf16, #ttnn_layout7>) -> tensor<1x1x25x25xbf16, #ttnn_layout2> loc(#loc9)
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x25x25xbf16, #ttnn_layout7>) -> () loc(#loc9)
        return %7 : tensor<1x1x25x25xbf16, #ttnn_layout2> loc(#loc10)
      } loc(#loc1)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("transpose_int")
#loc3 = loc("view_default")
#loc4 = loc("bmm_default")
#loc5 = loc("view_default_1")
#loc6 = loc("transpose_int_1")
#loc7 = loc("view_default_2")
#loc8 = loc("bmm_default_1")
#loc9 = loc("view_default_3")
#loc10 = loc("output")
