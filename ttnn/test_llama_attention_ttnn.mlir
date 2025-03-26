#dram = #ttnn.buffer_type<dram>
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 102112, erisc_l1_unreserved_base = 104992, dram_unreserved_base = 32, dram_unreserved_end = 1073061056, physical_cores = {worker = [ 18x18,  18x19,  18x20,  18x21,  18x22,  18x23,  18x24,  18x25,  19x18,  19x19,  19x20,  19x21,  19x22,  19x23,  19x24,  19x25,  20x18,  20x19,  20x20,  20x21,  20x22,  20x23,  20x24,  20x25,  21x18,  21x19,  21x20,  21x21,  21x22,  21x23,  21x24,  21x25,  22x18,  22x19,  22x20,  22x21,  22x22,  22x23,  22x24,  22x25,  23x18,  23x19,  23x20,  23x21,  23x22,  23x23,  23x24,  23x25,  24x18,  24x19,  24x20,  24x21,  24x22,  24x23,  24x24,  24x25,  25x18,  25x19,  25x20,  25x21,  25x22,  25x23,  25x24,  25x25] dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 12 + d1, d2), <1x1>, memref<1x100x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 12 + d1 * 12 + d2, d3), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 50 + d1, d2), <1x1>, memref<2x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1600 + d1 * 50 + d2, d3), <1x1>, memref<50x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<100x100x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x100x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<12x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 12 + d2, d3), <1x1>, memref<12x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 12 + d1, d2), <1x1>, memref<1x2x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 12 + d1, d2), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 12 + d1 * 12 + d2, d3), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1600 + d1 * 50 + d2, d3), <1x1>, memref<50x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 12 + d2, d3), <1x1>, memref<12x2x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 12 + d1, d2), <1x1>, memref<12x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 12 + d1, d2), <1x1>, memref<12x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 12 + d2, d3), <1x1>, memref<12x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout18 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3200 + d1 * 100 + d2, d3), <1x1>, memref<100x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout19 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 100 + d1, d2), <1x1>, memref<100x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module {
  tt.device_module {
    builtin.module attributes {tt.system_desc = #system_desc} {
      tt.device @default_device = <workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @test_llama_attention(%arg0: tensor<1x12x3200xf32, #ttnn_layout>, %arg1: tensor<1x1x12x12xf32, #ttnn_layout1>, %arg2: tensor<1x12xf32, #ttnn_layout2>, %arg3: tensor<1x50x1xf32, #ttnn_layout3>, %arg4: tensor<1x32x50x100xf32, #ttnn_layout4>, %arg5: tensor<1x1xf32, #ttnn_layout2>, %arg6: tensor<1x32x50x100xf32, #ttnn_layout4>, %arg7: tensor<1x32x50x100xf32, #ttnn_layout4>, %arg8: tensor<1x1xf32, #ttnn_layout2>, %arg9: tensor<1x32x50x100xf32, #ttnn_layout4>, %arg10: tensor<1x1xf32, #ttnn_layout2>, %arg11: tensor<3200x3200xf32, #ttnn_layout5>, %arg12: tensor<3200x3200xf32, #ttnn_layout5>, %arg13: tensor<3200x3200xf32, #ttnn_layout5>, %arg14: tensor<3200x3200xf32, #ttnn_layout5>) -> tensor<1x12x3200xf32, #ttnn_layout> {
        %0 = "ttnn.reshape"(%arg0) <{shape = [12 : i32, 3200 : i32]}> : (tensor<1x12x3200xf32, #ttnn_layout>) -> tensor<12x3200xf32, #ttnn_layout6>
        %1 = "ttnn.matmul"(%0, %arg11) <{transpose_a = false, transpose_b = false}> : (tensor<12x3200xf32, #ttnn_layout6>, tensor<3200x3200xf32, #ttnn_layout5>) -> tensor<12x3200xf32, #ttnn_layout6>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32, #ttnn_layout6>) -> tensor<1x12x32x100xf32, #ttnn_layout7>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<12x3200xf32, #ttnn_layout6>) -> ()
        %3 = "ttnn.transpose"(%2) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x12x32x100xf32, #ttnn_layout7>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x12x32x100xf32, #ttnn_layout7>) -> ()
        %4 = "ttnn.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 12 : i32]}> : (tensor<1x12xf32, #ttnn_layout2>) -> tensor<1x1x12xf32, #ttnn_layout9>
        %5 = "ttnn.matmul"(%arg3, %4) <{transpose_a = false, transpose_b = false}> : (tensor<1x50x1xf32, #ttnn_layout3>, tensor<1x1x12xf32, #ttnn_layout9>) -> tensor<1x50x12xf32, #ttnn_layout3>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x12xf32, #ttnn_layout9>) -> ()
        %6 = "ttnn.transpose"(%5) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x50x12xf32, #ttnn_layout3>) -> tensor<1x12x50xf32, #ttnn_layout10>
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x50x12xf32, #ttnn_layout3>) -> ()
        %7 = "ttnn.concat"(%6, %6) <{dim = 2 : si32}> : (tensor<1x12x50xf32, #ttnn_layout10>, tensor<1x12x50xf32, #ttnn_layout10>) -> tensor<1x12x100xf32, #ttnn_layout11>
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x12x50xf32, #ttnn_layout10>) -> ()
        %8 = "ttnn.cos"(%7) : (tensor<1x12x100xf32, #ttnn_layout11>) -> tensor<1x12x100xf32, #ttnn_layout11>
        %9 = "ttnn.reshape"(%8) <{shape = [1 : i32, 1 : i32, 12 : i32, 100 : i32]}> : (tensor<1x12x100xf32, #ttnn_layout11>) -> tensor<1x1x12x100xf32, #ttnn_layout12>
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1x12x100xf32, #ttnn_layout11>) -> ()
        %10 = "ttnn.multiply"(%3, %9) : (tensor<1x32x12x100xf32, #ttnn_layout8>, tensor<1x1x12x100xf32, #ttnn_layout12>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        %11 = "ttnn.matmul"(%arg4, %3) <{transpose_a = false, transpose_b = true}> : (tensor<1x32x50x100xf32, #ttnn_layout4>, tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<1x32x50x12xf32, #ttnn_layout13>
        %12 = "ttnn.transpose"(%11) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x50x12xf32, #ttnn_layout13>) -> tensor<1x32x12x50xf32, #ttnn_layout14>
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1x32x50x12xf32, #ttnn_layout13>) -> ()
        %13 = "ttnn.multiply"(%12, %arg5) : (tensor<1x32x12x50xf32, #ttnn_layout14>, tensor<1x1xf32, #ttnn_layout2>) -> tensor<1x32x12x50xf32, #ttnn_layout14>
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<1x32x12x50xf32, #ttnn_layout14>) -> ()
        %14 = "ttnn.matmul"(%arg6, %3) <{transpose_a = false, transpose_b = true}> : (tensor<1x32x50x100xf32, #ttnn_layout4>, tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<1x32x50x12xf32, #ttnn_layout13>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        %15 = "ttnn.transpose"(%14) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x50x12xf32, #ttnn_layout13>) -> tensor<1x32x12x50xf32, #ttnn_layout14>
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<1x32x50x12xf32, #ttnn_layout13>) -> ()
        %16 = "ttnn.concat"(%13, %15) <{dim = 3 : si32}> : (tensor<1x32x12x50xf32, #ttnn_layout14>, tensor<1x32x12x50xf32, #ttnn_layout14>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%15) <{force = false}> : (tensor<1x32x12x50xf32, #ttnn_layout14>) -> ()
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1x32x12x50xf32, #ttnn_layout14>) -> ()
        %17 = "ttnn.sin"(%7) : (tensor<1x12x100xf32, #ttnn_layout11>) -> tensor<1x12x100xf32, #ttnn_layout11>
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x12x100xf32, #ttnn_layout11>) -> ()
        %18 = "ttnn.reshape"(%17) <{shape = [1 : i32, 1 : i32, 12 : i32, 100 : i32]}> : (tensor<1x12x100xf32, #ttnn_layout11>) -> tensor<1x1x12x100xf32, #ttnn_layout12>
        "ttnn.deallocate"(%17) <{force = false}> : (tensor<1x12x100xf32, #ttnn_layout11>) -> ()
        %19 = "ttnn.multiply"(%16, %18) : (tensor<1x32x12x100xf32, #ttnn_layout8>, tensor<1x1x12x100xf32, #ttnn_layout12>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%16) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        %20 = "ttnn.add"(%10, %19) : (tensor<1x32x12x100xf32, #ttnn_layout8>, tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%19) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        %21 = "ttnn.reshape"(%20) <{shape = [32 : i32, 12 : i32, 100 : i32]}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<32x12x100xf32, #ttnn_layout15>
        "ttnn.deallocate"(%20) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        %22 = "ttnn.matmul"(%0, %arg12) <{transpose_a = false, transpose_b = false}> : (tensor<12x3200xf32, #ttnn_layout6>, tensor<3200x3200xf32, #ttnn_layout5>) -> tensor<12x3200xf32, #ttnn_layout6>
        %23 = "ttnn.reshape"(%22) <{shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32, #ttnn_layout6>) -> tensor<1x12x32x100xf32, #ttnn_layout7>
        "ttnn.deallocate"(%22) <{force = false}> : (tensor<12x3200xf32, #ttnn_layout6>) -> ()
        %24 = "ttnn.transpose"(%23) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x12x32x100xf32, #ttnn_layout7>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%23) <{force = false}> : (tensor<1x12x32x100xf32, #ttnn_layout7>) -> ()
        %25 = "ttnn.multiply"(%24, %9) : (tensor<1x32x12x100xf32, #ttnn_layout8>, tensor<1x1x12x100xf32, #ttnn_layout12>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1x1x12x100xf32, #ttnn_layout12>) -> ()
        %26 = "ttnn.matmul"(%arg7, %24) <{transpose_a = false, transpose_b = true}> : (tensor<1x32x50x100xf32, #ttnn_layout4>, tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<1x32x50x12xf32, #ttnn_layout13>
        %27 = "ttnn.transpose"(%26) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x50x12xf32, #ttnn_layout13>) -> tensor<1x32x12x50xf32, #ttnn_layout14>
        "ttnn.deallocate"(%26) <{force = false}> : (tensor<1x32x50x12xf32, #ttnn_layout13>) -> ()
        %28 = "ttnn.multiply"(%27, %arg8) : (tensor<1x32x12x50xf32, #ttnn_layout14>, tensor<1x1xf32, #ttnn_layout2>) -> tensor<1x32x12x50xf32, #ttnn_layout14>
        "ttnn.deallocate"(%27) <{force = false}> : (tensor<1x32x12x50xf32, #ttnn_layout14>) -> ()
        %29 = "ttnn.matmul"(%arg9, %24) <{transpose_a = false, transpose_b = true}> : (tensor<1x32x50x100xf32, #ttnn_layout4>, tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<1x32x50x12xf32, #ttnn_layout13>
        "ttnn.deallocate"(%24) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        %30 = "ttnn.transpose"(%29) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x50x12xf32, #ttnn_layout13>) -> tensor<1x32x12x50xf32, #ttnn_layout14>
        "ttnn.deallocate"(%29) <{force = false}> : (tensor<1x32x50x12xf32, #ttnn_layout13>) -> ()
        %31 = "ttnn.concat"(%28, %30) <{dim = 3 : si32}> : (tensor<1x32x12x50xf32, #ttnn_layout14>, tensor<1x32x12x50xf32, #ttnn_layout14>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%30) <{force = false}> : (tensor<1x32x12x50xf32, #ttnn_layout14>) -> ()
        "ttnn.deallocate"(%28) <{force = false}> : (tensor<1x32x12x50xf32, #ttnn_layout14>) -> ()
        %32 = "ttnn.multiply"(%31, %18) : (tensor<1x32x12x100xf32, #ttnn_layout8>, tensor<1x1x12x100xf32, #ttnn_layout12>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%31) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        "ttnn.deallocate"(%18) <{force = false}> : (tensor<1x1x12x100xf32, #ttnn_layout12>) -> ()
        %33 = "ttnn.add"(%25, %32) : (tensor<1x32x12x100xf32, #ttnn_layout8>, tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%32) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        "ttnn.deallocate"(%25) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        %34 = "ttnn.reshape"(%33) <{shape = [32 : i32, 12 : i32, 100 : i32]}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<32x12x100xf32, #ttnn_layout15>
        "ttnn.deallocate"(%33) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        %35 = "ttnn.matmul"(%21, %34) <{transpose_a = false, transpose_b = true}> : (tensor<32x12x100xf32, #ttnn_layout15>, tensor<32x12x100xf32, #ttnn_layout15>) -> tensor<32x12x12xf32, #ttnn_layout16>
        "ttnn.deallocate"(%34) <{force = false}> : (tensor<32x12x100xf32, #ttnn_layout15>) -> ()
        "ttnn.deallocate"(%21) <{force = false}> : (tensor<32x12x100xf32, #ttnn_layout15>) -> ()
        %36 = "ttnn.reshape"(%35) <{shape = [1 : i32, 32 : i32, 12 : i32, 12 : i32]}> : (tensor<32x12x12xf32, #ttnn_layout16>) -> tensor<1x32x12x12xf32, #ttnn_layout17>
        "ttnn.deallocate"(%35) <{force = false}> : (tensor<32x12x12xf32, #ttnn_layout16>) -> ()
        %37 = "ttnn.multiply"(%36, %arg10) : (tensor<1x32x12x12xf32, #ttnn_layout17>, tensor<1x1xf32, #ttnn_layout2>) -> tensor<1x32x12x12xf32, #ttnn_layout17>
        "ttnn.deallocate"(%36) <{force = false}> : (tensor<1x32x12x12xf32, #ttnn_layout17>) -> ()
        %38 = "ttnn.add"(%37, %arg1) : (tensor<1x32x12x12xf32, #ttnn_layout17>, tensor<1x1x12x12xf32, #ttnn_layout1>) -> tensor<1x32x12x12xf32, #ttnn_layout17>
        "ttnn.deallocate"(%37) <{force = false}> : (tensor<1x32x12x12xf32, #ttnn_layout17>) -> ()
        %39 = "ttnn.softmax"(%38) <{dimension = -1 : si32}> : (tensor<1x32x12x12xf32, #ttnn_layout17>) -> tensor<1x32x12x12xf32, #ttnn_layout17>
        "ttnn.deallocate"(%38) <{force = false}> : (tensor<1x32x12x12xf32, #ttnn_layout17>) -> ()
        %40 = "ttnn.reshape"(%39) <{shape = [32 : i32, 12 : i32, 12 : i32]}> : (tensor<1x32x12x12xf32, #ttnn_layout17>) -> tensor<32x12x12xf32, #ttnn_layout16>
        "ttnn.deallocate"(%39) <{force = false}> : (tensor<1x32x12x12xf32, #ttnn_layout17>) -> ()
        %41 = "ttnn.matmul"(%0, %arg13) <{transpose_a = false, transpose_b = false}> : (tensor<12x3200xf32, #ttnn_layout6>, tensor<3200x3200xf32, #ttnn_layout5>) -> tensor<12x3200xf32, #ttnn_layout6>
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<12x3200xf32, #ttnn_layout6>) -> ()
        %42 = "ttnn.reshape"(%41) <{shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32, #ttnn_layout6>) -> tensor<1x12x32x100xf32, #ttnn_layout7>
        "ttnn.deallocate"(%41) <{force = false}> : (tensor<12x3200xf32, #ttnn_layout6>) -> ()
        %43 = "ttnn.transpose"(%42) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x12x32x100xf32, #ttnn_layout7>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%42) <{force = false}> : (tensor<1x12x32x100xf32, #ttnn_layout7>) -> ()
        %44 = "ttnn.transpose"(%43) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<1x32x100x12xf32, #ttnn_layout18>
        "ttnn.deallocate"(%43) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        %45 = "ttnn.reshape"(%44) <{shape = [32 : i32, 100 : i32, 12 : i32]}> : (tensor<1x32x100x12xf32, #ttnn_layout18>) -> tensor<32x100x12xf32, #ttnn_layout19>
        "ttnn.deallocate"(%44) <{force = false}> : (tensor<1x32x100x12xf32, #ttnn_layout18>) -> ()
        %46 = "ttnn.matmul"(%40, %45) <{transpose_a = false, transpose_b = true}> : (tensor<32x12x12xf32, #ttnn_layout16>, tensor<32x100x12xf32, #ttnn_layout19>) -> tensor<32x12x100xf32, #ttnn_layout15>
        "ttnn.deallocate"(%45) <{force = false}> : (tensor<32x100x12xf32, #ttnn_layout19>) -> ()
        "ttnn.deallocate"(%40) <{force = false}> : (tensor<32x12x12xf32, #ttnn_layout16>) -> ()
        %47 = "ttnn.reshape"(%46) <{shape = [1 : i32, 32 : i32, 12 : i32, 100 : i32]}> : (tensor<32x12x100xf32, #ttnn_layout15>) -> tensor<1x32x12x100xf32, #ttnn_layout8>
        "ttnn.deallocate"(%46) <{force = false}> : (tensor<32x12x100xf32, #ttnn_layout15>) -> ()
        %48 = "ttnn.transpose"(%47) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> tensor<1x12x32x100xf32, #ttnn_layout7>
        "ttnn.deallocate"(%47) <{force = false}> : (tensor<1x32x12x100xf32, #ttnn_layout8>) -> ()
        %49 = "ttnn.reshape"(%48) <{shape = [12 : i32, 3200 : i32]}> : (tensor<1x12x32x100xf32, #ttnn_layout7>) -> tensor<12x3200xf32, #ttnn_layout6>
        "ttnn.deallocate"(%48) <{force = false}> : (tensor<1x12x32x100xf32, #ttnn_layout7>) -> ()
        %50 = "ttnn.matmul"(%49, %arg14) <{transpose_a = false, transpose_b = false}> : (tensor<12x3200xf32, #ttnn_layout6>, tensor<3200x3200xf32, #ttnn_layout5>) -> tensor<12x3200xf32, #ttnn_layout6>
        "ttnn.deallocate"(%49) <{force = false}> : (tensor<12x3200xf32, #ttnn_layout6>) -> ()
        %51 = "ttnn.reshape"(%50) <{shape = [1 : i32, 12 : i32, 3200 : i32]}> : (tensor<12x3200xf32, #ttnn_layout6>) -> tensor<1x12x3200xf32, #ttnn_layout>
        "ttnn.deallocate"(%50) <{force = false}> : (tensor<12x3200xf32, #ttnn_layout6>) -> ()
        return %51 : tensor<1x12x3200xf32, #ttnn_layout>
      }
    }
  }
}
