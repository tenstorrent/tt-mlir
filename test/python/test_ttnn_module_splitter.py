# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ttnn_module_splitter import TTNNModuleSplitter


def test1():
    ttnn_module_str = """
    #device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
    #dram = #ttnn.buffer_type<dram>
    #system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
    #ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    #ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>

    module attributes {tt.device = #device, tt.system_desc = #system_desc} {
        func.func @main(%arg0: tensor<1x128xf32, #ttnn_layout>, %arg1: tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout> {
            %0 = "ttnn.reshape"(%arg1) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32, #ttnn_layout1>) -> tensor<1x128xf32, #ttnn_layout>
            %1 = "ttnn.add"(%arg0, %0) : (tensor<1x128xf32, #ttnn_layout>, tensor<1x128xf32, #ttnn_layout>) -> tensor<1x128xf32, #ttnn_layout>
            "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x128xf32, #ttnn_layout>) -> ()
            return %1 : tensor<1x128xf32, #ttnn_layout>
        }
    }
    """

    splitter: TTNNModuleSplitter = TTNNModuleSplitter.create_from_module_str(
        ttnn_module_str
    )

    for op in splitter.get_sub_ops():
        print(op)

    for m in splitter.get_sub_modules():
        print(str(m))


def test2():
    ttnn_module_str = """
    #device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
    #dram = #ttnn.buffer_type<dram>
    #system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
    #system_memory = #ttnn.buffer_type<system_memory>
    #ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, u32>, #dram>, <interleaved>>
    #ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x24x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    #ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x24x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    #ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<954x24x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    #ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 4 + d1, d2), <1x1>, memref<1x24x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    #ttnn_layout5 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xui32, #dram>, <interleaved>>
    #ttnn_layout6 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xui32, #system_memory>>
    #ttnn_layout7 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!tt.tile<32x32, u32>, #system_memory>>
    #ttnn_layout8 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!tt.tile<32x32, u32>, #dram>, <interleaved>>
    #ttnn_layout9 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xf32, #dram>, <interleaved>>
    #ttnn_layout10 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xf32, #system_memory>>
    #ttnn_layout11 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #system_memory>>
    #ttnn_layout12 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    #ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x4xui32, #dram>, <interleaved>>
    #ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x4xui32, #system_memory>>
    #ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x1x!tt.tile<32x32, u32>, #system_memory>>
    #ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x1x!tt.tile<32x32, u32>, #dram>, <interleaved>>
    #ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    #ttnn_layout18 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 4 + d1, d2), <1x1>, memref<1x1x!tt.tile<32x32, u32>, #dram>, <interleaved>>
    #ttnn_layout19 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x1x!tt.tile<32x32, u32>, #dram>, <interleaved>>
    #ttnn_layout20 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 4 + d1, d2), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    #ttnn_layout21 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, u32>, #system_memory>>
    #ttnn_layout22 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xui32, #system_memory>>
    #ttnn_layout23 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xui32, #dram>, <interleaved>>
    #ttnn_layout24 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<954x24x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    #ttnn_layout25 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 4 + d1, d2), <1x1>, memref<1x24x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    #ttnn_layout26 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    #ttnn_layout27 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 4 + d1, d2), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
    #ttnn_layout28 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x24x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    #ttnn_layout29 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x24x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    module attributes {tt.device = #device, tt.system_desc = #system_desc} {
    func.func public @main(%arg0: tensor<1x4xui32, #ttnn_layout>, %arg1: tensor<512x768xf32, #ttnn_layout1>, %arg2: tensor<2x768xf32, #ttnn_layout2>, %arg3: tensor<30522x768xf32, #ttnn_layout3>) -> tensor<1x4x768xf32, #ttnn_layout4> {
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.full"(%0) <{fillValue = 2.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1xui32, #ttnn_layout5>
        %2 = "ttnn.from_device"(%1) : (tensor<1xui32, #ttnn_layout5>) -> tensor<1xui32, #ttnn_layout6>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1xui32, #ttnn_layout5>) -> ()
        %3 = "ttnn.to_layout"(%2) <{layout = #ttnn.layout<tile>}> : (tensor<1xui32, #ttnn_layout6>) -> tensor<1xui32, #ttnn_layout7>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1xui32, #ttnn_layout6>) -> ()
        %4 = "ttnn.to_device"(%3, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<1xui32, #ttnn_layout7>, !tt.device<#device>) -> tensor<1xui32, #ttnn_layout8>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1xui32, #ttnn_layout7>) -> ()
        %5 = "ttnn.full"(%0) <{fillValue = 1.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1xui32, #ttnn_layout5>
        %6 = "ttnn.from_device"(%5) : (tensor<1xui32, #ttnn_layout5>) -> tensor<1xui32, #ttnn_layout6>
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1xui32, #ttnn_layout5>) -> ()
        %7 = "ttnn.to_layout"(%6) <{layout = #ttnn.layout<tile>}> : (tensor<1xui32, #ttnn_layout6>) -> tensor<1xui32, #ttnn_layout7>
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1xui32, #ttnn_layout6>) -> ()
        %8 = "ttnn.to_device"(%7, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<1xui32, #ttnn_layout7>, !tt.device<#device>) -> tensor<1xui32, #ttnn_layout8>
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1xui32, #ttnn_layout7>) -> ()
        %9 = "ttnn.full"(%0) <{fillValue = 5.120000e+02 : f32}> : (!tt.device<#device>) -> tensor<1xui32, #ttnn_layout5>
        %10 = "ttnn.from_device"(%9) : (tensor<1xui32, #ttnn_layout5>) -> tensor<1xui32, #ttnn_layout6>
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1xui32, #ttnn_layout5>) -> ()
        %11 = "ttnn.to_layout"(%10) <{layout = #ttnn.layout<tile>}> : (tensor<1xui32, #ttnn_layout6>) -> tensor<1xui32, #ttnn_layout7>
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1xui32, #ttnn_layout6>) -> ()
        %12 = "ttnn.to_device"(%11, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<1xui32, #ttnn_layout7>, !tt.device<#device>) -> tensor<1xui32, #ttnn_layout8>
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1xui32, #ttnn_layout7>) -> ()
        %13 = "ttnn.full"(%0) <{fillValue = 5.110000e+02 : f32}> : (!tt.device<#device>) -> tensor<1xui32, #ttnn_layout5>
        %14 = "ttnn.from_device"(%13) : (tensor<1xui32, #ttnn_layout5>) -> tensor<1xui32, #ttnn_layout6>
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1xui32, #ttnn_layout5>) -> ()
        %15 = "ttnn.to_layout"(%14) <{layout = #ttnn.layout<tile>}> : (tensor<1xui32, #ttnn_layout6>) -> tensor<1xui32, #ttnn_layout7>
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<1xui32, #ttnn_layout6>) -> ()
        %16 = "ttnn.to_device"(%15, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<1xui32, #ttnn_layout7>, !tt.device<#device>) -> tensor<1xui32, #ttnn_layout8>
        "ttnn.deallocate"(%15) <{force = false}> : (tensor<1xui32, #ttnn_layout7>) -> ()
        %17 = "ttnn.full"(%0) <{fillValue = 3.052200e+04 : f32}> : (!tt.device<#device>) -> tensor<1xui32, #ttnn_layout5>
        %18 = "ttnn.from_device"(%17) : (tensor<1xui32, #ttnn_layout5>) -> tensor<1xui32, #ttnn_layout6>
        "ttnn.deallocate"(%17) <{force = false}> : (tensor<1xui32, #ttnn_layout5>) -> ()
        %19 = "ttnn.to_layout"(%18) <{layout = #ttnn.layout<tile>}> : (tensor<1xui32, #ttnn_layout6>) -> tensor<1xui32, #ttnn_layout7>
        "ttnn.deallocate"(%18) <{force = false}> : (tensor<1xui32, #ttnn_layout6>) -> ()
        %20 = "ttnn.to_device"(%19, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<1xui32, #ttnn_layout7>, !tt.device<#device>) -> tensor<1xui32, #ttnn_layout8>
        "ttnn.deallocate"(%19) <{force = false}> : (tensor<1xui32, #ttnn_layout7>) -> ()
        %21 = "ttnn.full"(%0) <{fillValue = 3.052100e+04 : f32}> : (!tt.device<#device>) -> tensor<1xui32, #ttnn_layout5>
        %22 = "ttnn.from_device"(%21) : (tensor<1xui32, #ttnn_layout5>) -> tensor<1xui32, #ttnn_layout6>
        "ttnn.deallocate"(%21) <{force = false}> : (tensor<1xui32, #ttnn_layout5>) -> ()
        %23 = "ttnn.to_layout"(%22) <{layout = #ttnn.layout<tile>}> : (tensor<1xui32, #ttnn_layout6>) -> tensor<1xui32, #ttnn_layout7>
        "ttnn.deallocate"(%22) <{force = false}> : (tensor<1xui32, #ttnn_layout6>) -> ()
        %24 = "ttnn.to_device"(%23, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<1xui32, #ttnn_layout7>, !tt.device<#device>) -> tensor<1xui32, #ttnn_layout8>
        "ttnn.deallocate"(%23) <{force = false}> : (tensor<1xui32, #ttnn_layout7>) -> ()
        %25 = "ttnn.full"(%0) <{fillValue = 0x7FC00000 : f32}> : (!tt.device<#device>) -> tensor<1xf32, #ttnn_layout9>
        %26 = "ttnn.from_device"(%25) : (tensor<1xf32, #ttnn_layout9>) -> tensor<1xf32, #ttnn_layout10>
        "ttnn.deallocate"(%25) <{force = false}> : (tensor<1xf32, #ttnn_layout9>) -> ()
        %27 = "ttnn.to_layout"(%26) <{layout = #ttnn.layout<tile>}> : (tensor<1xf32, #ttnn_layout10>) -> tensor<1xf32, #ttnn_layout11>
        "ttnn.deallocate"(%26) <{force = false}> : (tensor<1xf32, #ttnn_layout10>) -> ()
        %28 = "ttnn.to_device"(%27, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<1xf32, #ttnn_layout11>, !tt.device<#device>) -> tensor<1xf32, #ttnn_layout12>
        "ttnn.deallocate"(%27) <{force = false}> : (tensor<1xf32, #ttnn_layout11>) -> ()
        %29 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1xui32, #ttnn_layout5>
        %30 = "ttnn.from_device"(%29) : (tensor<1xui32, #ttnn_layout5>) -> tensor<1xui32, #ttnn_layout6>
        "ttnn.deallocate"(%29) <{force = false}> : (tensor<1xui32, #ttnn_layout5>) -> ()
        %31 = "ttnn.to_layout"(%30) <{layout = #ttnn.layout<tile>}> : (tensor<1xui32, #ttnn_layout6>) -> tensor<1xui32, #ttnn_layout7>
        "ttnn.deallocate"(%30) <{force = false}> : (tensor<1xui32, #ttnn_layout6>) -> ()
        %32 = "ttnn.to_device"(%31, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<1xui32, #ttnn_layout7>, !tt.device<#device>) -> tensor<1xui32, #ttnn_layout8>
        "ttnn.deallocate"(%31) <{force = false}> : (tensor<1xui32, #ttnn_layout7>) -> ()
        %33 = "ttnn.reshape"(%32) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1xui32, #ttnn_layout>
        %34 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4xui32, #ttnn_layout>
        %35 = "ttnn.add"(%33, %34) : (tensor<1x1xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%34) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %36 = "ttnn.arange"(%0) <{dtype = #tt.supportedDataTypes<u32>, end = 4 : i64, memory_config = #ttnn.memory_config<#dram, <<1x4>>, <interleaved>>, start = 0 : i64, step = 1 : i64}> : (!tt.device<#device>) -> tensor<1x1x1x4xui32, #ttnn_layout13>
        %37 = "ttnn.from_device"(%36) : (tensor<1x1x1x4xui32, #ttnn_layout13>) -> tensor<1x1x1x4xui32, #ttnn_layout14>
        "ttnn.deallocate"(%36) <{force = false}> : (tensor<1x1x1x4xui32, #ttnn_layout13>) -> ()
        %38 = "ttnn.to_layout"(%37) <{layout = #ttnn.layout<tile>}> : (tensor<1x1x1x4xui32, #ttnn_layout14>) -> tensor<1x1x1x4xui32, #ttnn_layout15>
        "ttnn.deallocate"(%37) <{force = false}> : (tensor<1x1x1x4xui32, #ttnn_layout14>) -> ()
        %39 = "ttnn.to_device"(%38, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x1>>, <interleaved>>}> : (tensor<1x1x1x4xui32, #ttnn_layout15>, !tt.device<#device>) -> tensor<1x1x1x4xui32, #ttnn_layout16>
        "ttnn.deallocate"(%38) <{force = false}> : (tensor<1x1x1x4xui32, #ttnn_layout15>) -> ()
        %40 = "ttnn.reshape"(%39) <{shape = [4 : i32]}> : (tensor<1x1x1x4xui32, #ttnn_layout16>) -> tensor<4xui32, #ttnn_layout8>
        "ttnn.deallocate"(%39) <{force = false}> : (tensor<1x1x1x4xui32, #ttnn_layout16>) -> ()
        %41 = "ttnn.reshape"(%40) <{shape = [1 : i32, 4 : i32]}> : (tensor<4xui32, #ttnn_layout8>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%40) <{force = false}> : (tensor<4xui32, #ttnn_layout8>) -> ()
        %42 = "ttnn.reshape"(%32) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1xui32, #ttnn_layout>
        %43 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4xui32, #ttnn_layout>
        %44 = "ttnn.add"(%42, %43) : (tensor<1x1xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%43) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%42) <{force = false}> : (tensor<1x1xui32, #ttnn_layout>) -> ()
        %45 = "ttnn.lt"(%arg0, %44) : (tensor<1x4xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%44) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %46 = "ttnn.reshape"(%20) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1xui32, #ttnn_layout>
        "ttnn.deallocate"(%20) <{force = false}> : (tensor<1xui32, #ttnn_layout8>) -> ()
        %47 = "ttnn.add"(%arg0, %46) : (tensor<1x4xui32, #ttnn_layout>, tensor<1x1xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%46) <{force = false}> : (tensor<1x1xui32, #ttnn_layout>) -> ()
        %48 = "ttnn.typecast"(%45) <{dtype = #tt.supportedDataTypes<u32>}> : (tensor<1x4xbf16, #ttnn_layout17>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%45) <{force = false}> : (tensor<1x4xbf16, #ttnn_layout17>) -> ()
        %49 = "ttnn.where"(%48, %47, %arg0) : (tensor<1x4xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%48) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%47) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %50 = "ttnn.reshape"(%49) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4x1xui32, #ttnn_layout18>
        "ttnn.deallocate"(%49) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %51 = "ttnn.reshape"(%32) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1x1xui32, #ttnn_layout19>
        %52 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4x1xui32, #ttnn_layout18>
        %53 = "ttnn.add"(%51, %52) : (tensor<1x1x1xui32, #ttnn_layout19>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xui32, #ttnn_layout18>
        "ttnn.deallocate"(%52) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        "ttnn.deallocate"(%51) <{force = false}> : (tensor<1x1x1xui32, #ttnn_layout19>) -> ()
        %54 = "ttnn.ge"(%50, %53) : (tensor<1x4x1xui32, #ttnn_layout18>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%53) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        %55 = "ttnn.reshape"(%24) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1x1xui32, #ttnn_layout19>
        "ttnn.deallocate"(%24) <{force = false}> : (tensor<1xui32, #ttnn_layout8>) -> ()
        %56 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4x1xui32, #ttnn_layout18>
        %57 = "ttnn.add"(%55, %56) : (tensor<1x1x1xui32, #ttnn_layout19>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xui32, #ttnn_layout18>
        "ttnn.deallocate"(%56) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        "ttnn.deallocate"(%55) <{force = false}> : (tensor<1x1x1xui32, #ttnn_layout19>) -> ()
        %58 = "ttnn.le"(%50, %57) : (tensor<1x4x1xui32, #ttnn_layout18>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%57) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        %59 = "ttnn.logical_and"(%54, %58) : (tensor<1x4x1xbf16, #ttnn_layout20>, tensor<1x4x1xbf16, #ttnn_layout20>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%58) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        "ttnn.deallocate"(%54) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        %60 = "ttnn.prod"(%59) <{all_dimensions = false, dim_arg = 2 : i64, keep_dim = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> tensor<1x4xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%59) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        %61 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<1x24>>, <interleaved>>, shape = #ttnn.shape<1x4x768>}> : (!tt.device<#device>) -> tensor<1x4x768xf32, #ttnn_layout4>
        %62 = "ttnn.reshape"(%50) <{shape = [1 : i32, 4 : i32]}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%50) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        %63 = "ttnn.from_device"(%62) : (tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout21>
        "ttnn.deallocate"(%62) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %64 = "ttnn.to_layout"(%63) <{layout = #ttnn.layout<row_major>}> : (tensor<1x4xui32, #ttnn_layout21>) -> tensor<1x4xui32, #ttnn_layout22>
        "ttnn.deallocate"(%63) <{force = false}> : (tensor<1x4xui32, #ttnn_layout21>) -> ()
        %65 = "ttnn.to_device"(%64, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x4>>, <interleaved>>}> : (tensor<1x4xui32, #ttnn_layout22>, !tt.device<#device>) -> tensor<1x4xui32, #ttnn_layout23>
        "ttnn.deallocate"(%64) <{force = false}> : (tensor<1x4xui32, #ttnn_layout22>) -> ()
        %66 = "ttnn.typecast"(%arg3) <{dtype = #tt.supportedDataTypes<bf16>}> : (tensor<30522x768xf32, #ttnn_layout3>) -> tensor<30522x768xbf16, #ttnn_layout24>
        %67 = "ttnn.typecast"(%61) <{dtype = #tt.supportedDataTypes<bf16>}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xbf16, #ttnn_layout25>
        "ttnn.deallocate"(%61) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        %68 = "ttnn.embedding"(%65, %66, %67) : (tensor<1x4xui32, #ttnn_layout23>, tensor<30522x768xbf16, #ttnn_layout24>, tensor<1x4x768xbf16, #ttnn_layout25>) -> tensor<1x4x768xbf16, #ttnn_layout25>
        "ttnn.deallocate"(%66) <{force = false}> : (tensor<30522x768xbf16, #ttnn_layout24>) -> ()
        "ttnn.deallocate"(%65) <{force = false}> : (tensor<1x4xui32, #ttnn_layout23>) -> ()
        %69 = "ttnn.typecast"(%68) <{dtype = #tt.supportedDataTypes<f32>}> : (tensor<1x4x768xbf16, #ttnn_layout25>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%67) <{force = false}> : (tensor<1x4x768xbf16, #ttnn_layout25>) -> ()
        %70 = "ttnn.reshape"(%60) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xbf16, #ttnn_layout17>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%60) <{force = false}> : (tensor<1x4xbf16, #ttnn_layout17>) -> ()
        %71 = "ttnn.reshape"(%28) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32, #ttnn_layout12>) -> tensor<1x1x1xf32, #ttnn_layout26>
        %72 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4x768xf32, #ttnn_layout4>
        %73 = "ttnn.add"(%71, %72) : (tensor<1x1x1xf32, #ttnn_layout26>, tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%72) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        "ttnn.deallocate"(%71) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout26>) -> ()
        %74 = "ttnn.typecast"(%70) <{dtype = #tt.supportedDataTypes<f32>}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> tensor<1x4x1xf32, #ttnn_layout27>
        "ttnn.deallocate"(%70) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        %75 = "ttnn.where"(%74, %69, %73) : (tensor<1x4x1xf32, #ttnn_layout27>, tensor<1x4x768xf32, #ttnn_layout4>, tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%74) <{force = false}> : (tensor<1x4x1xf32, #ttnn_layout27>) -> ()
        "ttnn.deallocate"(%73) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        "ttnn.deallocate"(%69) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        %76 = "ttnn.reshape"(%32) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1xui32, #ttnn_layout>
        %77 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4xui32, #ttnn_layout>
        %78 = "ttnn.add"(%76, %77) : (tensor<1x1xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%77) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%76) <{force = false}> : (tensor<1x1xui32, #ttnn_layout>) -> ()
        %79 = "ttnn.lt"(%41, %78) : (tensor<1x4xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%78) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %80 = "ttnn.reshape"(%12) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1xui32, #ttnn_layout>
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<1xui32, #ttnn_layout8>) -> ()
        %81 = "ttnn.add"(%41, %80) : (tensor<1x4xui32, #ttnn_layout>, tensor<1x1xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%80) <{force = false}> : (tensor<1x1xui32, #ttnn_layout>) -> ()
        %82 = "ttnn.typecast"(%79) <{dtype = #tt.supportedDataTypes<u32>}> : (tensor<1x4xbf16, #ttnn_layout17>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%79) <{force = false}> : (tensor<1x4xbf16, #ttnn_layout17>) -> ()
        %83 = "ttnn.where"(%82, %81, %41) : (tensor<1x4xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%82) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%81) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%41) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %84 = "ttnn.reshape"(%83) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4x1xui32, #ttnn_layout18>
        "ttnn.deallocate"(%83) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %85 = "ttnn.reshape"(%32) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1x1xui32, #ttnn_layout19>
        %86 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4x1xui32, #ttnn_layout18>
        %87 = "ttnn.add"(%85, %86) : (tensor<1x1x1xui32, #ttnn_layout19>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xui32, #ttnn_layout18>
        "ttnn.deallocate"(%86) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        "ttnn.deallocate"(%85) <{force = false}> : (tensor<1x1x1xui32, #ttnn_layout19>) -> ()
        %88 = "ttnn.ge"(%84, %87) : (tensor<1x4x1xui32, #ttnn_layout18>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%87) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        %89 = "ttnn.reshape"(%16) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1x1xui32, #ttnn_layout19>
        "ttnn.deallocate"(%16) <{force = false}> : (tensor<1xui32, #ttnn_layout8>) -> ()
        %90 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4x1xui32, #ttnn_layout18>
        %91 = "ttnn.add"(%89, %90) : (tensor<1x1x1xui32, #ttnn_layout19>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xui32, #ttnn_layout18>
        "ttnn.deallocate"(%90) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        "ttnn.deallocate"(%89) <{force = false}> : (tensor<1x1x1xui32, #ttnn_layout19>) -> ()
        %92 = "ttnn.le"(%84, %91) : (tensor<1x4x1xui32, #ttnn_layout18>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%91) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        %93 = "ttnn.logical_and"(%88, %92) : (tensor<1x4x1xbf16, #ttnn_layout20>, tensor<1x4x1xbf16, #ttnn_layout20>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%92) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        "ttnn.deallocate"(%88) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        %94 = "ttnn.prod"(%93) <{all_dimensions = false, dim_arg = 2 : i64, keep_dim = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> tensor<1x4xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%93) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        %95 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<1x24>>, <interleaved>>, shape = #ttnn.shape<1x4x768>}> : (!tt.device<#device>) -> tensor<1x4x768xf32, #ttnn_layout4>
        %96 = "ttnn.reshape"(%84) <{shape = [1 : i32, 4 : i32]}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%84) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        %97 = "ttnn.from_device"(%96) : (tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout21>
        "ttnn.deallocate"(%96) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %98 = "ttnn.to_layout"(%97) <{layout = #ttnn.layout<row_major>}> : (tensor<1x4xui32, #ttnn_layout21>) -> tensor<1x4xui32, #ttnn_layout22>
        "ttnn.deallocate"(%97) <{force = false}> : (tensor<1x4xui32, #ttnn_layout21>) -> ()
        %99 = "ttnn.to_device"(%98, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x4>>, <interleaved>>}> : (tensor<1x4xui32, #ttnn_layout22>, !tt.device<#device>) -> tensor<1x4xui32, #ttnn_layout23>
        "ttnn.deallocate"(%98) <{force = false}> : (tensor<1x4xui32, #ttnn_layout22>) -> ()
        %100 = "ttnn.typecast"(%arg1) <{dtype = #tt.supportedDataTypes<bf16>}> : (tensor<512x768xf32, #ttnn_layout1>) -> tensor<512x768xbf16, #ttnn_layout28>
        %101 = "ttnn.typecast"(%95) <{dtype = #tt.supportedDataTypes<bf16>}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xbf16, #ttnn_layout25>
        "ttnn.deallocate"(%95) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        %102 = "ttnn.embedding"(%99, %100, %101) : (tensor<1x4xui32, #ttnn_layout23>, tensor<512x768xbf16, #ttnn_layout28>, tensor<1x4x768xbf16, #ttnn_layout25>) -> tensor<1x4x768xbf16, #ttnn_layout25>
        "ttnn.deallocate"(%100) <{force = false}> : (tensor<512x768xbf16, #ttnn_layout28>) -> ()
        "ttnn.deallocate"(%99) <{force = false}> : (tensor<1x4xui32, #ttnn_layout23>) -> ()
        %103 = "ttnn.typecast"(%102) <{dtype = #tt.supportedDataTypes<f32>}> : (tensor<1x4x768xbf16, #ttnn_layout25>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%101) <{force = false}> : (tensor<1x4x768xbf16, #ttnn_layout25>) -> ()
        %104 = "ttnn.reshape"(%94) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xbf16, #ttnn_layout17>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%94) <{force = false}> : (tensor<1x4xbf16, #ttnn_layout17>) -> ()
        %105 = "ttnn.reshape"(%28) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32, #ttnn_layout12>) -> tensor<1x1x1xf32, #ttnn_layout26>
        %106 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4x768xf32, #ttnn_layout4>
        %107 = "ttnn.add"(%105, %106) : (tensor<1x1x1xf32, #ttnn_layout26>, tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%106) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        "ttnn.deallocate"(%105) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout26>) -> ()
        %108 = "ttnn.typecast"(%104) <{dtype = #tt.supportedDataTypes<f32>}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> tensor<1x4x1xf32, #ttnn_layout27>
        "ttnn.deallocate"(%104) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        %109 = "ttnn.where"(%108, %103, %107) : (tensor<1x4x1xf32, #ttnn_layout27>, tensor<1x4x768xf32, #ttnn_layout4>, tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%108) <{force = false}> : (tensor<1x4x1xf32, #ttnn_layout27>) -> ()
        "ttnn.deallocate"(%107) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        "ttnn.deallocate"(%103) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        %110 = "ttnn.reshape"(%32) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1xui32, #ttnn_layout>
        %111 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4xui32, #ttnn_layout>
        %112 = "ttnn.add"(%110, %111) : (tensor<1x1xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%111) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%110) <{force = false}> : (tensor<1x1xui32, #ttnn_layout>) -> ()
        %113 = "ttnn.lt"(%35, %112) : (tensor<1x4xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%112) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %114 = "ttnn.reshape"(%4) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1xui32, #ttnn_layout>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1xui32, #ttnn_layout8>) -> ()
        %115 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4xui32, #ttnn_layout>
        %116 = "ttnn.add"(%114, %115) : (tensor<1x1xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%115) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%114) <{force = false}> : (tensor<1x1xui32, #ttnn_layout>) -> ()
        %117 = "ttnn.add"(%33, %116) : (tensor<1x1xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%116) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%33) <{force = false}> : (tensor<1x1xui32, #ttnn_layout>) -> ()
        %118 = "ttnn.typecast"(%113) <{dtype = #tt.supportedDataTypes<u32>}> : (tensor<1x4xbf16, #ttnn_layout17>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%113) <{force = false}> : (tensor<1x4xbf16, #ttnn_layout17>) -> ()
        %119 = "ttnn.where"(%118, %117, %35) : (tensor<1x4xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>, tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%118) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%117) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%35) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %120 = "ttnn.reshape"(%119) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4x1xui32, #ttnn_layout18>
        "ttnn.deallocate"(%119) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %121 = "ttnn.reshape"(%32) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1x1xui32, #ttnn_layout19>
        "ttnn.deallocate"(%32) <{force = false}> : (tensor<1xui32, #ttnn_layout8>) -> ()
        %122 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4x1xui32, #ttnn_layout18>
        %123 = "ttnn.add"(%121, %122) : (tensor<1x1x1xui32, #ttnn_layout19>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xui32, #ttnn_layout18>
        "ttnn.deallocate"(%122) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        "ttnn.deallocate"(%121) <{force = false}> : (tensor<1x1x1xui32, #ttnn_layout19>) -> ()
        %124 = "ttnn.ge"(%120, %123) : (tensor<1x4x1xui32, #ttnn_layout18>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%123) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        %125 = "ttnn.reshape"(%8) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xui32, #ttnn_layout8>) -> tensor<1x1x1xui32, #ttnn_layout19>
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1xui32, #ttnn_layout8>) -> ()
        %126 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4x1xui32, #ttnn_layout18>
        %127 = "ttnn.add"(%125, %126) : (tensor<1x1x1xui32, #ttnn_layout19>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xui32, #ttnn_layout18>
        "ttnn.deallocate"(%126) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        "ttnn.deallocate"(%125) <{force = false}> : (tensor<1x1x1xui32, #ttnn_layout19>) -> ()
        %128 = "ttnn.le"(%120, %127) : (tensor<1x4x1xui32, #ttnn_layout18>, tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%127) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        %129 = "ttnn.logical_and"(%124, %128) : (tensor<1x4x1xbf16, #ttnn_layout20>, tensor<1x4x1xbf16, #ttnn_layout20>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%128) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        "ttnn.deallocate"(%124) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        %130 = "ttnn.prod"(%129) <{all_dimensions = false, dim_arg = 2 : i64, keep_dim = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> tensor<1x4xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%129) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        %131 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<1x24>>, <interleaved>>, shape = #ttnn.shape<1x4x768>}> : (!tt.device<#device>) -> tensor<1x4x768xf32, #ttnn_layout4>
        %132 = "ttnn.reshape"(%120) <{shape = [1 : i32, 4 : i32]}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> tensor<1x4xui32, #ttnn_layout>
        "ttnn.deallocate"(%120) <{force = false}> : (tensor<1x4x1xui32, #ttnn_layout18>) -> ()
        %133 = "ttnn.from_device"(%132) : (tensor<1x4xui32, #ttnn_layout>) -> tensor<1x4xui32, #ttnn_layout21>
        "ttnn.deallocate"(%132) <{force = false}> : (tensor<1x4xui32, #ttnn_layout>) -> ()
        %134 = "ttnn.to_layout"(%133) <{layout = #ttnn.layout<row_major>}> : (tensor<1x4xui32, #ttnn_layout21>) -> tensor<1x4xui32, #ttnn_layout22>
        "ttnn.deallocate"(%133) <{force = false}> : (tensor<1x4xui32, #ttnn_layout21>) -> ()
        %135 = "ttnn.to_device"(%134, %0) <{memory_config = #ttnn.memory_config<#dram, <<1x4>>, <interleaved>>}> : (tensor<1x4xui32, #ttnn_layout22>, !tt.device<#device>) -> tensor<1x4xui32, #ttnn_layout23>
        "ttnn.deallocate"(%134) <{force = false}> : (tensor<1x4xui32, #ttnn_layout22>) -> ()
        %136 = "ttnn.typecast"(%arg2) <{dtype = #tt.supportedDataTypes<bf16>}> : (tensor<2x768xf32, #ttnn_layout2>) -> tensor<2x768xbf16, #ttnn_layout29>
        %137 = "ttnn.typecast"(%131) <{dtype = #tt.supportedDataTypes<bf16>}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xbf16, #ttnn_layout25>
        "ttnn.deallocate"(%131) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        %138 = "ttnn.embedding"(%135, %136, %137) : (tensor<1x4xui32, #ttnn_layout23>, tensor<2x768xbf16, #ttnn_layout29>, tensor<1x4x768xbf16, #ttnn_layout25>) -> tensor<1x4x768xbf16, #ttnn_layout25>
        "ttnn.deallocate"(%136) <{force = false}> : (tensor<2x768xbf16, #ttnn_layout29>) -> ()
        "ttnn.deallocate"(%135) <{force = false}> : (tensor<1x4xui32, #ttnn_layout23>) -> ()
        %139 = "ttnn.typecast"(%138) <{dtype = #tt.supportedDataTypes<f32>}> : (tensor<1x4x768xbf16, #ttnn_layout25>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%137) <{force = false}> : (tensor<1x4x768xbf16, #ttnn_layout25>) -> ()
        %140 = "ttnn.reshape"(%130) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x4xbf16, #ttnn_layout17>) -> tensor<1x4x1xbf16, #ttnn_layout20>
        "ttnn.deallocate"(%130) <{force = false}> : (tensor<1x4xbf16, #ttnn_layout17>) -> ()
        %141 = "ttnn.reshape"(%28) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32, #ttnn_layout12>) -> tensor<1x1x1xf32, #ttnn_layout26>
        "ttnn.deallocate"(%28) <{force = false}> : (tensor<1xf32, #ttnn_layout12>) -> ()
        %142 = "ttnn.full"(%0) <{fillValue = 0.000000e+00 : f32}> : (!tt.device<#device>) -> tensor<1x4x768xf32, #ttnn_layout4>
        %143 = "ttnn.add"(%141, %142) : (tensor<1x1x1xf32, #ttnn_layout26>, tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%142) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        "ttnn.deallocate"(%141) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout26>) -> ()
        %144 = "ttnn.typecast"(%140) <{dtype = #tt.supportedDataTypes<f32>}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> tensor<1x4x1xf32, #ttnn_layout27>
        "ttnn.deallocate"(%140) <{force = false}> : (tensor<1x4x1xbf16, #ttnn_layout20>) -> ()
        %145 = "ttnn.where"(%144, %139, %143) : (tensor<1x4x1xf32, #ttnn_layout27>, tensor<1x4x768xf32, #ttnn_layout4>, tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%144) <{force = false}> : (tensor<1x4x1xf32, #ttnn_layout27>) -> ()
        "ttnn.deallocate"(%143) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        "ttnn.deallocate"(%139) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        %146 = "ttnn.add"(%75, %145) : (tensor<1x4x768xf32, #ttnn_layout4>, tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%145) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        "ttnn.deallocate"(%75) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        %147 = "ttnn.add"(%146, %109) : (tensor<1x4x768xf32, #ttnn_layout4>, tensor<1x4x768xf32, #ttnn_layout4>) -> tensor<1x4x768xf32, #ttnn_layout4>
        "ttnn.deallocate"(%146) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        "ttnn.deallocate"(%109) <{force = false}> : (tensor<1x4x768xf32, #ttnn_layout4>) -> ()
        return %147 : tensor<1x4x768xf32, #ttnn_layout4>
    }
    }
    """

    splitter: TTNNModuleSplitter = TTNNModuleSplitter.create_from_module_str(
        ttnn_module_str
    )

    for m in splitter.get_sub_modules():
        print(str(m))


if __name__ == "__main__":
    test1()
    test2()
