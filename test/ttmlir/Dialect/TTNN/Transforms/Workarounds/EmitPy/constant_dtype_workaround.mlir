// RUN: ttmlir-opt --ttnn-emitpy-workarounds -o %t %s
// RUN: FileCheck --input-file=%t %s

// Tests to verify emitpy constant op workarounds.

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout_f32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
    // Verify that when the value is float and dtype is float, no workaround is applied.
    func.func @test_constant_float_float() -> tensor<1x1x1x4xf32, #ttnn_layout_f32> {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]> : tensor<1x1x1x4xf32>}> : (!ttnn.device) -> tensor<1x1x1x4xf32, #ttnn_layout_f32>
        // CHECK: "ttnn.constant"
        // CHECK-SAME: value = dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]
        // CHECK-NOT: "ttnn.typecast"
        return %1 : tensor<1x1x1x4xf32, #ttnn_layout_f32>
    }

    // Verify that when the value is float and dtype is not float, the workaround is applied.
    func.func @test_constant_float_int32() -> tensor<1x1x1x4xsi32, #ttnn_layout_si32> {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]> : tensor<1x1x1x4xf32>}> : (!ttnn.device) -> tensor<1x1x1x4xsi32, #ttnn_layout_si32>
        // CHECK: "ttnn.constant"
        // CHECK-SAME: value = dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]
        // CHECK-SAME: -> tensor<1x1x1x4xf32
        // CHECK: "ttnn.typecast"
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
        // CHECK-SAME: -> tensor<1x1x1x4xsi32
        return %1 : tensor<1x1x1x4xsi32, #ttnn_layout_si32>
    }

    // Verify that when the value is int and dtype is int, the workaround is applied.
    func.func @test_constant_int32_int32() -> tensor<1x1x1x4xsi32, #ttnn_layout_si32> {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<[[[[1, 2, 3, 4]]]]> : tensor<1x1x1x4xsi32>}> : (!ttnn.device) -> tensor<1x1x1x4xsi32, #ttnn_layout_si32>
        // CHECK: "ttnn.constant"
        // CHECK-SAME: value = dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]
        // CHECK-SAME: -> tensor<1x1x1x4xf32
        // CHECK: "ttnn.typecast"
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
        // CHECK-SAME: -> tensor<1x1x1x4xsi32
        return %1 : tensor<1x1x1x4xsi32, #ttnn_layout_si32>
    }

    // Verify that when the value is int and dtype is bf16, the workaround is partially applied.
    // We need just to convert the value to f32, but the dtype remains bf16 and no cast is needed.
    func.func @test_constant_int32_bf16() -> tensor<1x1x1x4xbf16, #ttnn_layout_bf16> {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<[[[[1., 2., 3., 4.]]]]> : tensor<1x1x1x4xbf16>}> : (!ttnn.device) -> tensor<1x1x1x4xbf16, #ttnn_layout_bf16>
        // CHECK: "ttnn.constant"
        // CHECK-SAME: value = dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]
        // CHECK-SAME: -> tensor<1x1x1x4xbf16
        // CHECK-NOT: "ttnn.typecast"
        return %1 : tensor<1x1x1x4xbf16, #ttnn_layout_bf16>
    }

    // Verify that when the value is int and dtype is float, the workaround is not applied.
    func.func @test_constant_int32_float() -> tensor<1x1x1x4xf32, #ttnn_layout_f32> {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<[[[[1, 2, 3, 4]]]]> : tensor<1x1x1x4xsi32>}> : (!ttnn.device) -> tensor<1x1x1x4xf32, #ttnn_layout_f32>
        // CHECK: "ttnn.constant"
        // CHECK-SAME: value = dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]
        // CHECK-SAME: -> tensor<1x1x1x4xf32
        // CHECK-NOT: "ttnn.typecast"
        return %1 : tensor<1x1x1x4xf32, #ttnn_layout_f32>
    }
}
