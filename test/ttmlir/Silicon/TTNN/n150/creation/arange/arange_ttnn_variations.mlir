// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
#system_memory = #ttnn.buffer_type<system_memory>
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout_host_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128xbf16, #system_memory>>
#ttnn_layout_device_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128xbf16, #dram>, <interleaved>>
#ttnn_layout_device_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_l1_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
module attributes {} {

    // Test different arange op configurations
    //

    // Test case when arange is created on host
    func.func @arange_on_host() -> tensor<128xbf16, #ttnn_layout_host_rm_bf16> {
        // CHECK: ttnn.arange
        %0 = "ttnn.arange"() <{dtype = #ttcore.supportedDataTypes<bf16>, end = 128 : i64, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>, start = 0 : i64, step = 1 : i64}> : () -> tensor<128xbf16, #ttnn_layout_host_rm_bf16>
        return %0 : tensor<128xbf16, #ttnn_layout_host_rm_bf16>
    }

    // Test case when arange is created in DRAM in RowMajor layout
    func.func @arange_on_device_row_major() -> tensor<128xbf16, #ttnn_layout_device_rm_bf16> {
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        // CHECK: ttnn.arange
        %1 = "ttnn.arange"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, end = 128 : i64, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, start = 0 : i64, step = 1 : i64}> : (!ttnn.device) -> tensor<128xbf16, #ttnn_layout_device_rm_bf16>
        return %1 : tensor<128xbf16, #ttnn_layout_device_rm_bf16>
    }

    // Test case when arange is created in DRAM in Tile layout
    func.func @arange_on_device_tile() -> tensor<128xbf16, #ttnn_layout_device_tile_bf16> {
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        // CHECK: ttnn.arange
        %1 = "ttnn.arange"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, end = 128 : i64, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, start = 0 : i64, step = 1 : i64}> : (!ttnn.device) -> tensor<128xbf16, #ttnn_layout_device_tile_bf16>
        return %1 : tensor<128xbf16, #ttnn_layout_device_tile_bf16>
    }

    // Test case when arange is created in L1 in Tile layout
    func.func @arange_on_l1_tile() -> tensor<128xbf16, #ttnn_layout_l1_tile_bf16> {
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        // CHECK: ttnn.arange
        %1 = "ttnn.arange"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, end = 128 : i64, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <interleaved>>, start = 0 : i64, step = 1 : i64}> : (!ttnn.device) -> tensor<128xbf16, #ttnn_layout_l1_tile_bf16>
        return %1 : tensor<128xbf16, #ttnn_layout_l1_tile_bf16>
    }
}
