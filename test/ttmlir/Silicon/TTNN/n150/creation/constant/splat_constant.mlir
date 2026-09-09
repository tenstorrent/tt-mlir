
// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
#system_memory = #ttnn.buffer_type<system_memory>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_host_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #system_memory>>
#ttnn_layout_device_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {} {
    // Splat constant created on host (row-major).
    func.func @const_splat_host() -> tensor<32x32xbf16, #ttnn_layout_host_rm_bf16> {
        // CHECK-LABEL: func.func @const_splat_host
        // CHECK: ttnn.full
        // CHECK-NOT: ttnn.constant
        %0 = "ttnn.constant"() <{value = dense<2.000000e+00> : tensor<32x32xbf16>, layout = #ttnn.layout<row_major>}> : () -> tensor<32x32xbf16, #ttnn_layout_host_rm_bf16>
        return %0 : tensor<32x32xbf16, #ttnn_layout_host_rm_bf16>
    }

    // Splat constant created on device (tile layout) — the shape produced by
    // const-eval hoisting that triggered the original crash.
    func.func @const_splat_device_tile() -> tensor<32x32xbf16, #ttnn_layout_device_tile_bf16> {
        // CHECK-LABEL: func.func @const_splat_device_tile
        // CHECK: ttnn.full
        // CHECK-NOT: ttnn.constant
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{value = dense<2.000000e+00> : tensor<32x32xbf16>, layout = #ttnn.layout<tile>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout_device_tile_bf16>
        return %1 : tensor<32x32xbf16, #ttnn_layout_device_tile_bf16>
    }
}
