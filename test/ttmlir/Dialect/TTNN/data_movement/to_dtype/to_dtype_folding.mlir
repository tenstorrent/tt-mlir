// RUN: ttmlir-opt --canonicalize %s | FileCheck %s
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_host_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
module attributes {} {
    // Test case to verify the folding of to_dtype operation.
    func.func @from_host_to_host_f32_to_bf16_rm(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_host_rm> {
        // Verify that we fold the to_dtype when we try to cast to the same dtype.
        // CHECK: return %arg0 : tensor<64x128xf32, #ttnn_layout>
        %0 = "ttnn.to_dtype"(%arg0) <{dtype = #tt.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_host_rm>
        return %0 : tensor<64x128xf32, #ttnn_layout_host_rm>
    }
}
