// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_host_rm_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
#ttnn_layout_host_rm_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xsi32, #system_memory>>
#ttnn_layout_host_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>
module attributes {} {
    // Test case to verify the folding of to_dtype operation.
    func.func @to_dtype_folding(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm_f32>) -> tensor<64x128xf32, #ttnn_layout_host_rm_f32> {
        // Verify that we fold the to_dtype when we try to cast to the same dtype.
        // CHECK: return %arg0 : tensor<64x128xf32, #ttnn_layout>
        %0 = "ttnn.to_dtype"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout_host_rm_f32>) -> tensor<64x128xf32, #ttnn_layout_host_rm_f32>
        return %0 : tensor<64x128xf32, #ttnn_layout_host_rm_f32>
    }

    // Test case to verify consecutive to_dtype op folding.
    func.func @to_dtype_folding_consecutive_typecasts(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm_f32>) -> tensor<64x128xbf16, #ttnn_layout_host_rm_bf16> {
        // Verify that we fold two consecutive typecast ops into a single one.
        // CHECK: ttnn.to_dtype
        // CHECK-NEXT: return
        %0 = "ttnn.to_dtype"(%arg0) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<64x128xf32, #ttnn_layout_host_rm_f32>) -> tensor<64x128xi32, #ttnn_layout_host_rm_si32>
        %1 = "ttnn.to_dtype"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xi32, #ttnn_layout_host_rm_si32>) -> tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>
        return %1 : tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>
    }

    // Test case to verify that we do not fold consecutive to_dtype ops if the first to_dtype have more than a single use.
    func.func @to_dtype_folding_consecutive_typecasts_with_multiple_uses(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm_f32>) -> (tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>, tensor<64x128xi32, #ttnn_layout_host_rm_si32>) {
        // Verify that both to_dtypes exists.
        // CHECK: ttnn.to_dtype
        // CHECK: ttnn.to_dtype
        %0 = "ttnn.to_dtype"(%arg0) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<64x128xf32, #ttnn_layout_host_rm_f32>) -> tensor<64x128xi32, #ttnn_layout_host_rm_si32>
        %1 = "ttnn.to_dtype"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xi32, #ttnn_layout_host_rm_si32>) -> tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>
        %2 = "ttnn.add"(%0, %0) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<64x128xi32, #ttnn_layout_host_rm_si32>, tensor<64x128xi32, #ttnn_layout_host_rm_si32>) -> tensor<64x128xi32, #ttnn_layout_host_rm_si32>
        return %1, %2 : tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>, tensor<64x128xi32, #ttnn_layout_host_rm_si32>
    }
}
