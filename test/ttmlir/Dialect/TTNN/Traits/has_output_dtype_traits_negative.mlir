// RUN: not ttmlir-opt --split-input-file --ttcore-register-device="system-desc-path=%system_desc_path%" %s 2>&1 | FileCheck %s
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @has_output_dtype_trait_negative_test(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout> {
    %0 = "ttnn.multiply"(%arg0, %arg1) <{lhs_activations = [], output_dtype = #ttcore.supportedDataTypes<f32>, post_activations = [], rhs_activations = []}> : (tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout>
    "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    return %0 : tensor<64x128xf32, #ttnn_layout>
  }
}

// CHECK: error: 'ttnn.multiply' op Output tensor layout data type bf16 must match output data type attribute f32

// -----
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @has_output_dtype_trait_negative_test(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout> {
    %0 = "ttnn.multiply"(%arg0, %arg1) <{lhs_activations = [], post_activations = [], rhs_activations = []}> : (tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout>
    "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    return %0 : tensor<64x128xf32, #ttnn_layout>
  }
}

// CHECK: error: 'ttnn.multiply' op Output data type attribute is not defined for op that has output layout attribute.
