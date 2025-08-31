// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @has_output_dtype_trait_positive_test(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout> {
    %0 = "ttnn.multiply"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout>
    // CHECK: "ttnn.multiply"
    // CHECK-SAME: <{dtype = #ttcore.supportedDataTypes<f32>}>
    "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    return %0 : tensor<64x128xf32, #ttnn_layout>
  }
}
