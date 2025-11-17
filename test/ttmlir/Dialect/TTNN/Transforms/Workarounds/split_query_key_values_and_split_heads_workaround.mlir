// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module  {
  func.func @test_split_query_key_value_and_split_heads_rewrite(%arg0: tensor<1x174x648xf32>) -> (tensor<1x4x174x54xf32>, tensor<1x4x54x174xf32>, tensor<1x4x174x54xf32>){
    %0 = ttir.empty() : tensor<1x4x174x54xf32>
    %1 = ttir.empty() : tensor<1x4x54x174xf32>
    %2 = ttir.empty() : tensor<1x4x174x54xf32>
    // CHECK-LABEL: func.func @test_split_query_key_value_and_split_heads_rewrite
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.permute"
    %query, %key, %value = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 4 : ui32, transpose_key = true}> : (tensor<1x174x648xf32>, tensor<1x4x174x54xf32>, tensor<1x4x54x174xf32>, tensor<1x4x174x54xf32>) -> (tensor<1x4x174x54xf32>, tensor<1x4x54x174xf32>, tensor<1x4x174x54xf32>)
    return %query, %key, %value : tensor<1x4x174x54xf32>, tensor<1x4x54x174xf32>, tensor<1x4x174x54xf32>
  }
}
