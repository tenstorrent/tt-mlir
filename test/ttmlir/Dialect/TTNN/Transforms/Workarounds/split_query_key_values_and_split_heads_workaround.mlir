// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module  {
  func.func @test_split_query_key_value_and_split_heads_segformer(%arg0: tensor<1x174x648xf32>) -> (tensor<1x4x174x54xf32>, tensor<1x4x54x174xf32>, tensor<1x4x174x54xf32>){
    // CHECK-LABEL: func.func @test_split_query_key_value_and_split_heads_segformer
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.permute"
    %query, %key, %value = "ttir.split_query_key_value_and_split_heads"(%arg0) <{num_heads = 4 : ui32, transpose_key = true}> : (tensor<1x174x648xf32>) -> (tensor<1x4x174x54xf32>, tensor<1x4x54x174xf32>, tensor<1x4x174x54xf32>)
    return %query, %key, %value : tensor<1x4x174x54xf32>, tensor<1x4x54x174xf32>, tensor<1x4x174x54xf32>
  }

  func.func @test_split_query_key_value_and_split_heads_stable_diffusion_unet(%arg0: tensor<1x4096x960xbf16>) -> (tensor<1x8x4096x40xbf16>, tensor<1x8x4096x40xbf16>, tensor<1x8x4096x40xbf16>) {
    // CHECK-LABEL: func.func @test_split_query_key_value_and_split_heads_stable_diffusion_unet
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.permute"
    %query, %key, %value = "ttir.split_query_key_value_and_split_heads"(%arg0) <{num_heads = 8 : ui32, transpose_key = false}> : (tensor<1x4096x960xbf16>) -> (tensor<1x8x4096x40xbf16>, tensor<1x8x4096x40xbf16>, tensor<1x8x4096x40xbf16>)
    return %query, %key, %value : tensor<1x8x4096x40xbf16>, tensor<1x8x4096x40xbf16>, tensor<1x8x4096x40xbf16>
  }

  func.func @test_split_query_key_value_and_split_heads_gqa(%arg0: tensor<1x10x320xbf16>, %arg1: tensor<1x10x320xbf16>) -> (tensor<1x8x10x40xbf16>, tensor<1x4x10x40xbf16>, tensor<1x4x10x40xbf16>) {
    // CHECK-LABEL: func.func @test_split_query_key_value_and_split_heads_gqa
    // Check input_tensor is reshaped and permuted correctly for query.
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    // Check kv_input_tensor is sliced, reshaped and permuted correctly for key and value.
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    %query, %key, %value = "ttir.split_query_key_value_and_split_heads"(%arg0, %arg1) <{num_heads = 8 : ui32, num_kv_heads = 4 : ui32, transpose_key = false}> : (tensor<1x10x320xbf16>, tensor<1x10x320xbf16>) -> (tensor<1x8x10x40xbf16>, tensor<1x4x10x40xbf16>, tensor<1x4x10x40xbf16>)
    return %query, %key, %value : tensor<1x8x10x40xbf16>, tensor<1x4x10x40xbf16>, tensor<1x4x10x40xbf16>
  }

  func.func @test_split_query_key_value_and_split_heads_gqa_transposed_key(%arg0: tensor<1x10x320xbf16>, %arg1: tensor<1x10x320xbf16>) -> (tensor<1x8x10x40xbf16>, tensor<1x4x40x10xbf16>, tensor<1x4x10x40xbf16>) {
    // CHECK-LABEL: func.func @test_split_query_key_value_and_split_heads_gqa_transposed_key
    // Check input_tensor is reshaped and permuted correctly for query.
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    // Check kv_input_tensor is sliced, reshaped and permuted correctly for key and value.
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.permute"
    %query, %key, %value = "ttir.split_query_key_value_and_split_heads"(%arg0, %arg1) <{num_heads = 8 : ui32, num_kv_heads = 4 : ui32, transpose_key = true}> : (tensor<1x10x320xbf16>, tensor<1x10x320xbf16>) -> (tensor<1x8x10x40xbf16>, tensor<1x4x40x10xbf16>, tensor<1x4x10x40xbf16>)
    return %query, %key, %value : tensor<1x8x10x40xbf16>, tensor<1x4x40x10xbf16>, tensor<1x4x10x40xbf16>
  }
}
