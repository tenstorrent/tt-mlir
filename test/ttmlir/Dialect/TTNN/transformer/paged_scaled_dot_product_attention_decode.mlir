// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @test_paged_attention_with_sliding_window(%arg0: tensor<1x1x12x64xf32>, %arg1: tensor<128x12x32x64xf32>, %arg2: tensor<128x12x32x64xf32>, %arg3: tensor<1x4xi32>, %arg4: tensor<1xi32>) -> tensor<1x1x12x64xf32> {
    // CHECK-LABEL: @test_paged_attention_with_sliding_window
    // CHECK: ttnn.paged_scaled_dot_product_attention_decode
    // CHECK-DAG: is_causal = true
    // CHECK-DAG: sliding_window_size = 512 : ui32
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>
    %0 = ttir.empty() : tensor<1x1x12x64xf32>
    %1 = "ttir.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %0, %arg4) <{is_causal = true, sliding_window_size = 512 : ui32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 0>}> : (tensor<1x1x12x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x64xf32>
    return %1 : tensor<1x1x12x64xf32>
  }

  func.func @test_paged_attention_without_sliding_window(%arg0: tensor<1x1x12x64xf32>, %arg1: tensor<128x12x32x64xf32>, %arg2: tensor<128x12x32x64xf32>, %arg3: tensor<1x4xi32>, %arg4: tensor<1xi32>) -> tensor<1x1x12x64xf32> {
    // CHECK-LABEL: @test_paged_attention_without_sliding_window
    // CHECK: ttnn.paged_scaled_dot_product_attention_decode
    // CHECK-DAG: is_causal = true
    // CHECK-NOT: sliding_window_size
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>
    %0 = ttir.empty() : tensor<1x1x12x64xf32>
    %1 = "ttir.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %0, %arg4) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 0>}> : (tensor<1x1x12x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x64xf32>
    return %1 : tensor<1x1x12x64xf32>
  }
}
