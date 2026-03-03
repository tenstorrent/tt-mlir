// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Tests for conv2d weight and bias preparation with various data types and configurations.

module {
  // Test: bf16 conv2d without bias
  // Verifies prepare_conv2d_weights is generated for bf16 weights.
  func.func @conv2d_bf16_no_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK-LABEL: @conv2d_bf16_no_bias
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK: "ttnn.conv2d"
    %0 = "ttir.conv2d"(%arg0, %arg1) <{dilation = 1 : i32, groups = 1 : i32, padding = 0 : i32, stride = 1 : i32}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16>
    return %0 : tensor<1x30x30x64xbf16>
  }

  // Test: bf16 conv2d with bias as function argument
  // Verifies both prepare_conv2d_weights and prepare_conv2d_bias are generated.
  func.func @conv2d_bf16_with_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK-LABEL: @conv2d_bf16_with_bias
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK: "ttnn.prepare_conv2d_bias"
    // CHECK: "ttnn.conv2d"
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2) <{dilation = 1 : i32, groups = 1 : i32, padding = 0 : i32, stride = 1 : i32}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %0 : tensor<1x30x30x64xbf16>
  }

  // Test: f32 conv2d without bias
  // Verifies prepare_conv2d_weights is generated for f32 weights.
  func.func @conv2d_f32_no_bias(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x64x3x3xf32>) -> tensor<1x30x30x64xf32> {
    // CHECK-LABEL: @conv2d_f32_no_bias
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK: "ttnn.conv2d"
    %0 = "ttir.conv2d"(%arg0, %arg1) <{dilation = 1 : i32, groups = 1 : i32, padding = 0 : i32, stride = 1 : i32}> : (tensor<1x32x32x64xf32>, tensor<64x64x3x3xf32>) -> tensor<1x30x30x64xf32>
    return %0 : tensor<1x30x30x64xf32>
  }

  // Test: f32 conv2d with bias
  // Verifies both prepare_conv2d_weights and prepare_conv2d_bias are generated for f32.
  func.func @conv2d_f32_with_bias(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<1x1x1x64xf32>) -> tensor<1x30x30x64xf32> {
    // CHECK-LABEL: @conv2d_f32_with_bias
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK: "ttnn.prepare_conv2d_bias"
    // CHECK: "ttnn.conv2d"
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2) <{dilation = 1 : i32, groups = 1 : i32, padding = 0 : i32, stride = 1 : i32}> : (tensor<1x32x32x64xf32>, tensor<64x64x3x3xf32>, tensor<1x1x1x64xf32>) -> tensor<1x30x30x64xf32>
    return %0 : tensor<1x30x30x64xf32>
  }

  // Test: Mixed dtypes - f32 weights with bf16 input
  // Verifies weight dtype conversion is reflected in prepare_conv2d_weights attributes.
  func.func @conv2d_mixed_dtypes_f32_weights(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK-LABEL: @conv2d_mixed_dtypes_f32_weights
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK-SAME: weights_dtype = bf16
    // CHECK-SAME: input_dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.prepare_conv2d_bias"
    // CHECK-SAME: weights_dtype = bf16
    // CHECK-SAME: input_dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<bf16>
    %0 = "ttir.conv2d"(%arg0, %arg1)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xf32>) -> tensor<1x30x30x64xbf16>
    %1 = "ttir.add"(%0, %arg2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1: tensor<1x30x30x64xbf16>
  }

  // Test: Larger batch size
  // Verifies prepare_conv2d_weights works with batch size > 1.
  func.func @conv2d_bf16_batch16(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x30x30x64xbf16> {
    // CHECK-LABEL: @conv2d_bf16_batch16
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK: "ttnn.prepare_conv2d_bias"
    // CHECK: "ttnn.conv2d"
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
                stride = 1: i32,
                padding = 0: i32,
                dilation = 1: i32,
                groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<16x30x30x64xbf16>
    return %0 : tensor<16x30x30x64xbf16>
  }
}
