// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
// Test that non-bf16 inputs to conv3d are automatically converted to bf16

module {
  func.func public @test_conv3d_f32_to_bf16(%arg0: tensor<1x8x28x28x4xf32>, %arg1: tensor<16x4x3x3x3xf32>) -> tensor<1x6x26x26x16xf32> {
    // CHECK-LABEL: func.func public @test_conv3d_f32_to_bf16
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.conv3d"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              padding_mode = "zeros"
            }> : (tensor<1x8x28x28x4xf32>, tensor<16x4x3x3x3xf32>) -> tensor<1x6x26x26x16xf32>
    return %0 : tensor<1x6x26x26x16xf32>
  }

  func.func public @test_conv3d_f32_to_bf16_with_bias(%arg0: tensor<1x8x28x28x4xf32>, %arg1: tensor<16x4x3x3x3xf32>, %arg2: tensor<1x1x1x1x16xf32>) -> tensor<1x6x26x26x16xf32> {
    // CHECK-LABEL: func.func public @test_conv3d_f32_to_bf16_with_bias
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.conv3d"
    %0 = "ttir.conv3d"(%arg0, %arg1, %arg2)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              padding_mode = "zeros"
            }> : (tensor<1x8x28x28x4xf32>, tensor<16x4x3x3x3xf32>, tensor<1x1x1x1x16xf32>) -> tensor<1x6x26x26x16xf32>
    return %0 : tensor<1x6x26x26x16xf32>
  }

  func.func public @test_conv3d_bf16_no_dtype_workaround(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<16x4x3x3x3xbf16>) -> tensor<1x6x26x26x16xbf16> {
    // CHECK-LABEL: func.func public @test_conv3d_bf16_no_dtype_workaround
    // CHECK: "ttnn.conv3d"
    %0 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              padding_mode = "zeros"
            }> : (tensor<1x8x28x28x4xbf16>, tensor<16x4x3x3x3xbf16>) -> tensor<1x6x26x26x16xbf16>
    return %0 : tensor<1x6x26x26x16xbf16>
  }
}
