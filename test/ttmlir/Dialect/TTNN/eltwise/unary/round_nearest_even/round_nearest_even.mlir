// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module attributes {} {
  func.func @round_nearest_even_4d_f32(%arg0: tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32> {
    // CHECK-LABEL: func.func @round_nearest_even_4d_f32
    // CHECK: %[[VAL_0:[0-9]+]] = "ttnn.round_nearest_even"(%arg0)
    // CHECK-SAME: -> tensor<1x32x32x64xf32, #ttnn.layout<
    %0 = "ttir.round_nearest_even"(%arg0): (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    return %0 : tensor<1x32x32x64xf32>
  }

  func.func @round_nearest_even_2d_f16(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    // CHECK-LABEL: func.func @round_nearest_even_2d_f16
    // CHECK: %[[VAL_0:[0-9]+]] = "ttnn.round_nearest_even"(%arg0)
    // CHECK-SAME: -> tensor<8x16xf16, #ttnn.layout<
    %0 = "ttir.round_nearest_even"(%arg0): (tensor<8x16xf16>) -> tensor<8x16xf16>
    return %0 : tensor<8x16xf16>
  }

  func.func @round_nearest_even_1d(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    // CHECK-LABEL: func.func @round_nearest_even_1d
    // CHECK: %[[VAL_0:[0-9]+]] = "ttnn.round_nearest_even"(%arg0)
    // CHECK-SAME: -> tensor<128xf32, #ttnn.layout<
    %0 = "ttir.round_nearest_even"(%arg0): (tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
}
