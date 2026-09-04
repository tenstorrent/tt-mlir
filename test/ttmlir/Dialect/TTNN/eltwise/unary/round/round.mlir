// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module attributes {} {
  func.func @round_4d_f32(%arg0: tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32> {
    // CHECK-LABEL: func.func @round_4d_f32
    // CHECK: %[[VAL_0:[0-9]+]] = "ttnn.round"(%arg0)
    %0 = "ttir.round"(%arg0): (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    return %0 : tensor<1x32x32x64xf32>
  }

  func.func @round_2d_f16(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    // CHECK-LABEL: func.func @round_2d_f16
    // CHECK: %[[VAL_0:[0-9]+]] = "ttnn.round"(%arg0)
    %0 = "ttir.round"(%arg0): (tensor<8x16xf16>) -> tensor<8x16xf16>
    return %0 : tensor<8x16xf16>
  }

  func.func @round_1d(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    // CHECK-LABEL: func.func @round_1d
    // CHECK: %[[VAL_0:[0-9]+]] = "ttnn.round"(%arg0)
    %0 = "ttir.round"(%arg0): (tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
}
