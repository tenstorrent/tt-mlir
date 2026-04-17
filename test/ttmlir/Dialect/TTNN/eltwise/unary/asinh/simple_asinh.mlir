// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @asinh_f32(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %1 = "ttir.asinh"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttnn.asinh"
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: -> tensor<64x128xf32
    return %1 : tensor<64x128xf32>
  }

  func.func @asinh_bf16(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %1 = "ttir.asinh"(%arg0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.asinh"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: -> tensor<64x128xbf16
    return %1 : tensor<64x128xbf16>
  }

  func.func @asinh_rank3(%arg0: tensor<2x64x128xf32>) -> tensor<2x64x128xf32> {
    %1 = "ttir.asinh"(%arg0) : (tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
    // CHECK: "ttnn.asinh"
    // CHECK-SAME: tensor<2x64x128xf32
    // CHECK-SAME: -> tensor<2x64x128xf32
    return %1 : tensor<2x64x128xf32>
  }

  func.func @asinh_rank4(%arg0: tensor<2x4x64x128xf32>) -> tensor<2x4x64x128xf32> {
    %1 = "ttir.asinh"(%arg0) : (tensor<2x4x64x128xf32>) -> tensor<2x4x64x128xf32>
    // CHECK: "ttnn.asinh"
    // CHECK-SAME: tensor<2x4x64x128xf32
    // CHECK-SAME: -> tensor<2x4x64x128xf32
    return %1 : tensor<2x4x64x128xf32>
  }
}
