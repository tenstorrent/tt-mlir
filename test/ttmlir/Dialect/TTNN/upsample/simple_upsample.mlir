// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @upsample2d_scale_unifrom(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x128x3xbf16> {
    %0 = tensor.empty() : tensor<4x64x128x3xbf16>
    // CHECK: "ttnn.upsample"
    // CHECK-SAME: tensor<4x32x64x3xbf16
    // CHECK-SAME: tensor<4x64x128x3xbf16
    %1 = "ttir.upsample2d"(%arg0, %0) <{scale_factor = 2 : si32}> : (tensor<4x32x64x3xbf16>, tensor<4x64x128x3xbf16>) -> tensor<4x64x128x3xbf16>
    return %1 : tensor<4x64x128x3xbf16>
  }

  func.func @upsample2d_scale_nonunifrom(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x64x3xbf16> {
    %0 = tensor.empty() : tensor<4x64x64x3xbf16>
    // CHECK: "ttnn.upsample"
    // CHECK-SAME: tensor<4x32x64x3xbf16
    // CHECK-SAME: tensor<4x64x64x3xbf16
    %1 = "ttir.upsample2d"(%arg0, %0) <{scale_factor = array<i32: 2, 1>}> : (tensor<4x32x64x3xbf16>, tensor<4x64x64x3xbf16>) -> tensor<4x64x64x3xbf16>
    return %1 : tensor<4x64x64x3xbf16>
  }
}
