// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @upsample2d_scale_uniform(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x128x3xbf16> {
    %0 = ttir.empty() : tensor<4x64x128x3xbf16>
    // CHECK: "ttnn.upsample"
    // CHECK-SAME: tensor<4x32x64x3xbf16
    // CHECK-SAME: tensor<4x64x128x3xbf16
    %1 = "ttir.upsample2d"(%arg0, %0) <{scale_factor = 2 : si32, mode = "nearest"}> : (tensor<4x32x64x3xbf16>, tensor<4x64x128x3xbf16>) -> tensor<4x64x128x3xbf16>
    return %1 : tensor<4x64x128x3xbf16>
  }

  func.func @upsample2d_scale_nonuniform(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x64x3xbf16> {
    %0 = ttir.empty() : tensor<4x64x64x3xbf16>
    // CHECK: "ttnn.upsample"
    // CHECK-SAME: tensor<4x32x64x3xbf16
    // CHECK-SAME: tensor<4x64x64x3xbf16
    %1 = "ttir.upsample2d"(%arg0, %0) <{scale_factor = array<i32: 2, 1>, mode = "nearest"}> : (tensor<4x32x64x3xbf16>, tensor<4x64x64x3xbf16>) -> tensor<4x64x64x3xbf16>
    return %1 : tensor<4x64x64x3xbf16>
  }

  func.func @upsample2d_scale_nonuniform_bilinear(%arg0: tensor<4x32x64x32xbf16>) -> tensor<4x96x128x32xbf16> {
    %0 = ttir.empty() : tensor<4x96x128x32xbf16>
    %1 = "ttir.upsample2d"(%arg0, %0) <{scale_factor = array<i32: 3, 2>, mode = "bilinear"}> : (tensor<4x32x64x32xbf16>, tensor<4x96x128x32xbf16>) -> tensor<4x96x128x32xbf16>
    return %1 : tensor<4x96x128x32xbf16>
  }

  func.func @upsample2d_scale_uniform_bilinear(%arg0: tensor<4x32x64x32xbf16>) -> tensor<4x96x192x32xbf16> {
    %0 = ttir.empty() : tensor<4x96x192x32xbf16>
    %1 = "ttir.upsample2d"(%arg0, %0) <{scale_factor = 3 : si32, mode = "bilinear"}> : (tensor<4x32x64x32xbf16>, tensor<4x96x192x32xbf16>) -> tensor<4x96x192x32xbf16>
    return %1 : tensor<4x96x192x32xbf16>
  }

  func.func @upsample2d_scale_uniform_bilinear_misaligned_channel(%arg0: tensor<4x32x64x37xbf16>) -> tensor<4x96x192x37xbf16> {
    %0 = ttir.empty() : tensor<4x96x192x37xbf16>
    %1 = "ttir.upsample2d"(%arg0, %0) <{scale_factor = 3 : si32, mode = "bilinear"}> : (tensor<4x32x64x37xbf16>, tensor<4x96x192x37xbf16>) -> tensor<4x96x192x37xbf16>
    return %1 : tensor<4x96x192x37xbf16>
  }
}
