// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @test_clamp_scalar(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    // CHECK: = tosa.clamp %arg0 {max_val = 5.000000e+00 : f32, min_val = 2.000000e+00 : f32}
    %1 = "ttir.clamp_scalar"(%arg0) {min = 2.0 : f32, max = 5.0 : f32} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    // CHECK: return %{{[0-9]+}} : tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }

  func.func @test_clamp_scalar_negative_range(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: = tosa.clamp %arg0 {max_val = -1.000000e+00 : f32, min_val = -5.000000e+00 : f32}
    %1 = "ttir.clamp_scalar"(%arg0) {min = -5.0 : f32, max = -1.0 : f32} : (tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: return %{{[0-9]+}} : tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }
}
