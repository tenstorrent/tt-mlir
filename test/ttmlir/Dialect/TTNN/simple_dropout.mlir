// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @dropout(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: "ttnn.dropout"
    // CHECK-SAME: prob = 2.000000e-01 : f32
    // CHECK-SAME: scale = 1.250000e+00 : f32
    // CHECK-SAME: seed = 21 : ui32
    // CHECK-SAME: use_per_device_seed = true
    %1 = "ttir.dropout"(%arg0) <{prob = 0.2 : f32, scale = 1.25 : f32, seed = 21 : ui32, use_per_device_seed = true}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
