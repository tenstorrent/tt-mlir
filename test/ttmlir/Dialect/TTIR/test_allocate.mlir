// RUN: ttmlir-opt --ttcore-register-device --ttir-allocate %s | FileCheck %s
// UNSUPPORTED: true
#l1_ = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32, #layout>, %arg1: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout> {
    // CHECK: = "ttir.alloc"
    // CHECK-NOT: = ttir.empty() : tensor<64x128xf32>
    %0 = ttir.empty() : tensor<64x128xf32, #layout>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout>
    return %1 : tensor<64x128xf32, #layout>
  }
}
