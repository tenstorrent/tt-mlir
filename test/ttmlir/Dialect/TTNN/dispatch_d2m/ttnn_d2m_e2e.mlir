// RUN: ttmlir-opt --ttcore-register-device --ttcore-wrap-device-module --ttnn-d2m-fusing --ttnn-through-d2m-pipeline --ttnn-collaspe-d2m --canonicalize %s | FileCheck %s


#l1 = #ttnn.buffer_type<l1>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>>
module {
  // CHECK-LABEL: func.func @two_independent_chains
  func.func @two_independent_chains(%arg0: tensor<64x128xbf16, #layout>, %arg1: tensor<128x256xbf16, #layout>, %arg2: tensor<64x128xbf16, #layout>, %arg3: tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout> {
    // CHECK: %[[MM1:.*]] = "ttnn.matmul"
    // CHECK: %[[EMPTY1:.*]] = "ttnn.empty"
    // CHECK: "ttnn.generic"(%[[MM1]], %[[EMPTY1]])
    // CHECK-SAME: symbol_ref = @datamovement_kernel2
    // CHECK-SAME: symbol_ref = @compute_kernel3
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %1 = "ttnn.exp"(%0) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %2 = "ttnn.log"(%1) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %3 = "ttnn.neg"(%2) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    // CHECK: %[[MM2:.*]] = "ttnn.matmul"
    // CHECK: %[[EMPTY2:.*]] = "ttnn.empty"
    // CHECK: "ttnn.generic"(%[[MM2]], %[[EMPTY2]])
    // CHECK-SAME: symbol_ref = @datamovement_kernel0
    // CHECK-SAME: symbol_ref = @compute_kernel1
    %4 = "ttnn.matmul"(%arg2, %arg3) : (tensor<64x128xbf16, #layout>, tensor<128x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %5 = "ttnn.abs"(%4) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>
    %6 = "ttnn.sigmoid"(%5) : (tensor<64x256xbf16, #layout>) -> tensor<64x256xbf16, #layout>

    return %6 : tensor<64x256xbf16, #layout>
  }
  // CHECK: func.func private @datamovement_kernel0
  // CHECK: func.func private @compute_kernel1
  // CHECK: func.func private @datamovement_kernel2
  // CHECK: func.func private @compute_kernel3
}
