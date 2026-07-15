// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck --input-file=%t %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x128x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  // ones exponent broadcasts over the base, and the base shape matches the
  // result, so pow_tensor is dropped.
  // CHECK-LABEL: func.func @fold_pow_ones
  func.func @fold_pow_ones(%arg0: tensor<1x512x4096xf32, #ttnn_layout>) -> tensor<1x512x4096xf32, #ttnn_layout> {
    %0 = "ttnn.ones"() <{shape = #ttnn.shape<1x1x1>}> : () -> tensor<1x1x1xf32, #ttnn_layout1>
    // CHECK-NOT: "ttnn.pow_tensor"
    // CHECK: return %arg0
    %1 = "ttnn.pow_tensor"(%arg0, %0) : (tensor<1x512x4096xf32, #ttnn_layout>, tensor<1x1x1xf32, #ttnn_layout1>) -> tensor<1x512x4096xf32, #ttnn_layout>
    return %1 : tensor<1x512x4096xf32, #ttnn_layout>
  }

  // full(1.0) exponent is also folded.
  // CHECK-LABEL: func.func @fold_pow_full_one
  func.func @fold_pow_full_one(%arg0: tensor<1x512x4096xf32, #ttnn_layout>) -> tensor<1x512x4096xf32, #ttnn_layout> {
    %0 = "ttnn.full"() <{fill_value = 1.000000e+00 : f32, shape = #ttnn.shape<1x1x1>}> : () -> tensor<1x1x1xf32, #ttnn_layout1>
    // CHECK-NOT: "ttnn.pow_tensor"
    // CHECK: return %arg0
    %1 = "ttnn.pow_tensor"(%arg0, %0) : (tensor<1x512x4096xf32, #ttnn_layout>, tensor<1x1x1xf32, #ttnn_layout1>) -> tensor<1x512x4096xf32, #ttnn_layout>
    return %1 : tensor<1x512x4096xf32, #ttnn_layout>
  }

  // full(2.0) exponent is not one, so it is kept.
  // CHECK-LABEL: func.func @no_fold_pow_full_two
  func.func @no_fold_pow_full_two(%arg0: tensor<1x512x4096xf32, #ttnn_layout>) -> tensor<1x512x4096xf32, #ttnn_layout> {
    %0 = "ttnn.full"() <{fill_value = 2.000000e+00 : f32, shape = #ttnn.shape<1x1x1>}> : () -> tensor<1x1x1xf32, #ttnn_layout1>
    // CHECK: "ttnn.pow_tensor"
    %1 = "ttnn.pow_tensor"(%arg0, %0) : (tensor<1x512x4096xf32, #ttnn_layout>, tensor<1x1x1xf32, #ttnn_layout1>) -> tensor<1x512x4096xf32, #ttnn_layout>
    return %1 : tensor<1x512x4096xf32, #ttnn_layout>
  }
}
