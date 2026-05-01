// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @argmax_last_dim_bf16
  // CHECK-NOT: "ttir.argmax"
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>]
  // CHECK: d2m.argmax
  // CHECK-SAME: <128, 96>
  // CHECK-NOT: d2m.tile_reduce_max
  func.func @argmax_last_dim_bf16(%arg: tensor<128x96xbf16>) -> tensor<128x1xsi32> {
    %0 = "ttir.argmax"(%arg) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<128x96xbf16>) -> tensor<128x1xsi32>
    return %0 : tensor<128x1xsi32>
  }

  // CHECK-LABEL: func.func @argmax_last_dim_f32
  // CHECK-NOT: "ttir.argmax"
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>]
  // CHECK: d2m.argmax
  // CHECK-SAME: <64, 96>
  // CHECK-NOT: d2m.tile_reduce_max
  func.func @argmax_last_dim_f32(%arg: tensor<64x96xf32>) -> tensor<64x1xsi32> {
    %0 = "ttir.argmax"(%arg) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<64x96xf32>) -> tensor<64x1xsi32>
    return %0 : tensor<64x1xsi32>
  }

  // CHECK-LABEL: func.func @argmax_last_dim_i32
  // CHECK-NOT: "ttir.argmax"
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>]
  // CHECK: d2m.argmax
  // CHECK-SAME: <32, 96>
  // CHECK-NOT: d2m.tile_reduce_max
  func.func @argmax_last_dim_i32(%arg: tensor<32x96xsi32>) -> tensor<32x1xsi32> {
    %0 = "ttir.argmax"(%arg) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<32x96xsi32>) -> tensor<32x1xsi32>
    return %0 : tensor<32x1xsi32>
  }
}
