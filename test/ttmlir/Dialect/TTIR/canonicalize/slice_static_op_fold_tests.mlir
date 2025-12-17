// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @consecutive_slice_static_fold(%arg0: tensor<1x133x133x96xbf16>) -> tensor<1x128x128x96xbf16> {
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: ends = [1 : i32, 128 : i32, 128 : i32, 96 : i32]
    // CHECK-NOT: "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 128 : i32, 133 : i32, 96 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x133x133x96xbf16>) -> tensor<1x128x133x96xbf16>
    %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 128 : i32, 128 : i32, 96 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x128x133x96xbf16>) -> tensor<1x128x128x96xbf16>
    return %2: tensor<1x128x128x96xbf16>
  }

  func.func @consecutive_slice_static_fold_non_unit_steps(%arg0: tensor<1x128x128x96xbf16>) -> tensor<1x64x64x96xbf16> {
      // CHECK: "ttir.slice_static"
      // CHECK-SAME: begins = [0 : i32, 1 : i32, 0 : i32, 0 : i32]
      // CHECK-SAME: ends = [1 : i32, 128 : i32, 128 : i32, 96 : i32]
      // CHECK-SAME: step = [1 : i32, 2 : i32, 2 : i32, 1 : i32]
      // CHECK-NOT: "ttir.slice_static"
      %1 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 128 : i32, 128 : i32, 96 : i32], step = [1 : i32, 2 : i32, 1 : i32, 1 : i32]}> : (tensor<1x128x128x96xbf16>) -> tensor<1x64x128x96xbf16>
      %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 64 : i32, 128 : i32, 96 : i32], step = [1 : i32, 1 : i32, 2 : i32, 1 : i32]}> : (tensor<1x64x128x96xbf16>) -> tensor<1x64x64x96xbf16>
      return %2: tensor<1x64x64x96xbf16>
    }

  func.func @consecutive_slice_static_no_fold_multiple_producer_uses(%arg0: tensor<128x128x128xbf16>) -> (tensor<64 x 31 x 31 x bf16>, tensor<64 x 64 x 64 x bf16>) {
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [128 : i32, 128 : i32, 128 : i32]
    // CHECK-SAME: step = [2 : i32, 2 : i32, 2 : i32]
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 2 : i32, 2 : i32]
    // CHECK-SAME: ends = [64 : i32, 64 : i32, 64 : i32]
    // CHECK-SAME: step = [1 : i32, 2 : i32, 2 : i32]
    %1 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [128 : i32, 128 : i32, 128 : i32], step = [2 : i32, 2 : i32, 2 : i32]}> : (tensor<128x128x128xbf16>) -> tensor<64x64x64xbf16>
    %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 2 : i32, 2 : i32], ends = [64 : i32, 64 : i32, 64 : i32], step = [1 : i32, 2 : i32, 2 : i32]}> : (tensor<64x64x64xbf16>) -> tensor<64x31x31xbf16>
    %3 = "ttir.permute"(%1) <{permutation = array<i64: 2, 1, 0>}> : (tensor<64 x 64 x 64 x bf16>) -> tensor<64 x 64 x 64 x bf16>
    return %2, %3 : tensor<64 x 31 x 31 x bf16>, tensor<64 x 64 x 64 x bf16>
  }
}
