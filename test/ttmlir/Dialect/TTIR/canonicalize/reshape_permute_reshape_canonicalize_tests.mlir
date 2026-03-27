// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // Trailing reshape removes one leading unit dim from permute output.
  // reshape(256x32 → 1x1x256x32) + permute[0,1,3,2] + reshape(1x1x32x256 → 1x32x256)
  // → reshape(256x32 → 1x256x32) + permute[0,2,1]
  func.func @reshape_permute_reshape_drop_one(%arg0: tensor<256x32xbf16>) -> tensor<1x32x256xbf16> {
    // CHECK-LABEL: @reshape_permute_reshape_drop_one
    // CHECK: "ttir.reshape"
    // CHECK-SAME: shape = [1 : i32, 256 : i32, 32 : i32]
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 1>
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 256 : i32, 32 : i32]}> : (tensor<256x32xbf16>) -> tensor<1x1x256x32xbf16>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x1x256x32xbf16>) -> tensor<1x1x32x256xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 256 : i32]}> : (tensor<1x1x32x256xbf16>) -> tensor<1x32x256xbf16>
    return %2 : tensor<1x32x256xbf16>
  }

  // Trailing reshape removes two leading unit dims. Our pattern fires first,
  // producing reshape(8x8 → 8x8) + permute[1,0]. Then identity reshape folds
  // away, leaving just permute[1,0].
  func.func @reshape_permute_reshape_drop_two(%arg0: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
    // CHECK-LABEL: @reshape_permute_reshape_drop_two
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 1, 0>
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 8 : i32, 8 : i32]}> : (tensor<8x8xbf16>) -> tensor<1x1x8x8xbf16>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x1x8x8xbf16>) -> tensor<1x1x8x8xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [8 : i32, 8 : i32]}> : (tensor<1x1x8x8xbf16>) -> tensor<8x8xbf16>
    return %2 : tensor<8x8xbf16>
  }

  // Trailing reshape removes a leading 1 from permute output. Pattern matches,
  // folding to reshape(6 → 2x3) + permute[1,0].
  func.func @reshape_permute_reshape_leading_one_removed(%arg0: tensor<6xbf16>) -> tensor<3x2xbf16> {
    // CHECK-LABEL: @reshape_permute_reshape_leading_one_removed
    // CHECK: "ttir.reshape"
    // CHECK-SAME: shape = [2 : i32, 3 : i32]
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 1, 0>
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 2 : i32, 3 : i32]}> : (tensor<6xbf16>) -> tensor<1x2x3xbf16>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x2x3xbf16>) -> tensor<1x3x2xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [3 : i32, 2 : i32]}> : (tensor<1x3x2xbf16>) -> tensor<3x2xbf16>
    return %2 : tensor<3x2xbf16>
  }

  // Pattern should NOT match: permute shuffles non-unit dim into leading position.
  func.func @reshape_permute_reshape_no_match_perm_mixes(%arg0: tensor<6x4xbf16>) -> tensor<4x6xbf16> {
    // CHECK-LABEL: @reshape_permute_reshape_no_match_perm_mixes
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 2, 0, 1>
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 6 : i32, 4 : i32]}> : (tensor<6x4xbf16>) -> tensor<1x6x4xbf16>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 2, 0, 1>}> : (tensor<1x6x4xbf16>) -> tensor<4x1x6xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [4 : i32, 6 : i32]}> : (tensor<4x1x6xbf16>) -> tensor<4x6xbf16>
    return %2 : tensor<4x6xbf16>
  }
}
