// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @consecutive_repeat_fold(%arg0: tensor<1x128x128x96xbf16>) -> tensor<8x256x256x1536xbf16> {
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 8, 2, 2, 16>
    // CHECK-NOT: "ttir.repeat"

    %1 = "ttir.repeat"(%arg0) <{repeat_dimensions = array<i64: 1, 2, 2, 4>}> : (tensor<1x128x128x96xbf16>) -> tensor<1x256x256x384xbf16>
    %2 = "ttir.repeat"(%1) <{repeat_dimensions = array<i64: 8, 1, 1, 4>}> : (tensor<1x256x256x384xbf16>) -> tensor<8x256x256x1536xbf16>
    return %2 : tensor<8x256x256x1536xbf16>
  }

  func.func @consecutive_repeat_fold_no_fold_multiple_producer_uses(%arg0: tensor<1x128x128x96xbf16>) -> (tensor<8x256x256x384xbf16>, tensor<384x256x256x1xbf16>) {
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 1, 2, 2, 4>
    // CHECK: "ttir.repeat"
    // CHECK-SAME: repeat_dimensions = array<i64: 8, 1, 1, 1>

    %1 = "ttir.repeat"(%arg0) <{repeat_dimensions = array<i64: 1, 2, 2, 4>}> : (tensor<1x128x128x96xbf16>) -> tensor<1x256x256x384xbf16>
    %2 = "ttir.repeat"(%1) <{repeat_dimensions = array<i64: 8, 1, 1, 1>}> : (tensor<1x256x256x384xbf16>) -> tensor<8x256x256x384xbf16>
    %3 = "ttir.permute"(%1) <{permutation = array<i64: 3, 2, 1, 0>}> : (tensor<1x256x256x384xbf16>) -> tensor<384x256x256x1xbf16>
    return %2, %3 : tensor<8x256x256x384xbf16>, tensor<384x256x256x1xbf16>
  }

  func.func @identity_repeat_fold(%arg0: tensor<1x128x128x96xbf16>) -> tensor<1x128x128x96xbf16> {
    // CHECK-NOT: "ttir.repeat"

    %1 = "ttir.repeat"(%arg0) <{repeat_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x128x128x96xbf16>) -> tensor<1x128x128x96xbf16>
    %2 = "ttir.add"(%1, %1) : (tensor<1x128x128x96xbf16>, tensor<1x128x128x96xbf16>) -> tensor<1x128x128x96xbf16>
    return %2 : tensor<1x128x128x96xbf16>
  }
}
