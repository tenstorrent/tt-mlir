// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// ttir.squeeze and ttir.unsqueeze are normalized into ttir.reshape.

// The reshape's shape comes from the result type rather than being recomputed
// from `dim`, so a negative dim needs no special handling.
module {
  func.func @unsqueeze_negative_dim(%arg0: tensor<1x8x128x64xbf16>) -> tensor<1x8x128x64x1xbf16> {
    // CHECK-LABEL: @unsqueeze_negative_dim
    // CHECK: "ttir.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 128 : i32, 64 : i32, 1 : i32]
    %0 = "ttir.unsqueeze"(%arg0) <{dim = -1 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x8x128x64x1xbf16>
    return %0 : tensor<1x8x128x64x1xbf16>
  }
}

module {
  func.func @squeeze_negative_dim(%arg0: tensor<8x128x64x1xbf16>) -> tensor<8x128x64xbf16> {
    // CHECK-LABEL: @squeeze_negative_dim
    // CHECK: "ttir.reshape"(%arg0)
    // CHECK-SAME: shape = [8 : i32, 128 : i32, 64 : i32]
    %0 = "ttir.squeeze"(%arg0) <{dim = -1 : si32}> : (tensor<8x128x64x1xbf16>) -> tensor<8x128x64xbf16>
    return %0 : tensor<8x128x64xbf16>
  }
}

// Normalizing also puts the shape change within reach of the existing reshape
// folds: an unsqueeze/squeeze round trip collapses entirely, which it could not
// do while the two ops were opaque to each other.
module {
  func.func @unsqueeze_squeeze_round_trip(%arg0: tensor<1x8x128x64xbf16>) -> tensor<1x8x128x64xbf16> {
    // CHECK-LABEL: @unsqueeze_squeeze_round_trip
    // CHECK-NOT: ttir.reshape
    // CHECK: return %arg0
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 2 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x8x1x128x64xbf16>
    %1 = "ttir.squeeze"(%0) <{dim = 2 : si32}> : (tensor<1x8x1x128x64xbf16>) -> tensor<1x8x128x64xbf16>
    return %1 : tensor<1x8x128x64xbf16>
  }
}
