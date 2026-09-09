// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1" --split-input-file %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" --split-input-file %s | FileCheck %s

// Aligned H*W = 64.
// CHECK-LABEL: func.func @gn_aligned
// CHECK: "ttnn.group_norm"
module {
  func.func @gn_aligned(%arg0: tensor<1x1x64x480xbf16>) -> tensor<1x1x64x480xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x64x480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }
}

// -----

// Non-aligned H*W = 50 (XTTS-v2 conditioning).
// CHECK-LABEL: func.func @gn_non_tile_aligned
// CHECK: "ttnn.group_norm"
module {
  func.func @gn_non_tile_aligned(%arg0: tensor<1x1x50x480xbf16>) -> tensor<1x1x50x480xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x50x480xbf16>) -> tensor<1x1x50x480xbf16>
    return %1 : tensor<1x1x50x480xbf16>
  }
}

// -----

// N=2, H*W=16: per-sample non-aligned even though N*H*W=32 is aligned.
// CHECK-LABEL: func.func @gn_n2_non_tile_aligned
// CHECK: "ttnn.group_norm"
module {
  func.func @gn_n2_non_tile_aligned(%arg0: tensor<2x1x16x256xbf16>) -> tensor<2x1x16x256xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<2x1x16x256xbf16>) -> tensor<2x1x16x256xbf16>
    return %1 : tensor<2x1x16x256xbf16>
  }
}

// -----

// Non-aligned H*W with non-tile-aligned C: C=80, G=8 -> padded C=160, G=16.
// CHECK-LABEL: func.func @gn_non_tile_aligned_unaligned_channels
// CHECK: "ttnn.group_norm"
module {
  func.func @gn_non_tile_aligned_unaligned_channels(%arg0: tensor<1x1x50x80xbf16>) -> tensor<1x1x50x80xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x50x80xbf16>) -> tensor<1x1x50x80xbf16>
    return %1 : tensor<1x1x50x80xbf16>
  }
}
