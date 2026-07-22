// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1" --split-input-file %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" --split-input-file %s | FileCheck %s

// With the optimizer on, group_norm keeps the fused kernel only when the
// per-sample flattened height (H*W) is tile-aligned; non-aligned shapes are
// silently wrong on the fused kernel (op model does not flag it) so must
// decompose.

// Aligned H*W = 64: keep the fused kernel.
// CHECK-LABEL: func.func @gn_aligned
// CHECK: "ttnn.group_norm"
module {
  func.func @gn_aligned(%arg0: tensor<1x1x64x480xbf16>) -> tensor<1x1x64x480xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x64x480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }
}

// -----

// Non-aligned H*W = 50 (XTTS-v2 conditioning): decompose.
// CHECK-LABEL: func.func @gn_non_tile_aligned
// CHECK-NOT: "ttnn.group_norm"
module {
  func.func @gn_non_tile_aligned(%arg0: tensor<1x1x50x480xbf16>) -> tensor<1x1x50x480xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x50x480xbf16>) -> tensor<1x1x50x480xbf16>
    return %1 : tensor<1x1x50x480xbf16>
  }
}

// -----

// N=2, H*W=16: not aligned even though N*H*W=32 is, so decompose.
// CHECK-LABEL: func.func @gn_n2_non_tile_aligned
// CHECK-NOT: "ttnn.group_norm"
module {
  func.func @gn_n2_non_tile_aligned(%arg0: tensor<2x1x16x256xbf16>) -> tensor<2x1x16x256xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<2x1x16x256xbf16>) -> tensor<2x1x16x256xbf16>
    return %1 : tensor<2x1x16x256xbf16>
  }
}
