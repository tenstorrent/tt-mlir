// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1" --split-input-file %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" --split-input-file %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0" --split-input-file %s | FileCheck --check-prefix=OPT0 %s

// The fused ttnn.group_norm kernel used to reduce over the tile-padding rows as
// if they were data, so a non-tile-aligned per-sample H*W was silently wrong and
// every such shape was decomposed. tt-metal #50682 corrects for those rows
// analytically, so the alignment precondition is gone: with the optimizer on the
// op model alone decides, and non-aligned shapes keep the fused kernel.
//
// Optimization level 0 configures no op-model validation, so it still decomposes
// everything -- unchanged by #50682.

// Aligned H*W = 64: keep the fused kernel.
// CHECK-LABEL: func.func @gn_aligned
// CHECK: "ttnn.group_norm"
// OPT0-LABEL: func.func @gn_aligned
// OPT0-NOT: "ttnn.group_norm"
module {
  func.func @gn_aligned(%arg0: tensor<1x1x64x480xbf16>) -> tensor<1x1x64x480xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x64x480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }
}

// -----

// Non-aligned H*W = 50 (XTTS-v2 conditioning): the #50682 correction handles the
// tile-padding rows, so this now keeps the fused kernel instead of decomposing.
// CHECK-LABEL: func.func @gn_non_tile_aligned
// CHECK: "ttnn.group_norm"
// OPT0-LABEL: func.func @gn_non_tile_aligned
// OPT0-NOT: "ttnn.group_norm"
module {
  func.func @gn_non_tile_aligned(%arg0: tensor<1x1x50x480xbf16>) -> tensor<1x1x50x480xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x50x480xbf16>) -> tensor<1x1x50x480xbf16>
    return %1 : tensor<1x1x50x480xbf16>
  }
}

// -----

// N=2, H*W=16: the correction keys off the PER-SAMPLE height, so this is a
// non-aligned case even though N*H*W=32 looks aligned. It keeps the fused kernel
// too. Note H*W=16 is the K=1 padding-fraction regime: measured on Blackhole it
// stays within tolerance for near-zero-mean inputs (0.038 vs an aligned-control
// 0.041) but exceeds the op's 0.08 under a large input mean (0.084). See #50682.
// CHECK-LABEL: func.func @gn_n2_non_tile_aligned
// CHECK: "ttnn.group_norm"
// OPT0-LABEL: func.func @gn_n2_non_tile_aligned
// OPT0-NOT: "ttnn.group_norm"
module {
  func.func @gn_n2_non_tile_aligned(%arg0: tensor<2x1x16x256xbf16>) -> tensor<2x1x16x256xbf16> {
    %1 = "ttir.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<2x1x16x256xbf16>) -> tensor<2x1x16x256xbf16>
    return %1 : tensor<2x1x16x256xbf16>
  }
}
