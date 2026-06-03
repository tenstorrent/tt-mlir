// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m %s | FileCheck %s

// Each test checks:
//  1. fill_buffer logical shape (via its metal_layout decl at module top).
//  2. composite_view concat dim, logicalSizes, and result physical shape.
// Logical-shape DAG checks live above CHECK-LABEL so they match the module-
// top metal_layout decls; per-op CHECK-SAMEs match within the func.

// CHECK-DAG: #ttcore.metal_layout<logical_shape = 224x32,
// CHECK-DAG: #ttcore.metal_layout<logical_shape = 224x256,
// CHECK-DAG: #ttcore.metal_layout<logical_shape = 32x16,
// CHECK-DAG: #ttcore.metal_layout<logical_shape = 32x48,
// CHECK-DAG: #ttcore.metal_layout<logical_shape = 32x96,

module {
  // Simple high-pad on inner dim: input 224x224 + fill 224x32 -> 224x256.
  // CHECK-LABEL: func.func @pad_high_inner
  func.func @pad_high_inner(%arg0: tensor<224x224xbf16>) -> tensor<224x256xbf16> {
    // Hi-pad fill: logical 224x32, physical 224x32.
    // CHECK: d2m.fill_buffer
    // CHECK-SAME: value = 1.000000e+00 : bf16
    // CHECK-SAME: tensor<1x1x224x32xbf16
    // composite_view: [input, fill] along dim=1, logical 224x224 + 224x32 -> 224x256.
    // CHECK: d2m.composite_view
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: logicalSizes = array<i64: 224, 32>
    // CHECK-SAME: tensor<1x1x224x256xbf16
    %0 = ttir.empty() : tensor<224x256xbf16>
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 32>, value = 1.0 : f32}> : (tensor<224x224xbf16>) -> tensor<224x256xbf16>
    return %1 : tensor<224x256xbf16>
  }

  // Both lo (16) and hi (48) pads on dim1: two fills + 3-input composite_view.
  // CHECK-LABEL: func.func @pad_both_sides
  func.func @pad_both_sides(%arg0: tensor<32x32xbf16>) -> tensor<32x96xbf16> {
    // Lo fill: logical 32x16, physical 32x32 (16 aligns up to 32).
    // CHECK: d2m.fill_buffer
    // CHECK-SAME: tensor<1x1x32x32xbf16
    // Hi fill: logical 32x48, physical 32x64 (48 aligns up to 64).
    // CHECK: d2m.fill_buffer
    // CHECK-SAME: tensor<1x1x32x64xbf16
    // composite_view: [lo, input, hi] along dim=1, logical 16+32+48 -> 96.
    // CHECK: d2m.composite_view
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: logicalSizes = array<i64: 16, 32, 48>
    // CHECK-SAME: tensor<1x1x32x96xbf16
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 16, 48>, value = 0.0 : f32}> : (tensor<32x32xbf16>) -> tensor<32x96xbf16>
    return %0 : tensor<32x96xbf16>
  }

  // Multi-axis pad is not yet supported by D2M lowering; pattern bails out.
  // TODO: lift this restriction by flattening nested composite_views in
  // ExpandDMAReadCompositeView.
}
