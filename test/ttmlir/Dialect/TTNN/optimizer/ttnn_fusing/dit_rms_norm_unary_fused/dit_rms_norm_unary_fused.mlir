// RMSNorm + unary activation fusing tests.
//
// Pattern: <activation>(rms_norm(x, weight)) -> dit_rms_norm_unary_fused.
//

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s | FileCheck %s

module {

  // Basic: silu(rms_norm(x, weight)) fuses, baking the activation into the op.
  // CHECK-LABEL: @fuse_silu
  // CHECK: "ttnn.dit_rms_norm_unary_fused"
  // CHECK-SAME: activation = "silu"
  // CHECK-NOT: "ttnn.silu"
  func.func @fuse_silu(%arg0: tensor<32x512xf32>, %arg1: tensor<512xf32>) -> tensor<32x512xf32> {
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{normalized_shape = array<i64: 512>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x512xf32>, tensor<512xf32>) -> tensor<32x512xf32>
    %1 = "ttir.silu"(%0) : (tensor<32x512xf32>) -> tensor<32x512xf32>
    return %1 : tensor<32x512xf32>
  }

  // Look-through: the fusion sees past a value-preserving permute between the
  // rms_norm and the activation, so gelu(permute(rms_norm)) still fuses.
  // CHECK-LABEL: @fuse_gelu_through_permute
  // CHECK: "ttnn.dit_rms_norm_unary_fused"
  // CHECK-SAME: activation = "gelu"
  // CHECK: "ttnn.permute"
  // CHECK-NOT: "ttnn.gelu"
  func.func @fuse_gelu_through_permute(%arg0: tensor<32x512xf32>, %arg1: tensor<512xf32>) -> tensor<512x32xf32> {
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{normalized_shape = array<i64: 512>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x512xf32>, tensor<512xf32>) -> tensor<32x512xf32>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 1, 0>}> : (tensor<32x512xf32>) -> tensor<512x32xf32>
    %2 = "ttir.gelu"(%1) : (tensor<512x32xf32>) -> tensor<512x32xf32>
    return %2 : tensor<512x32xf32>
  }

  // Negative: the rms_norm result feeds two consumers, so fusing would change
  // the other consumer; the pair must remain unfused.
  // CHECK-LABEL: @no_fuse_multi_use
  // CHECK: "ttnn.rms_norm"
  // CHECK: "ttnn.silu"
  // CHECK-NOT: "ttnn.dit_rms_norm_unary_fused"
  func.func @no_fuse_multi_use(%arg0: tensor<32x512xf32>, %arg1: tensor<512xf32>) -> (tensor<32x512xf32>, tensor<32x512xf32>) {
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{normalized_shape = array<i64: 512>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x512xf32>, tensor<512xf32>) -> tensor<32x512xf32>
    %1 = "ttir.silu"(%0) : (tensor<32x512xf32>) -> tensor<32x512xf32>
    return %0, %1 : tensor<32x512xf32>, tensor<32x512xf32>
  }
}
