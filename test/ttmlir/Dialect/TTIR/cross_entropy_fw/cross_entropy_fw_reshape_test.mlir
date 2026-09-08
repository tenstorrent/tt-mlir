// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

// ttml::metal::cross_entropy_fw only accepts a 4D (N, 1, H, W) input with a 2D
// (N, H) target, so leading dimensions are collapsed into N and the result is
// reshaped back.
module {
  // Rank 2 input: (H, W) with a rank 1 (H) target, so N is 1.
  // CHECK-LABEL: func.func @cross_entropy_fw_rank2
  func.func @cross_entropy_fw_rank2(%input: tensor<32x64xbf16>, %target: tensor<32xui32>)
      -> tensor<32x1xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 32 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg1) <{shape = [1 : i32, 32 : i32]}>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: composite_name = "cross_entropy_fw"
    // CHECK-SAME: -> tensor<1x1x32x1xbf16>
    %0 = "ttcore.composite"(%input, %target) <{composite_name = "cross_entropy_fw", decomposition = @decomp_rank2}>
        : (tensor<32x64xbf16>, tensor<32xui32>) -> tensor<32x1xbf16>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [32 : i32, 1 : i32]}>
    return %0 : tensor<32x1xbf16>
  }

  // Rank 5 input: leading dims collapse into N = 2*3*4 = 24.
  // CHECK-LABEL: func.func @cross_entropy_fw_rank5
  func.func @cross_entropy_fw_rank5(%input: tensor<2x3x4x32x64xbf16>, %target: tensor<24x32xui32>)
      -> tensor<2x3x4x32x1xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [24 : i32, 1 : i32, 32 : i32, 64 : i32]}>
    // CHECK-NOT: "ttir.reshape"(%arg1)
    // CHECK: "ttcore.composite"
    // CHECK-SAME: composite_name = "cross_entropy_fw"
    // CHECK-SAME: -> tensor<24x1x32x1xbf16>
    %0 = "ttcore.composite"(%input, %target) <{composite_name = "cross_entropy_fw", decomposition = @decomp_rank5}>
        : (tensor<2x3x4x32x64xbf16>, tensor<24x32xui32>) -> tensor<2x3x4x32x1xbf16>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [2 : i32, 3 : i32, 4 : i32, 32 : i32, 1 : i32]}>
    return %0 : tensor<2x3x4x32x1xbf16>
  }

  // A channel dimension greater than 1 is folded into N, since the kernel reads
  // one target page per row over N*C*Ht rows.
  // CHECK-LABEL: func.func @cross_entropy_fw_channels
  func.func @cross_entropy_fw_channels(%input: tensor<2x3x32x64xbf16>, %target: tensor<6x32xui32>)
      -> tensor<2x3x32x1xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [6 : i32, 1 : i32, 32 : i32, 64 : i32]}>
    // CHECK-NOT: "ttir.reshape"(%arg1)
    // CHECK: "ttcore.composite"
    // CHECK-SAME: composite_name = "cross_entropy_fw"
    // CHECK-SAME: -> tensor<6x1x32x1xbf16>
    %0 = "ttcore.composite"(%input, %target) <{composite_name = "cross_entropy_fw", decomposition = @decomp_channels}>
        : (tensor<2x3x32x64xbf16>, tensor<6x32xui32>) -> tensor<2x3x32x1xbf16>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [2 : i32, 3 : i32, 32 : i32, 1 : i32]}>
    return %0 : tensor<2x3x32x1xbf16>
  }

  // Only target is reshaped, since the input is already 4D.
  // CHECK-LABEL: func.func @cross_entropy_fw_target_reshape
  func.func @cross_entropy_fw_target_reshape(%input: tensor<1x1x32x64xbf16>, %target: tensor<32xui32>)
      -> tensor<1x1x32x1xbf16> {
    // CHECK-NOT: "ttir.reshape"(%arg0)
    // CHECK-DAG: "ttir.reshape"(%arg1) <{shape = [1 : i32, 32 : i32]}>
    // CHECK: "ttcore.composite"
    // CHECK-SAME: composite_name = "cross_entropy_fw"
    // CHECK-NOT: "ttir.reshape"
    %0 = "ttcore.composite"(%input, %target) <{composite_name = "cross_entropy_fw", decomposition = @decomp_target_reshape}>
        : (tensor<1x1x32x64xbf16>, tensor<32xui32>) -> tensor<1x1x32x1xbf16>
    return %0 : tensor<1x1x32x1xbf16>
  }

  // Already canonical, so it must be left alone.
  // CHECK-LABEL: func.func @cross_entropy_fw_canonical
  func.func @cross_entropy_fw_canonical(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>)
      -> tensor<4x1x32x1xbf16> {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttcore.composite"
    // CHECK-SAME: composite_name = "cross_entropy_fw"
    // CHECK-NOT: "ttir.reshape"
    // CHECK: return
    %0 = "ttcore.composite"(%input, %target) <{composite_name = "cross_entropy_fw", decomposition = @decomp_canonical}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>) -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
  func.func private @decomp_rank2(
      tensor<32x64xbf16>, tensor<32xui32>) -> tensor<32x1xbf16>
  func.func private @decomp_rank5(
      tensor<2x3x4x32x64xbf16>, tensor<24x32xui32>)
      -> tensor<2x3x4x32x1xbf16>
  func.func private @decomp_channels(
      tensor<2x3x32x64xbf16>, tensor<6x32xui32>) -> tensor<2x3x32x1xbf16>
  func.func private @decomp_target_reshape(
      tensor<1x1x32x64xbf16>, tensor<32xui32>) -> tensor<1x1x32x1xbf16>
  func.func private @decomp_canonical(
      tensor<4x1x32x64xbf16>, tensor<4x32xui32>) -> tensor<4x1x32x1xbf16>
}
