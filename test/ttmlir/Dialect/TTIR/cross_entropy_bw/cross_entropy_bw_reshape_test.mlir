// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

// ttml::metal::cross_entropy_bw only accepts a 4D (N, 1, H, W) input with a 2D
// (N, H) target and a (1, 1, 1, 1) grad, so leading dimensions are collapsed
// into N and the result is reshaped back.
module {
  // Rank 2 input: (H, W) with a rank 1 (H) target, so N is 1.
  // CHECK-LABEL: func.func @cross_entropy_bw_rank2
  func.func @cross_entropy_bw_rank2(%input: tensor<32x64xbf16>, %target: tensor<32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<32x64xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 32 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg1) <{shape = [1 : i32, 32 : i32]}>
    // CHECK: "ttir.cross_entropy_bw"
    // CHECK-SAME: -> tensor<1x1x32x64xbf16>
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<32x64xbf16>, tensor<32xui32>, tensor<1x1x1x1xbf16>) -> tensor<32x64xbf16>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [32 : i32, 64 : i32]}>
    return %0 : tensor<32x64xbf16>
  }

  // Rank 5 input: leading dims collapse into N = 2*3*4 = 24.
  // CHECK-LABEL: func.func @cross_entropy_bw_rank5
  func.func @cross_entropy_bw_rank5(%input: tensor<2x3x4x32x64xbf16>, %target: tensor<24x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<2x3x4x32x64xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [24 : i32, 1 : i32, 32 : i32, 64 : i32]}>
    // CHECK: "ttir.cross_entropy_bw"
    // CHECK-SAME: -> tensor<24x1x32x64xbf16>
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<2x3x4x32x64xbf16>, tensor<24x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<2x3x4x32x64xbf16>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [2 : i32, 3 : i32, 4 : i32, 32 : i32, 64 : i32]}>
    return %0 : tensor<2x3x4x32x64xbf16>
  }

  // A channel dimension greater than 1 is folded into N, since the kernel reads
  // one target page per row over N*C*Ht rows.
  // CHECK-LABEL: func.func @cross_entropy_bw_channels
  func.func @cross_entropy_bw_channels(%input: tensor<2x3x32x64xbf16>, %target: tensor<6x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<2x3x32x64xbf16> {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [6 : i32, 1 : i32, 32 : i32, 64 : i32]}>
    // CHECK: "ttir.cross_entropy_bw"
    // CHECK-SAME: -> tensor<6x1x32x64xbf16>
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<2x3x32x64xbf16>, tensor<6x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<2x3x32x64xbf16>
    return %0 : tensor<2x3x32x64xbf16>
  }

  // A scalar grad of another rank is reshaped to (1, 1, 1, 1).
  // CHECK-LABEL: func.func @cross_entropy_bw_grad_rank1
  func.func @cross_entropy_bw_grad_rank1(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1xbf16>)
      -> tensor<4x1x32x64xbf16> {
    // CHECK: "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
    // CHECK: "ttir.cross_entropy_bw"
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>, tensor<1xbf16>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }

  // Already canonical, so it must be left alone.
  // CHECK-LABEL: func.func @cross_entropy_bw_canonical
  func.func @cross_entropy_bw_canonical(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x1x32x64xbf16> {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.cross_entropy_bw"
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }
}
