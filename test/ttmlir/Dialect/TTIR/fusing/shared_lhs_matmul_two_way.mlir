// Two-way shared-LHS fusion is opt-in; enable it for the CHECK run.
// RUN: ttmlir-opt --ttir-fusing="enable-shared-lhs-double-matmul-fusion=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// With the flag off (default) the two-way pairs must NOT fuse.
// RUN: ttmlir-opt --ttir-fusing %s -o %t.disabled.mlir
// RUN: FileCheck %s --check-prefix=DISABLED --input-file=%t.disabled.mlir

module {
  func.func @swiglu_gate_up(
      %input: tensor<1024x4096xbf16>,
      %w_gate: tensor<4096x14336xbf16>,
      %w_up: tensor<4096x14336xbf16>) -> tensor<1024x14336xbf16> {
    // CHECK-LABEL: func.func @swiglu_gate_up
    // CHECK: "ttir.concat"
    // CHECK-SAME: (tensor<4096x14336xbf16>, tensor<4096x14336xbf16>) -> tensor<4096x28672xbf16>
    // CHECK: %[[FUSED:[0-9]+]] = "ttir.matmul"
    // CHECK-SAME: -> tensor<1024x28672xbf16>
    // CHECK-DAG: %[[UP:[0-9]+]] = "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 14336 : i32]
    // CHECK-DAG: %[[GATE:[0-9]+]] = "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 14336 : i32], ends = [1024 : i32, 28672 : i32]
    // CHECK: %[[SILU:[0-9]+]] = "ttir.silu"(%[[GATE]])
    // CHECK: "ttir.multiply"(%[[SILU]], %[[UP]])
    // DISABLED-LABEL: func.func @swiglu_gate_up
    // DISABLED-NOT: "ttir.concat"
    // DISABLED-NOT: "ttir.slice_static"
    %gate = "ttir.matmul"(%input, %w_gate) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x14336xbf16>) -> tensor<1024x14336xbf16>
    %silu = "ttir.silu"(%gate) : (tensor<1024x14336xbf16>) -> tensor<1024x14336xbf16>
    %up = "ttir.matmul"(%input, %w_up) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x14336xbf16>) -> tensor<1024x14336xbf16>
    %out = "ttir.multiply"(%silu, %up) : (tensor<1024x14336xbf16>, tensor<1024x14336xbf16>) -> tensor<1024x14336xbf16>
    return %out : tensor<1024x14336xbf16>
  }
}

module {
  func.func @two_way_plain(
      %input: tensor<32x512xbf16>,
      %w0: tensor<512x384xbf16>,
      %w1: tensor<512x384xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>) {
    // CHECK-LABEL: func.func @two_way_plain
    // CHECK: "ttir.concat"
    // CHECK-SAME: (tensor<512x384xbf16>, tensor<512x384xbf16>) -> tensor<512x768xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: -> tensor<32x768xbf16>
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    // DISABLED-LABEL: func.func @two_way_plain
    // DISABLED-NOT: "ttir.concat"
    // DISABLED-NOT: "ttir.slice_static"
    %0 = "ttir.matmul"(%input, %w0) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
    %1 = "ttir.matmul"(%input, %w1) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
    return %0, %1 : tensor<32x384xbf16>, tensor<32x384xbf16>
  }
}

module {
  func.func @single_matmul_no_fuse(
      %input: tensor<32x512xbf16>,
      %w0: tensor<512x384xbf16>) -> tensor<32x384xbf16> {
    // CHECK-LABEL: func.func @single_matmul_no_fuse
    // CHECK: "ttir.matmul"
    // CHECK-NOT: "ttir.concat"
    // CHECK-NOT: "ttir.slice_static"
    %0 = "ttir.matmul"(%input, %w0) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
    return %0 : tensor<32x384xbf16>
  }
}
