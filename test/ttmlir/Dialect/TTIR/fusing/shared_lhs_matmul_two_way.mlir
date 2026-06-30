// RUN: ttmlir-opt --ttir-fusing %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module {
  func.func @swiglu_gate_up(
      %input: tensor<1024x4096xbf16>,
      %w_gate: tensor<4096x14336xbf16>,
      %w_up: tensor<4096x14336xbf16>) -> tensor<1024x14336xbf16> {
    // CHECK-LABEL: func.func @swiglu_gate_up
    // CHECK: "ttir.concat"
    // CHECK-SAME: (tensor<4096x14336xbf16>, tensor<4096x14336xbf16>) -> tensor<4096x28672xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: -> tensor<1024x28672xbf16>
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.silu"
    // CHECK: "ttir.multiply"
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
