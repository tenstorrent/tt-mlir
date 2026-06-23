// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

// A degenerate dot_general whose only contracting dimension has extent 1 is a
// pure outer product. It must decompose to a broadcast ttir.multiply (which
// keeps full f32 precision) rather than ttir.matmul (whose hifi4 path splits
// f32 into bf16 hi/lo and loses precision on large products, e.g. RoPE freqs).
module @jit_dot_general_degenerate {
  func.func public @test_dot_general_degenerate(%arg0: tensor<32x1xf32>, %arg1: tensor<1x64xf32>) -> tensor<32x64xf32> {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x1xf32>, tensor<1x64xf32>) -> tensor<32x64xf32>
    // CHECK-LABEL: func.func public @test_dot_general_degenerate
    // CHECK: "ttir.multiply"
    // CHECK-NOT: "ttir.matmul"
    return %0 : tensor<32x64xf32>
  }
}
