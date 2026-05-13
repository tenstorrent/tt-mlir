// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 enable-create-d2m-subgraphs=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// Two D2M subgraphs (unary-only eltwise chains) separated by a matmul.
// Chain 1 (abs, neg, exp) -> matmul -> Chain 2 (sigmoid, neg, abs)
// Each chain should become its own D2M subgraph, compiled to ttnn.generic ops.

module {
  func.func @unary_matmul_unary(
      %arg0: tensor<64x128xbf16>,
      %arg1: tensor<128x64xbf16>)
      -> tensor<64x64xbf16> {
    %0 = "ttir.abs"(%arg0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %1 = "ttir.neg"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %2 = "ttir.exp"(%1) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %3 = "ttir.matmul"(%2, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<64x64xbf16>
    %4 = "ttir.sigmoid"(%3) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %5 = "ttir.neg"(%4) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %6 = "ttir.abs"(%5) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %6 : tensor<64x64xbf16>
  }
}

// CHECK: func.func @unary_matmul_unary

// D2M subgraph 1 (pre-matmul chain: abs, neg, exp) compiled to generic ops.
// CHECK: "ttnn.generic"

// Matmul separating the two subgraphs.
// CHECK: "ttnn.matmul"

// D2M subgraph 2 (post-matmul chain: sigmoid, neg, abs) compiled to generic ops.
// CHECK: "ttnn.generic"

// No D2M subgraph ops should remain after compilation.
// CHECK-NOT: ttnn.d2m_subgraph
