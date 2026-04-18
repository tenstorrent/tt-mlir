// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 enable-d2m-fusing-pass=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that the full TTIR-to-TTNN backend pipeline with the greedy optimizer
// (optimization-level=2) and D2M fusing handles fork-join patterns.
// The first matmul output is consumed by both the d2m_subgraph (add, multiply)
// and the final matmul (fork point). The D2M subgraph is compiled through
// TTMetal and inlined back as ttnn.generic ops.
module {
  func.func @fork_join(%arg0: tensor<64x64xbf16>,
                       %arg1: tensor<64x64xbf16>,
                       %arg2: tensor<64x64xbf16>) -> (tensor<64x64xbf16>) {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %1 = "ttir.add"(%0, %arg2) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = "ttir.multiply"(%1, %arg0) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %3 = "ttir.matmul"(%0, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}

// Verify structure after full pipeline: matmul -> generic(add) ->
// generic(multiply) -> generic(to_layout) -> matmul.
// D2M subgraph ops should be compiled into ttnn.generic ops.

// CHECK: func.func @fork_join

// First matmul: output in L1.
// CHECK: %[[MATMUL_0:.*]] = "ttnn.matmul"
// CHECK-SAME: #ttnn.buffer_type<l1>

// D2M subgraph compiled into generic ops (add + multiply + to_layout).
// CHECK: "ttnn.generic"
// CHECK: "ttnn.generic"
// CHECK: "ttnn.generic"

// Second matmul: output in L1.
// CHECK: %[[MATMUL_1:.*]] = "ttnn.matmul"
// CHECK-SAME: #ttnn.buffer_type<l1>

// No D2M subgraph ops should remain after compilation.
// CHECK-NOT: ttnn.d2m_subgraph
