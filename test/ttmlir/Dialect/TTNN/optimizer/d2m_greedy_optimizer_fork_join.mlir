// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 enable-d2m-fusing-pass=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that the full TTIR-to-TTNN backend pipeline with the greedy optimizer
// (optimization-level=2) and D2M fusing can successfully compile a fork-join
// pattern. The first matmul output is consumed by both the D2M subgraph
// (add + multiply) and the second matmul. The greedy optimizer and D2M pipeline
// should handle the fork point correctly, assigning appropriate layouts and
// compiling the D2M subgraph through TTMetal.
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

// The D2M subgraph ops should be collapsed (inlined as ttnn.generic ops) and
// sharded layouts should be present. Verify key structural properties:
// CHECK: func.func @fork_join
// CHECK: "ttnn.matmul"
// CHECK: "ttnn.generic"
// CHECK: "ttnn.generic"
// CHECK: "ttnn.matmul"
