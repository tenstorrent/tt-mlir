// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 enable-d2m-fusing-pass=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that the full TTIR-to-TTNN backend pipeline with the greedy optimizer
// (optimization-level=2) and D2M fusing can successfully compile a linear chain
// of matmul, add, and multiply operations.
// D2M fusing should fuse the eltwise ops (add + multiply) into a D2M subgraph
// and the greedy optimizer should assign L1-sharded layouts where possible.
// The D2M subgraph is then compiled through TTMetal and inlined back.
module {
  func.func @simple_64x64(
      %arg0: tensor<64x64xbf16>,
      %arg1: tensor<64x64xbf16>,
      %arg2: tensor<64x64xbf16>,
      %arg3: tensor<64x64xbf16>,
      %arg4: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %1 = "ttir.matmul"(%0, %arg2) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = "ttir.add"(%1, %arg3) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %3 = "ttir.multiply"(%2, %arg0) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %4 = "ttir.matmul"(%3, %arg4) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %4 : tensor<64x64xbf16>
  }
}

// The D2M subgraph ops should be collapsed (inlined as ttnn.generic ops) and
// sharded layouts should be present. Verify key structural properties:
// CHECK: func.func @simple_64x64
// CHECK: "ttnn.matmul"
// CHECK: "ttnn.matmul"
// CHECK: "ttnn.generic"
// CHECK: "ttnn.generic"
// CHECK: "ttnn.matmul"
