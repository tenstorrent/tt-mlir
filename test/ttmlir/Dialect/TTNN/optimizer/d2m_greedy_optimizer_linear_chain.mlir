// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 enable-d2m-subgraphs=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that the full TTIR-to-TTNN backend pipeline with the greedy optimizer
// (optimization-level=2) and D2M fusion can successfully compile a linear chain
// of matmul, add, and multiply operations.
// D2M fusion (ttnn-d2m-subgraphs and elementwisefusion passes) should fuse the eltwise ops (add + multiply) into a D2M subgraph
// and the greedy optimizer should assign L1-sharded layouts where possible.
// The D2M subgraph is then compiled through TTMetal and inlined back as
// ttnn.generic ops.
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

// Verify structure after full pipeline: matmul -> matmul -> generic(add) ->
// generic(multiply) -> generic(to_layout) -> matmul.
// D2M subgraph ops should be compiled into ttnn.generic ops.
// Matmuls should get L1-sharded layouts from the greedy optimizer.

// CHECK: func.func @simple_64x64

// First matmul: output in L1.
// CHECK: %[[MATMUL_0:.*]] = "ttnn.matmul"
// CHECK-SAME: #ttnn.buffer_type<l1>

// Second matmul: output in L1.
// CHECK: %[[MATMUL_1:.*]] = "ttnn.matmul"
// CHECK-SAME: #ttnn.buffer_type<l1>

// D2M subgraph compiled into 2 generic ops (fused add/multiply + to_layout).
// CHECK: "ttnn.generic"
// CHECK-SAME: #ttnn.compute_kernel<symbol_ref = @[[FUSED_KERNEL:[a-zA-Z0-9_]+]]
// CHECK: "ttnn.generic"
// CHECK-NOT: "ttnn.generic"

// Third matmul: output in L1.
// CHECK: %[[MATMUL_2:.*]] = "ttnn.matmul"
// CHECK-SAME: #ttnn.buffer_type<l1>

// No D2M subgraph ops should remain after compilation.
// CHECK-NOT: ttnn.d2m_subgraph

// The fused compute kernel contains both add_tiles and mul_tiles.
// CHECK: func.func private @[[FUSED_KERNEL]]()
// CHECK: add_tiles
// CHECK: mul_tiles
