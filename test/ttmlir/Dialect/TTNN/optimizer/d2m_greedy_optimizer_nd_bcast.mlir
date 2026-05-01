// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 enable-create-d2m-subgraphs=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// End-to-end regression: a TTIR graph with implicit ND broadcasts in an
// elementwise chain used to crash the full ttir-to-ttnn-backend-pipeline
// (with optimization-level=2 and enable-create-d2m-subgraphs=true) inside
// TTIRToD2M::getMulticastGridDims when the d2m-subgraphs pass pulled the chain
// into a d2m_subgraph.
//
// The fix builds the d2m.generic / linalg.generic indexing maps in the
// physical (post-layout-collapse) iteration domain (derived from per-operand
// physical shard-tile-shape comparison), so the broadcast-needing ND ops can
// stay inside the d2m_subgraph and the pipeline runs to completion all the
// way down to ttnn.generic compiled kernels.

module {
  func.func @nd4_bcast_chain(
      %arg0: tensor<1x1x32x128xbf16>,
      %arg1: tensor<1x1x32x1xbf16>,
      %arg2: tensor<1x1x1x128xbf16>) -> tensor<1x1x32x128xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<1x1x32x128xbf16>, tensor<1x1x32x1xbf16>) -> tensor<1x1x32x128xbf16>
    %1 = "ttir.multiply"(%0, %arg2) : (tensor<1x1x32x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x1x32x128xbf16>
    return %1 : tensor<1x1x32x128xbf16>
  }
}

// CHECK: func.func @nd4_bcast_chain
// The implicit broadcasted ops should be fused via the d2m path and compiled
// to ttnn.generic kernels rather than left as native ttnn.add / ttnn.multiply.
// CHECK-NOT: "ttnn.add"
// CHECK-NOT: "ttnn.multiply"
// CHECK: "ttnn.generic"
