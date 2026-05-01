// RUN: ttmlir-opt --ttcore-register-device --ttcore-wrap-device-module --ttnn-create-d2m-subgraphs %s | FileCheck %s

// Verify that the TTNN d2m-subgraphs pass DOES fuse elementwise ops with
// implicit broadcasts on >2D operands into a d2m_subgraph. The TTIRToD2M
// elementwise rewriter builds the d2m.generic indexing maps in the physical
// (post-layout-collapse) iteration domain, so >2D logical broadcasts that
// collapse to 2D physical layouts are fully supported (the broadcast is
// expressed as either an outer per-shard-tile broadcast in the indexing map
// or an in-tile broadcast via d2m.tile_bcast).

#l1 = #ttnn.buffer_type<l1>
#layout4d_full = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout4d_col_bcast = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout4d_row_bcast = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout3d_full = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout3d_col_bcast = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout2d = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout2d_row = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

module {

  // 4D chain with implicit broadcast on the trailing dim of one operand and
  // on a leading dim of another. Both ops should be fused into a single
  // d2m_subgraph (the broadcasts are handled at lower levels).
  // CHECK-LABEL: func.func @nd4_bcast_chain_fused
  func.func @nd4_bcast_chain_fused(
      %arg0: tensor<1x1x32x128xbf16, #layout4d_full>,
      %arg1: tensor<1x1x32x1xbf16, #layout4d_col_bcast>,
      %arg2: tensor<1x1x1x128xbf16, #layout4d_row_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full> {
    // CHECK: ttnn.d2m_subgraph
    // CHECK-NOT: "ttnn.add"
    // CHECK-NOT: "ttnn.multiply"
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x128xbf16, #layout4d_full>, tensor<1x1x32x1xbf16, #layout4d_col_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    %1 = "ttnn.multiply"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x128xbf16, #layout4d_full>, tensor<1x1x1x128xbf16, #layout4d_row_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    return %1 : tensor<1x1x32x128xbf16, #layout4d_full>
  }

  // 3D chain with implicit broadcast: also fused.
  // CHECK-LABEL: func.func @nd3_bcast_chain_fused
  func.func @nd3_bcast_chain_fused(
      %arg0: tensor<1x32x128xbf16, #layout3d_full>,
      %arg1: tensor<1x32x1xbf16, #layout3d_col_bcast>) -> tensor<1x32x128xbf16, #layout3d_full> {
    // CHECK: ttnn.d2m_subgraph
    // CHECK-NOT: "ttnn.add"
    // CHECK-NOT: "ttnn.multiply"
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x32x128xbf16, #layout3d_full>, tensor<1x32x1xbf16, #layout3d_col_bcast>) -> tensor<1x32x128xbf16, #layout3d_full>
    %1 = "ttnn.multiply"(%0, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x32x128xbf16, #layout3d_full>, tensor<1x32x128xbf16, #layout3d_full>) -> tensor<1x32x128xbf16, #layout3d_full>
    return %1 : tensor<1x32x128xbf16, #layout3d_full>
  }

  // 4D chain WITHOUT broadcasts: all operands match output shape. Should also
  // still be fused into a d2m_subgraph.
  // CHECK-LABEL: func.func @nd4_no_bcast_still_fused
  func.func @nd4_no_bcast_still_fused(
      %arg0: tensor<1x1x32x128xbf16, #layout4d_full>,
      %arg1: tensor<1x1x32x128xbf16, #layout4d_full>,
      %arg2: tensor<1x1x32x128xbf16, #layout4d_full>) -> tensor<1x1x32x128xbf16, #layout4d_full> {
    // CHECK: ttnn.d2m_subgraph
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x128xbf16, #layout4d_full>, tensor<1x1x32x128xbf16, #layout4d_full>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    %1 = "ttnn.multiply"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x128xbf16, #layout4d_full>, tensor<1x1x32x128xbf16, #layout4d_full>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    return %1 : tensor<1x1x32x128xbf16, #layout4d_full>
  }

  // 2D chain WITH implicit broadcast: still fused (covered the regression
  // baseline before/after the ND fix).
  // CHECK-LABEL: func.func @nd2_bcast_still_fused
  func.func @nd2_bcast_still_fused(
      %arg0: tensor<64x256xbf16, #layout2d>,
      %arg1: tensor<1x256xbf16, #layout2d_row>,
      %arg2: tensor<64x256xbf16, #layout2d>) -> tensor<64x256xbf16, #layout2d> {
    // CHECK: ttnn.d2m_subgraph
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout2d>, tensor<1x256xbf16, #layout2d_row>) -> tensor<64x256xbf16, #layout2d>
    %1 = "ttnn.multiply"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x256xbf16, #layout2d>, tensor<64x256xbf16, #layout2d>) -> tensor<64x256xbf16, #layout2d>
    return %1 : tensor<64x256xbf16, #layout2d>
  }
}
