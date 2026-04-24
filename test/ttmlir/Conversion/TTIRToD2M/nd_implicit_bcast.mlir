// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-DAG: #[[MAP_ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP_COL_BCAST:.+]] = affine_map<(d0, d1) -> (d0, 0)>

// Verify that TTIRToD2M correctly lowers elementwise ops with implicit
// broadcasts whose logical operand rank (>2) is greater than the physical
// post-collapse rank (=2 for the standard TTNN layout).
//
// The d2m.generic / linalg.generic indexing maps must be built in the
// (collapsed) physical iteration domain, derived from per-operand physical
// shard-tile-shape comparison. In-tile broadcasts are still expressed via
// d2m.tile_bcast inside the body.

#dram = #ttnn.buffer_type<dram>

// 4D operand layouts that all collapse to a 2D physical memref:
//   1x1x32x128 -> physical (32, 128) -> tiles (1, 4)   "_full"
//   1x1x32x1   -> physical (32, 1)   -> tiles (1, 1)   "_col_bcast"
//   1x1x1x128  -> physical (1, 128)  -> tiles (1, 4)   "_row_bcast"
//   1x1x1x1    -> physical (1, 1)    -> tiles (1, 1)   "_scalar_bcast"
#layout4d_full        = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout4d_col_bcast   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout4d_row_bcast   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout4d_scalar_bcast = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {

  // ND implicit row broadcast (input is 1 along the second-to-last logical
  // dim): physical shard tiles match output (1,4 vs 1,4), so the outer
  // indexing map is identity and only an in-tile row broadcast is needed.
  // CHECK-LABEL: func.func @nd4_row_bcast
  func.func @nd4_row_bcast(
      %arg0: tensor<1x1x32x128xbf16, #layout4d_full>,
      %arg1: tensor<1x1x1x128xbf16, #layout4d_row_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full> {
    // CHECK: d2m.generic
    // CHECK-SAME: indexing_maps = [#[[MAP_ID]], #[[MAP_ID]], #[[MAP_ID]]]
    // CHECK-SAME: iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<1x1x32x128xbf16, #layout4d_full>, tensor<1x1x1x128xbf16, #layout4d_row_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    return %0 : tensor<1x1x32x128xbf16, #layout4d_full>
  }

  // ND implicit col broadcast (input is 1 along the last logical dim):
  // physical shard tiles differ (1,1 vs 1,4), so the outer indexing map
  // broadcasts on physical dim 1, plus an in-tile col broadcast.
  // CHECK-LABEL: func.func @nd4_col_bcast
  func.func @nd4_col_bcast(
      %arg0: tensor<1x1x32x128xbf16, #layout4d_full>,
      %arg1: tensor<1x1x32x1xbf16, #layout4d_col_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full> {
    // CHECK: d2m.generic
    // CHECK-SAME: indexing_maps = [#[[MAP_ID]], #[[MAP_COL_BCAST]], #[[MAP_ID]]]
    // CHECK-SAME: iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<1x1x32x128xbf16, #layout4d_full>, tensor<1x1x32x1xbf16, #layout4d_col_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    return %0 : tensor<1x1x32x128xbf16, #layout4d_full>
  }

  // ND implicit scalar broadcast (input is 1 in all dims): physical shard
  // tiles differ on the last dim (1,1 vs 1,4), in-tile broadcast is scalar.
  // CHECK-LABEL: func.func @nd4_scalar_bcast
  func.func @nd4_scalar_bcast(
      %arg0: tensor<1x1x32x128xbf16, #layout4d_full>,
      %arg1: tensor<1x1x1x1xbf16, #layout4d_scalar_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full> {
    // CHECK: d2m.generic
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type scalar>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<1x1x32x128xbf16, #layout4d_full>, tensor<1x1x1x1xbf16, #layout4d_scalar_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    return %0 : tensor<1x1x32x128xbf16, #layout4d_full>
  }

  // Dual ND broadcast: one input row-bcast, one input col-bcast.
  // CHECK-LABEL: func.func @nd4_dual_bcast
  func.func @nd4_dual_bcast(
      %arg0: tensor<1x1x1x128xbf16, #layout4d_row_bcast>,
      %arg1: tensor<1x1x32x1xbf16, #layout4d_col_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full> {
    // CHECK: d2m.generic
    // CHECK: linalg.generic
    // CHECK-DAG: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
    // CHECK-DAG: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<1x1x1x128xbf16, #layout4d_row_bcast>, tensor<1x1x32x1xbf16, #layout4d_col_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    return %0 : tensor<1x1x32x128xbf16, #layout4d_full>
  }

  // ND chain that previously crashed in TTIRToD2M::getMulticastGridDims when
  // the d2m-fusing pass pulled it into a d2m_subgraph. With the physical-rank
  // indexing-map fix it lowers to two d2m.generic ops (one per ttir op).
  // CHECK-LABEL: func.func @nd4_bcast_chain
  func.func @nd4_bcast_chain(
      %arg0: tensor<1x1x32x128xbf16, #layout4d_full>,
      %arg1: tensor<1x1x32x1xbf16, #layout4d_col_bcast>,
      %arg2: tensor<1x1x1x128xbf16, #layout4d_row_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full> {
    // CHECK-COUNT-2: d2m.generic
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<1x1x32x128xbf16, #layout4d_full>, tensor<1x1x32x1xbf16, #layout4d_col_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    %1 = "ttir.multiply"(%0, %arg2) : (tensor<1x1x32x128xbf16, #layout4d_full>, tensor<1x1x1x128xbf16, #layout4d_row_bcast>) -> tensor<1x1x32x128xbf16, #layout4d_full>
    return %1 : tensor<1x1x32x128xbf16, #layout4d_full>
  }
}

// ----- Original gpt-oss-20b subgraph regression (reduced) -----
//
// Reproducer for the original error reported by the gpt-oss-20b subgraph: a
// chain of `ttir.gt` (comparison) and `ttir.logical_and` (logical) ops with
// 4D logical operands that all collapse to a 2D physical TTNN layout. The
// pre-fix TTIRToD2M produced 4D logical-rank indexing maps but a 2D physical
// iteration domain, leading to an index-out-of-bounds crash inside
// getMulticastGridDims when looking up iteratorTypes[iterDimPos]. Both the
// comparison-op (`isComparisonOp`) and the logical-op (NEZ-decomposition)
// codepaths in `createComputeRegion` go through the same indexing-map
// construction, so this test covers both ops in addition to the
// arithmetic ones above.

#dram2 = #ttnn.buffer_type<dram>
#layout38 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram2>, <interleaved>>
#layout41 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, si32>, #dram2>, <interleaved>>
#layout65 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram2>, <interleaved>>
#layout67 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram2>, <interleaved>>

module {
  // CHECK-LABEL: func.func private @d2m_subgraph
  func.func private @d2m_subgraph(
      %arg0: tensor<1x1x1x128xsi32, #layout41>,
      %arg1: tensor<1x1x17x1xsi32,  #layout65>,
      %arg2: tensor<1x1x1x1xbf16,   #layout38>,
      %arg3: tensor<1x1x17x128xbf16,#layout67>)
      -> tensor<1x1x17x128xbf16, #layout67> {
    // Three elementwise ops, each lowered to its own d2m.generic.
    // CHECK-COUNT-3: d2m.generic
    %0 = "ttir.gt"(%arg0, %arg1)
        : (tensor<1x1x1x128xsi32, #layout41>,
           tensor<1x1x17x1xsi32,  #layout65>)
        -> tensor<1x1x17x128xbf16, #layout67>
    %1 = "ttir.logical_and"(%arg2, %0)
        : (tensor<1x1x1x1xbf16,   #layout38>,
           tensor<1x1x17x128xbf16,#layout67>)
        -> tensor<1x1x17x128xbf16, #layout67>
    %2 = "ttir.logical_and"(%1, %arg3)
        : (tensor<1x1x17x128xbf16,#layout67>,
           tensor<1x1x17x128xbf16,#layout67>)
        -> tensor<1x1x17x128xbf16, #layout67>
    return %2 : tensor<1x1x17x128xbf16, #layout67>
  }
}
