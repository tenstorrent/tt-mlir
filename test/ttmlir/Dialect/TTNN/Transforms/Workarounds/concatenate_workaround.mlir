// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// Layouts for the overflow case: [1993728, 3] and [1993728, 1] int32 TILE_LAYOUT DRAM.
// 1993728 = 62304 * 32, so dim=0 is tile-aligned.
// Last dims 3 and 1 are not tile-aligned (padded to 32), triggering the
// untilize -> transpose(-2,-1) -> concat -> transpose(-2,-1) -> retilize path.
// After transpose, the new last dim is 1993728, giving:
//   single_page_size = 4 * 1993728 = 7,974,912 bytes >> usable L1.
#dram = #ttnn.buffer_type<dram>
#layout_a = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<62304x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<62304x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<62304x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

// Layout for the no-overflow case: [64, 3] and [64, 1] int32.
// After transpose the new last dim is 64: 4 * 64 * 2 = 512 bytes << usable L1.
#layout_small_a = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_small_b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_small_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

module {
  // Test: concat [1993728, 3] + [1993728, 1] along dim=1 (last dim, unaligned).
  // dim[-2] = 1993728, so CB after transpose = 4 * 1993728 * 2 >> L1.
  // Workaround should decompose into chunk concats along dim=0 followed by
  // a final concat along dim=0.
  func.func @concat_last_dim_unaligned_cb_exceeds_l1(
      %arg0: tensor<1993728x3xsi32, #layout_a>,
      %arg1: tensor<1993728x1xsi32, #layout_b>)
      -> tensor<1993728x4xsi32, #layout_out> {
    // CHECK-LABEL: func.func @concat_last_dim_unaligned_cb_exceeds_l1
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.concat"
    // CHECK: "ttnn.concat"
    %0 = "ttnn.concat"(%arg0, %arg1)
        <{dim = 1 : si32}>
        : (tensor<1993728x3xsi32, #layout_a>,
           tensor<1993728x1xsi32, #layout_b>)
       -> tensor<1993728x4xsi32, #layout_out>
    return %0 : tensor<1993728x4xsi32, #layout_out>
  }

  // Test: concat [64, 3] + [64, 1] along dim=1 (last dim, unaligned).
  // dim[-2] = 64, so CB after transpose = 4 * 64 * 2 = 512 bytes << L1.
  // Workaround should NOT fire — op is left unchanged.
  func.func @concat_last_dim_unaligned_cb_fits_l1(
      %arg0: tensor<64x3xsi32, #layout_small_a>,
      %arg1: tensor<64x1xsi32, #layout_small_b>)
      -> tensor<64x4xsi32, #layout_small_out> {
    // CHECK-LABEL: func.func @concat_last_dim_unaligned_cb_fits_l1
    // CHECK-NOT: "ttnn.slice_static"
    // CHECK: "ttnn.concat"
    // CHECK-SAME: dim = 1
    %0 = "ttnn.concat"(%arg0, %arg1)
        <{dim = 1 : si32}>
        : (tensor<64x3xsi32, #layout_small_a>,
           tensor<64x1xsi32, #layout_small_b>)
       -> tensor<64x4xsi32, #layout_small_out>
    return %0 : tensor<64x4xsi32, #layout_small_out>
  }

  // Test: concat [1993728, 32] + [1993728, 32] along dim=1 (last dim, aligned).
  // Both inputs have logical[-1] == padded[-1] == 32, so untilize does not
  // trigger and the workaround should NOT fire.
  func.func @concat_last_dim_aligned_no_workaround(
      %arg0: tensor<1993728x32xsi32, #layout_a>,
      %arg1: tensor<1993728x32xsi32, #layout_a>)
      -> tensor<1993728x64xsi32, #layout_out> {
    // CHECK-LABEL: func.func @concat_last_dim_aligned_no_workaround
    // CHECK-NOT: "ttnn.slice_static"
    // CHECK: "ttnn.concat"
    // CHECK-SAME: dim = 1
    %0 = "ttnn.concat"(%arg0, %arg1)
        <{dim = 1 : si32}>
        : (tensor<1993728x32xsi32, #layout_a>,
           tensor<1993728x32xsi32, #layout_a>)
       -> tensor<1993728x64xsi32, #layout_out>
    return %0 : tensor<1993728x64xsi32, #layout_out>
  }

  // Test: concat [1993728, 3] + [1993728, 1] along dim=0 (not last dim).
  // Workaround only applies when concat dim is last dim, so should NOT fire.
  func.func @concat_non_last_dim_no_workaround(
      %arg0: tensor<1993728x3xsi32, #layout_a>,
      %arg1: tensor<1993728x3xsi32, #layout_a>)
      -> tensor<3987456x3xsi32, #layout_out> {
    // CHECK-LABEL: func.func @concat_non_last_dim_no_workaround
    // CHECK-NOT: "ttnn.slice_static"
    // CHECK: "ttnn.concat"
    // CHECK-SAME: dim = 0
    %0 = "ttnn.concat"(%arg0, %arg1)
        <{dim = 0 : si32}>
        : (tensor<1993728x3xsi32, #layout_a>,
           tensor<1993728x3xsi32, #layout_a>)
       -> tensor<3987456x3xsi32, #layout_out>
    return %0 : tensor<3987456x3xsi32, #layout_out>
  }
}
