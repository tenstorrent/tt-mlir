// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-rm-layout-propagation %s | FileCheck %s

// Regression test for the data-parallel gemma4 runtime abort
// ("Layout mismatch, expected TILE, got ROW_MAJOR").
//
// TTNNRowMajorLayoutPropagation seeds from integer Input arguments and removes
// the redundant tilizing to_layout on %arg0, flipping the producer to
// ROW_MAJOR. The consumer here is an all_gather: it is OpModelExempt and
// layout-preserving, so propagation must stop there. Before the fix, the
// all_gather was left consuming a ROW_MAJOR operand while still declaring a
// TILE result -- a mismatch only caught at runtime. The fix re-tilizes the
// operand so the all_gather sees a TILE input matching its TILE output.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1xui32, #dram>, <interleaved>>
#ttnn_layout_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>

module @all_gather_rm_reconcile attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func public @main
  func.func public @main(%arg0: tensor<1x1xui32, #ttnn_layout_rm> {ttcore.argument_type = #ttcore.argument_type<input>}) -> (tensor<1x8xui32, #ttnn_layout_tile> {}) attributes {tt.function_type = "forward_device"} {
    // The redundant tilizing to_layout on the input is removed by the pass, but
    // a tilizing to_layout must remain (re-inserted) feeding the all_gather so
    // its operand is TILE rather than the now-RowMajor %arg0. Without the fix,
    // the all_gather would consume %arg0 directly (RowMajor in, TILE out).
    // CHECK: %[[TILE:[^ ]+]] = "ttnn.to_layout"(%arg0)
    // CHECK: "ttnn.all_gather"(%[[TILE]])
    %0 = "ttnn.to_layout"(%arg0) : (tensor<1x1xui32, #ttnn_layout_rm>) -> tensor<1x1xui32, #ttnn_layout_tile>
    %1 = "ttnn.all_gather"(%0) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32, num_links = 1 : ui32}> : (tensor<1x1xui32, #ttnn_layout_tile>) -> tensor<1x8xui32, #ttnn_layout_tile>
    return %1 : tensor<1x8xui32, #ttnn_layout_tile>
  }
}
