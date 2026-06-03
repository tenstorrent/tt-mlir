// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 enable-greedy-optimizer=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Regression test for the prepare_conv3d_weights → to_layout → conv3d chain
// under the optimizer pipeline. tt-metal's prepare_conv3d_weights runtime
// kernel always returns a ROW_MAJOR tensor (its last op is `reshape`, which
// preserves ROW_MAJOR). The downstream Conv3dOp's weight workaround forces
// TILE. If the optimizer's layout propagation elides the to_layout between
// them, the flatbuffer encodes TILE for the prepare op output while the
// runtime produces ROW_MAJOR — runtime debug assert at
// `runtime/lib/ttnn/debug/debug_apis.cpp:31` fires with
// "Layout mismatch, expected TILE, got ROW_MAJOR" and silicon execution
// breaks.
//
// TTNNPrepareConv3dWeights enforces the invariant. This test pins
// the IR shape: prepare's result must be ROW_MAJOR, immediately followed
// by a to_layout that converts to TILE for the consumer Conv3dOp.
module {
  func.func @prepare_weights_layout(
      %arg0: tensor<1x8x28x28x128xbf16>,
      %arg1: tensor<32x128x3x3x3xbf16>)
      -> tensor<1x6x26x26x32xbf16> {
    // prepare_conv3d_weights must produce a ROW_MAJOR memref (no `tile<>`
    // in its memref element type), followed by a to_layout op converting
    // to TILE for the consumer Conv3dOp.
    //
    // The IR shape is:
    //   %1 = prepare_conv3d_weights(...) -> tensor<..., #row_major_layout>
    //   "ttnn.deallocate"(%argweight) ...     // optional dealloc
    //   %2 = to_layout(%1) -> tensor<..., #tile_layout>
    //   %3 = conv3d(input, %2, ...)
    //
    // We assert structurally: a `to_layout` to tile must appear between
    // prepare and conv3d, and the prepare op's result memref must NOT be
    // tiled.
    //
    // CHECK-DAG: #[[PREPARE_RM_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<3456x32xbf16, #dram>
    // CHECK-DAG: #[[WEIGHTS_TILE_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<108x1x!ttcore.tile<32x32, bf16>, #dram>
    // CHECK: %[[PREPARE_WEIGHTS:.*]] = "ttnn.prepare_conv3d_weights"{{.*}} -> tensor<3456x32xbf16, #[[PREPARE_RM_LAYOUT]]>
    // CHECK: %[[TILED_WEIGHTS:.*]] = "ttnn.to_layout"(%[[PREPARE_WEIGHTS]]){{.*}} -> tensor<3456x32xbf16, #[[WEIGHTS_TILE_LAYOUT]]>
    // CHECK: "ttnn.conv3d"({{.*}}, %[[TILED_WEIGHTS]],
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x128xbf16>, tensor<32x128x3x3x3xbf16>)
        -> tensor<1x6x26x26x32xbf16>
    return %0 : tensor<1x6x26x26x32xbf16>
  }
}
