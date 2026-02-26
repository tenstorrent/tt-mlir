// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-d2m-fusing %s | FileCheck %s
//
// Pattern: SLICE_0 -> ADD_0 (fused) -> SLICE_1 -> ADD_1 (fused) -> MULTIPLY (fused).
// firstOp = ADD_0. Inputs = SLICE_0, SLICE_1, scalar. lastInputDefiner = SLICE_1.
// The subgraph must be inserted after SLICE_1 (and after BUFFER), not at ADD_0.

#l1 = #ttnn.buffer_type<l1>
#layout_256 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#layout_128 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

module {
  func.func @d2m_fusing_two_legs_insertion_dominance(%arg0: tensor<64x256xbf16, #layout_256>, %arg1: tensor<64x128xbf16, #layout_128>) -> tensor<64x128xbf16, #layout_128> {
    %0 = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 128 : i32], step = [1 : i32, 1 : i32]}> : (tensor<64x256xbf16, #layout_256>) -> tensor<64x128xbf16, #layout_128>
    %1 = "ttnn.add"(%0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xbf16, #layout_128>, tensor<64x128xbf16, #layout_128>) -> tensor<64x128xbf16, #layout_128>
    %2 = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 128 : i32], ends = [64 : i32, 256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<64x256xbf16, #layout_256>) -> tensor<64x128xbf16, #layout_128>
    %3 = "ttnn.add"(%2, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xbf16, #layout_128>, tensor<64x128xbf16, #layout_128>) -> tensor<64x128xbf16, #layout_128>
    %4 = "ttnn.multiply"(%1, %3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xbf16, #layout_128>, tensor<64x128xbf16, #layout_128>) -> tensor<64x128xbf16, #layout_128>
    return %4 : tensor<64x128xbf16, #layout_128>
  }
}
// CHECK-LABEL: func.func @d2m_fusing_two_legs_insertion_dominance
// CHECK: %[[SLICE_0:.*]] = "ttnn.slice_static"(%arg0)
// CHECK: %[[SLICE_1:.*]] = "ttnn.slice_static"(%arg0)
// CHECK: %[[BUFFER:.*]] = "ttnn.empty"
// CHECK: %[[D2M_OUTPUT:.*]] = ttnn.d2m_subgraph @d2m_subgraph_0
// CHECK: return %[[D2M_OUTPUT]]

// CHECK: func.func private @d2m_subgraph_0
// CHECK: %[[ADD_0:.*]] = "ttnn.add"
// CHECK: %[[ADD_1:.*]] = "ttnn.add"
// CHECK: %[[MULTIPLY:.*]] = "ttnn.multiply"(%[[ADD_0]], %[[ADD_1]])
// CHECK: return %[[MULTIPLY]]
