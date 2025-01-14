// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-attach-metal-layout %s | FileCheck %s
// CHECK-LABEL: func.func @maximum(
// CHECK-SAME: %arg0: tensor<64x128xf32, #[[LAYOUT:layout]]>
// CHECK-SAME: %arg1: tensor<64x128xf32, #[[LAYOUT]]>
// CHECK-SAME: ) -> tensor<64x128xf32, #layout>
func.func @maximum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C0:.*]] = tensor.empty() : tensor<64x128xf32, #[[LAYOUT]]>
  // CHECK: %[[C1:.*]] = "ttir.maximum"(%arg0, %arg1, %[[C0]]) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32, #[[LAYOUT]]>, tensor<64x128xf32, #[[LAYOUT]]>, tensor<64x128xf32, #[[LAYOUT]]>) -> tensor<64x128xf32, #[[LAYOUT]]>
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = "ttir.maximum"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: return %[[C1]] : tensor<64x128xf32, #[[LAYOUT]]>
  return %1 : tensor<64x128xf32>
}

#l1_ = #tt.memory_space<l1>
#layout1 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <4x4>, memref<64x96xf32, #l1_>>
#layout2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <4x1>, memref<64x32xf32, #l1_>>
// CHECK-LABEL: func.func @reduceW(
// CHECK-SAME: %arg0: tensor<256x384xf32, #[[LAYOUT1:layout1]]>
// CHECK-SAME: ) -> tensor<256x32xf32, #[[LAYOUT2:layout2]]>
func.func @reduceW(%arg0: tensor<256x384xf32, #layout1>) -> tensor<256x32xf32, #layout2> {
  // CHECK: %[[C0:.*]] = tensor.empty() : tensor<256x32xf32, #[[LAYOUT2]]>
  // CHECK: %[[C1:.*]] = "ttir.sum"(%arg0, %0) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<256x384xf32, #[[LAYOUT1]]>, tensor<256x32xf32, #[[LAYOUT2]]>) -> tensor<256x32xf32, #[[LAYOUT2]]>
  %0 = tensor.empty() : tensor<256x32xf32, #layout2>
  %1 = "ttir.sum"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>,
                               dim_arg = [-1: i32],
                               keep_dim = true}> :
    (tensor<256x384xf32, #layout1>, tensor<256x32xf32, #layout2>) -> tensor<256x32xf32, #layout2>
  // CHECK: return %[[C1]] : tensor<256x32xf32, #[[LAYOUT2]]>
  return %1 : tensor<256x32xf32, #layout2>
}
