// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttmetal-backend-pipeline="system-desc-path=%system_desc_path%"  %s | FileCheck %s
#l1_ = #tt.memory_space<l1>
#layout1 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <4x4>, memref<64x96xf32, #l1_>>
#layout2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <4x1>, memref<64x32xf32, #l1_>>

func.func @reduceW(%arg0: tensor<256x384xf32, #layout1>) -> tensor<256x32xf32, #layout2> {
  %0 = tensor.empty() : tensor<256x32xf32, #layout2>
  // CHECK: %[[C:.*]] = "ttmetal.enqueue_program"[[C:.*]]
  %1 = "ttir.sum"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>,
                               dim_arg = [-1: i32],
                               keep_dim = true}> :
    (tensor<256x384xf32, #layout1>, tensor<256x32xf32, #layout2>) -> tensor<256x32xf32, #layout2>
  return %1 : tensor<256x32xf32, #layout2>
}

#layout3 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x4>, memref<32x96xf32, #l1_>>
func.func @reduceH(%arg0: tensor<256x384xf32, #layout1>) -> tensor<32x384xf32, #layout3> {
  %0 = tensor.empty() : tensor<32x384xf32, #layout3>
  // CHECK: %[[C:.*]] = "ttmetal.enqueue_program"[[C:.*]]
  %1 = "ttir.sum"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>,
                               dim_arg = [-2: i32],
                               keep_dim = true}> :
    (tensor<256x384xf32, #layout1>, tensor<32x384xf32, #layout3>) -> tensor<32x384xf32, #layout3>
  return %1 : tensor<32x384xf32, #layout3>
}

#layout4 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<32x32xf32, #l1_>>
func.func @reduceWH(%arg0: tensor<256x384xf32, #layout1>) -> tensor<32x32xf32, #layout4> {
  %0 = tensor.empty() : tensor<32x32xf32, #layout4>
  // CHECK: %[[C:.*]] = "ttmetal.enqueue_program"[[C:.*]]
  %1 = "ttir.sum"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>,
                               dim_arg = [-1: i32, -2: i32],
                               keep_dim = true}> :
    (tensor<256x384xf32, #layout1>, tensor<32x32xf32, #layout4>) -> tensor<32x32xf32, #layout4>
  return %1 : tensor<32x32xf32, #layout4>
}

func.func @maxReduceWH(%arg0: tensor<256x384xf32, #layout1>) -> tensor<32x32xf32, #layout4> {
  %0 = tensor.empty() : tensor<32x32xf32, #layout4>
  // CHECK: %[[C:.*]] = "ttmetal.enqueue_program"[[C:.*]]
  %1 = "ttir.max" (%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>,
                               dim_arg = [-1: i32, -2: i32],
                               keep_dim = true}> :
    (tensor<256x384xf32, #layout1>, tensor<32x32xf32, #layout4>) -> tensor<32x32xf32, #layout4>
  return %1 : tensor<32x32xf32, #layout4>
}
