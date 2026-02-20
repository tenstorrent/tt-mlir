// UNSUPPORTED: true
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir
#l1_ = #ttcore.memory_space<l1>
#layout1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <4x4>, memref<64x96xf32, #l1_>>
#layout2 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <4x1>, memref<64x32xf32, #l1_>>

func.func @reduceW(%arg0: tensor<256x384xf32, #layout1>) -> tensor<256x32xf32, #layout2> {
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.sum"(%arg0) <{dim_arg = [-1: i32],
                               keep_dim = true}> :
    (tensor<256x384xf32, #layout1>) -> tensor<256x32xf32, #layout2>
  return %1 : tensor<256x32xf32, #layout2>
}

#layout3 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x4>, memref<32x96xf32, #l1_>>
func.func @reduceH(%arg0: tensor<256x384xf32, #layout1>) -> tensor<32x384xf32, #layout3> {
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.sum"(%arg0) <{dim_arg = [-2: i32],
                               keep_dim = true}> :
    (tensor<256x384xf32, #layout1>) -> tensor<32x384xf32, #layout3>
  return %1 : tensor<32x384xf32, #layout3>
}

#layout4 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<32x32xf32, #l1_>>
func.func @reduceWH(%arg0: tensor<256x384xf32, #layout1>) -> tensor<32x32xf32, #layout4> {
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.sum"(%arg0) <{dim_arg = [-1: i32, -2: i32],
                               keep_dim = true}> :
    (tensor<256x384xf32, #layout1>) -> tensor<32x32xf32, #layout4>
  return %1 : tensor<32x32xf32, #layout4>
}

func.func @maxReduceWH(%arg0: tensor<256x384xf32, #layout1>) -> tensor<32x32xf32, #layout4> {
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.max" (%arg0) <{dim_arg = [-1: i32, -2: i32],
                                keep_dim = true}> :
    (tensor<256x384xf32, #layout1>) -> tensor<32x32xf32, #layout4>
  return %1 : tensor<32x32xf32, #layout4>
}
