// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% override-device-shape=1,1 collapse-tensors-2d=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %basename_t.ttm %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func @simple_outer_permute(%arg0: tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32> {
  %0 = ttir.empty() : tensor<1x32x32x32xf32>
  %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32>
  // requires a compute op after to invoke compute kernels
  %2 = ttir.empty() : tensor<1x32x32x32xf32>
  %3 = "ttir.abs"(%1, %2) : (tensor<1x32x32x32xf32>,tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32>
  // CHECK: call_opaque "noc_async_read"
  // CHECK: call_opaque "noc_async_read_barrier"
  return %3 : tensor<1x32x32x32xf32>
}
