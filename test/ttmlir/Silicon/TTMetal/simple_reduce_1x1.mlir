// RUN: ttmlir-opt --ttir-to-ttmetal-backend-pipeline="system-desc-path=%system_desc_path%"  %s | FileCheck %s
#l1_ = #tt.memory_space<l1>

func.func @reduceW(%arg0: tensor<64x256xf32>) -> tensor<64x32xf32> {
  %0 = tensor.empty() : tensor<64x32xf32>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.sum"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>,
                               dim = [-1: i32],
                               keep_dim = true}> :
    (tensor<64x256xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  return %1 : tensor<64x32xf32>
}

func.func @reduceH(%arg0: tensor<256x64xf32>) -> tensor<32x64xf32> {
  %0 = tensor.empty() : tensor<32x64xf32>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.sum"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>,
                               dim = [-2: i32],
                               keep_dim = true}> :
    (tensor<256x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
  return %1 : tensor<32x64xf32>
}

func.func @reduceWH(%arg0: tensor<256x64xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.sum"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>,
                               dim = [-1: i32, -2: i32],
                               keep_dim = true}> :
    (tensor<256x64xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
