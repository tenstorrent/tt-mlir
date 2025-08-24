// UNSUPPORTED: true
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t %s
// RUN: FileCheck %s --input-file=%t
#l1_ = #ttcore.memory_space<l1>

func.func @reduceW(%arg0: tensor<64x256xf32>) -> tensor<64x32xf32> {
  %0 = ttir.empty() : tensor<64x32xf32>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-1: i32],
                               keep_dim = true}> :
    (tensor<64x256xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  return %1 : tensor<64x32xf32>
}

func.func @reduceH(%arg0: tensor<256x64xf32>) -> tensor<32x64xf32> {
  %0 = ttir.empty() : tensor<32x64xf32>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-2: i32],
                               keep_dim = true}> :
    (tensor<256x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
  return %1 : tensor<32x64xf32>
}

func.func @reduceWH(%arg0: tensor<256x64xf32>) -> tensor<32x32xf32> {
  %0 = ttir.empty() : tensor<32x32xf32>
  // CHECK: = "ttmetal.enqueue_program"
  %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-1: i32, -2: i32],
                               keep_dim = true}> :
    (tensor<256x64xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
