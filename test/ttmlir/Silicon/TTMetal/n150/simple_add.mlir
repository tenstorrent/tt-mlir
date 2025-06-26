// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% override-device-shape=1,1" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: "ttmetal.create_buffer"
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: "ttmetal.enqueue_program"
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: "ttmetal.enqueue_read_buffer"
  // CHECK: "ttmetal.finish"
  return %1 : tensor<64x128xf32>
}

func.func @add_unaligned(%arg0: tensor<33x128xf32>, %arg1: tensor<33x128xf32>) -> tensor<33x128xf32> {
    // CHECK: "ttmetal.create_buffer"
    %0 = ttir.empty() : tensor<33x128xf32>
    // CHECK: "ttmetal.enqueue_program"
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<33x128xf32>, tensor<33x128xf32>, tensor<33x128xf32>) -> tensor<33x128xf32>
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: "ttmetal.finish"
    return %1 : tensor<33x128xf32>
}

func.func @add_3d(%arg0: tensor<2x32x128xf32>, %arg1: tensor<2x32x128xf32>) -> tensor<2x32x128xf32> {
    // CHECK: "ttmetal.create_buffer"
    %0 = ttir.empty() : tensor<2x32x128xf32>
    // CHECK: "ttmetal.enqueue_program"
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x32x128xf32>, tensor<2x32x128xf32>, tensor<2x32x128xf32>) -> tensor<2x32x128xf32>
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: "ttmetal.finish"
    return %1 : tensor<2x32x128xf32>
}

func.func @add_3d_unaligned(%arg0: tensor<2x33x128xf32>, %arg1: tensor<2x33x128xf32>) -> tensor<2x33x128xf32> {
    // CHECK: "ttmetal.create_buffer"
    %0 = ttir.empty() : tensor<2x33x128xf32>
    // CHECK: "ttmetal.enqueue_program"
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x33x128xf32>, tensor<2x33x128xf32>, tensor<2x33x128xf32>) -> tensor<2x33x128xf32>
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: "ttmetal.finish"
    return %1 : tensor<2x33x128xf32>
}
