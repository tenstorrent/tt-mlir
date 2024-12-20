// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
// For ND tensors, shape dimensions greater than 4 should be 1, shape at index0 is 2

module @ReshapeTest attributes {} {
  func.func @forward(%arg0: tensor<4x49x384xf32> {ttir.name = "inp_1"}, %arg1: tensor<4x49x384xf32> {ttir.name = "inp_2"}) -> (tensor<196x384xf32> {ttir.name = "ReshapeTest_295.output_reshape_743"}) {
    %0 = tensor.empty() : tensor<4x49x384xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<4x49x384xf32>, tensor<4x49x384xf32>, tensor<4x49x384xf32>) -> tensor<4x49x384xf32>
    %2 = tensor.empty() : tensor<2x2x7x7x384xf32>
    %3 = "ttir.reshape"(%1, %2) <{shape = [2 : i32, 2 : i32, 7 : i32, 7 : i32, 384 : i32]}> : (tensor<4x49x384xf32>, tensor<2x2x7x7x384xf32>) -> tensor<2x2x7x7x384xf32>
    %4 = tensor.empty() : tensor<2x7x2x7x384xf32>
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    %5 = "ttir.transpose"(%3, %4) <{dim0 = -4 : si32, dim1 = -3 : si32}> : (tensor<2x2x7x7x384xf32>, tensor<2x7x2x7x384xf32>) -> tensor<2x7x2x7x384xf32>
    %6 = tensor.empty() : tensor<196x384xf32>
    %7 = "ttir.reshape"(%5, %6) <{shape = [196 : i32, 384 : i32]}> : (tensor<2x7x2x7x384xf32>, tensor<196x384xf32>) -> tensor<196x384xf32>
    return %7 : tensor<196x384xf32>
  }
}
