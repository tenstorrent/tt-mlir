// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module @MNISTLinear attributes {} {
  func.func @forward(%arg0: tensor<1x784xf32> {ttir.name = "input_1"}, %arg1: tensor<784x512xf32> {ttir.name = "linear_relu_stack.0.weight"}, %arg2: tensor<512xf32> {ttir.name = "linear_relu_stack.0.bias"}, %arg3: tensor<512x512xf32> {ttir.name = "linear_relu_stack.2.weight"}, %arg4: tensor<512xf32> {ttir.name = "linear_relu_stack.2.bias"}, %arg5: tensor<512x10xf32> {ttir.name = "linear_relu_stack.4.weight"}, %arg6: tensor<10xf32> {ttir.name = "linear_relu_stack.4.bias"}) -> (tensor<1x10xf32> {ttir.name = "MNISTLinear_350.output_add_981"}) {
    %0 = tensor.empty() : tensor<1x512xf32>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x784xf32>, tensor<784x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %2 = tensor.empty() : tensor<1x512xf32>
    %3 = "ttir.add"(%1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512xf32>, tensor<512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %4 = tensor.empty() : tensor<1x512xf32>
    %5 = "ttir.relu"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %6 = tensor.empty() : tensor<1x512xf32>
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<1x512xf32>, tensor<512x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %8 = tensor.empty() : tensor<1x512xf32>
    %9 = "ttir.add"(%7, %arg4, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x512xf32>, tensor<512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %10 = tensor.empty() : tensor<1x512xf32>
    %11 = "ttir.relu"(%9, %10) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %12 = tensor.empty() : tensor<1x10xf32>
    %13 = "ttir.matmul"(%11, %arg5, %12) : (tensor<1x512xf32>, tensor<512x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %14 = tensor.empty() : tensor<1x10xf32>
    %15 = "ttir.add"(%13, %arg6, %14) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x10xf32>, tensor<10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %15 : tensor<1x10xf32>
  }
}
