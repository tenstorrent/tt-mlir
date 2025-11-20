// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s
// Need to ensure that model is valid MLIR module

module @SimpleModel attributes {} {
  func.func @forward(%arg0: tensor<1x784xf32> {ttir.name = "input_1"}, %arg1: tensor<10x784xf32> {ttir.name = "linear.weight"}, %arg2: tensor<10xf32> {ttir.name = "linear.bias"}) -> (tensor<1x10xf32> {ttir.name = "SimpleModel_472.output_softmax_1495"}) {
    %0 = "ttir.transpose"(%arg1) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<10x784xf32>) -> tensor<784x10xf32>
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<1x784xf32>, tensor<784x10xf32>) -> tensor<1x10xf32>
    %2 = "ttir.add"(%1, %arg2) : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    %3 = "ttir.softmax"(%2) <{dimension = -1 : si32}> : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %3 : tensor<1x10xf32>
  }
  func.func @backward(%arg0: tensor<1x10xf32> {ttir.name = "loss_SimpleModel_472.output_softmax_1495"}, %arg1: tensor<1x10xf32> {ttir.name = "SimpleModel_472.output_softmax_1495"}, %arg2: tensor<1x784xf32> {ttir.name = "input_1"}) -> (tensor<1x10xf32> {ttir.name = "grad_acc_linear.bias_grad_accumulator"}, tensor<10x784xf32> {ttir.name = "grad_acc_linear.weight_grad_accumulator"}) {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %1 = "ttir.sum"(%0) <{keep_dim = true}> : (tensor<1x10xf32>) -> tensor<1x1xf32>
    %2 = "ttir.subtract"(%arg0, %1) : (tensor<1x10xf32>, tensor<1x1xf32>) -> tensor<1x10xf32>
    %3 = "ttir.multiply"(%2, %arg1) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %4 = "ttir.transpose"(%arg2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x784xf32>) -> tensor<784x1xf32>
    %5 = "ttir.matmul"(%4, %3) : (tensor<784x1xf32>, tensor<1x10xf32>) -> tensor<784x10xf32>
    %6 = "ttir.transpose"(%5) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<784x10xf32>) -> tensor<10x784xf32>
    return %3, %6 : tensor<1x10xf32>, tensor<10x784xf32>
  }
}
