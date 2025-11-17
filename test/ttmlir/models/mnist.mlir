func.func @forward(%arg0: tensor<1x784xf32> {ttir.name = "input_1"}, %arg1: tensor<784x512xf32> {ttir.name = "linear_relu_stack.0.weight"}, %arg2: tensor<1x512xf32> {ttir.name = "linear_relu_stack.0.bias"}, %arg3: tensor<512x512xf32> {ttir.name = "linear_relu_stack.2.weight"}, %arg4: tensor<1x512xf32> {ttir.name = "linear_relu_stack.2.bias"}, %arg5: tensor<512x10xf32> {ttir.name = "linear_relu_stack.4.weight"}, %arg6: tensor<1x10xf32> {ttir.name = "linear_relu_stack.4.bias"}) -> (tensor<1x10xf32> {ttir.name = "MNISTLinear_350.output_add_981"}) {
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x784xf32>, tensor<784x512xf32>) -> tensor<1x512xf32>
  %1 = "ttir.add"(%0, %arg2) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  %2 = "ttir.relu"(%1) : (tensor<1x512xf32>) -> tensor<1x512xf32>
  %3 = "ttir.matmul"(%2, %arg3) : (tensor<1x512xf32>, tensor<512x512xf32>) -> tensor<1x512xf32>
  %4 = "ttir.add"(%3, %arg4) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  %5 = "ttir.relu"(%4) : (tensor<1x512xf32>) -> tensor<1x512xf32>
  %6 = "ttir.matmul"(%5, %arg5) : (tensor<1x512xf32>, tensor<512x10xf32>) -> tensor<1x10xf32>
  %7 = "ttir.add"(%6, %arg6) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  return %7 : tensor<1x10xf32>
}