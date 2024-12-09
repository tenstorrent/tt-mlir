// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | ttmlir-translate --ttnn-to-flatbuffer
func.func @embedding_1d_tensor(%arg0: tensor<32xf32>, %arg1: tensor<512x128xf32>) -> tensor<32x128xf32> {
  %0 = tensor.empty() : tensor<32x128xf32>
  %1 = "ttir.embedding"(%arg0, %arg1, %0) : (tensor<32xf32>, tensor<512x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
  return %1 : tensor<32x128xf32>
}
