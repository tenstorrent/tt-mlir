// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t
func.func @embedding_1d_tensor(%arg0: tensor<32xf32>, %arg1: tensor<512x128xf32>) -> tensor<32x128xf32> {
  %0 = ttir.empty() : tensor<32x128xf32>
  %1 = "ttir.embedding"(%arg0, %arg1, %0) : (tensor<32xf32>, tensor<512x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
  return %1 : tensor<32x128xf32>
}
