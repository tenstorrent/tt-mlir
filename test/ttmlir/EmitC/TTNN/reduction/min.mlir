// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @forward(%arg0: tensor<512x1024xf32>) -> tensor<1xf32> {
  %0 = ttir.empty() : tensor<1xf32>
  %1 = "ttir.min"(%arg0, %0) <{keep_dim = false}> : (tensor<512x1024xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
}
