// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @forward(%arg0: tensor<512x1024xf32>) -> tensor<f32> {
  %0 = ttir.empty() : tensor<f32>
  %1 = "ttir.min"(%arg0, %0) <{keep_dim = false}> : (tensor<512x1024xf32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}
