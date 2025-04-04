// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module {
  func.func @forward(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>> {
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
  }
}
