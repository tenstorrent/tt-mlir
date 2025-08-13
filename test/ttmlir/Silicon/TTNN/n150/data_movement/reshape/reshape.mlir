// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @reshape(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  %0 = ttir.empty() : tensor<2x4x32x32xbf16>
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 32: i32]}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  // CHECK: "ttnn.reshape"
  return %1 : tensor<2x4x32x32xbf16>
}

func.func @reshape_scalar(%arg0: tensor<f32>) -> tensor<1xf32> {
  %0 = ttir.empty() : tensor<1xf32>
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [1: i32]}> : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  // CHECK: "ttnn.reshape"
  return %1 : tensor<1xf32>
}

func.func @reshape_to_scalar(%arg0: tensor<1xf32>) -> tensor<f32> {
  %0 = ttir.empty() : tensor<f32>
  %1 = "ttir.reshape"(%arg0, %0) <{shape = []}> : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: "ttnn.reshape"
  return %1 : tensor<f32>
}

func.func @reshape_ui8(%arg0: tensor<71xui8>) -> tensor<71x1xui8> {
  %0 = ttir.empty() : tensor<71x1xui8>
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [71: i32, 1: i32]}> : (tensor<71xui8>, tensor<71x1xui8>) -> tensor<71x1xui8>
  // CHECK: "ttnn.typecast"
  // CHECK: "ttnn.reshape"
  // CHECK: "ttnn.typecast"
  return %1 : tensor<71x1xui8>
}
