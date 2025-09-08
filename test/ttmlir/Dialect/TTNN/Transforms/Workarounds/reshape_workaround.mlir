// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// reshape workaround is a workaround pass that typecasts before and after ui8 reshape
func.func @reshape_ui8(%arg0: tensor<71x1xui8>) -> tensor<71xui8> {
  %0 = ttir.empty() : tensor<71xui8>
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [71: i32]}> : (tensor<71x1xui8>, tensor<71xui8>) -> tensor<71xui8>
  // CHECK: "ttnn.typecast"
  // CHECK: "ttnn.reshape"
  // CHECK: "ttnn.typecast"
  return %1 : tensor<71xui8>
}
