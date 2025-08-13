// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
func.func @reshape_ui8(%arg0: tensor<71xui8>) -> tensor<71x1xui8> {
  %0 = ttir.empty() : tensor<71x1xui8>
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [71: i32, 1: i32]}> : (tensor<71xui8>, tensor<71x1xui8>) -> tensor<71x1xui8>
  // CHECK: "ttnn.typecast"
  // CHECK: "ttnn.reshape"
  // CHECK: "ttnn.typecast"
  return %1 : tensor<71x1xui8>
}
}
