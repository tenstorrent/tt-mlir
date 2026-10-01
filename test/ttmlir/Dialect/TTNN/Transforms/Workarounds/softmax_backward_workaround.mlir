// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @softmax_backward(%y: tensor<1x1x32x64xf32>, %grad: tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32> {
    // CHECK-DAG: #[[LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}!ttcore.tile<32x32, f32>
    // CHECK-LABEL: func.func @softmax_backward
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[R:.*]] = "ttnn.softmax_backward"
    // CHECK-SAME: -> tensor<1x1x32x64xf32, #[[LAYOUT]]>
    // CHECK-NOT: "ttnn.typecast"
    %0 = "ttnn.softmax_backward"(%y, %grad) <{dimension = -1 : si32}> : (tensor<1x1x32x64xf32>, tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32>
    return %0 : tensor<1x1x32x64xf32>
  }
}
