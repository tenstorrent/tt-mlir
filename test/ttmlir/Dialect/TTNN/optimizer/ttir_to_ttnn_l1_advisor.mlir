// RUN: ttmlir-opt --ttir-to-ttnn-l1-advisor="system-desc-path=%system_desc_path% optimization-level=0" %s | FileCheck %s

module {
  func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func @add
    // CHECK: "ttnn.add"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
