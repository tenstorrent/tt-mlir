// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @add(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK: "ttnn.zeros"
    %0 = ttir.empty() : tensor<32x32xf32>
    // CHECK: %{{.*}} = call @hoisted_ttir_add_32x32_32x32_32x32_func_decl
    %1 = "ttir.add"(%arg0, %arg1, %0) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
