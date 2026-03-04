// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // le(a, b) -> ge(b, a)
  func.func @le_to_ge(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-NOT: "ttir.le"
    // CHECK: "ttir.ge"(%arg1, %arg0)
    %1 = "ttir.le"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }

  // lt(a, b) -> gt(b, a)
  func.func @lt_to_gt(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-NOT: "ttir.lt"
    // CHECK: "ttir.gt"(%arg1, %arg0)
    %1 = "ttir.lt"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }

  // le with integer tensors
  func.func @le_to_ge_integer(%arg0: tensor<32x32xi32>, %arg1: tensor<32x32xi32>) -> tensor<32x32xi32> {
    // CHECK-NOT: "ttir.le"
    // CHECK: "ttir.ge"(%arg1, %arg0)
    %1 = "ttir.le"(%arg0, %arg1) : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    return %1 : tensor<32x32xi32>
  }

  // lt with integer tensors
  func.func @lt_to_gt_integer(%arg0: tensor<32x32xi32>, %arg1: tensor<32x32xi32>) -> tensor<32x32xi32> {
    // CHECK-NOT: "ttir.lt"
    // CHECK: "ttir.gt"(%arg1, %arg0)
    %1 = "ttir.lt"(%arg0, %arg1) : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    return %1 : tensor<32x32xi32>
  }
}
