// RUN: ttmlir-opt --remove-dead-values -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    // CHECK: = "ttir.multiply"
    %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = ttir.empty() : tensor<64x128xf32>
    // CHECK-NOT: = "ttir.add"
    %3 = "ttir.add"(%arg0, %arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %4 = ttir.empty() : tensor<64x128xf32>
    // CHECK-NOT: = "ttir.subtract"
    %5 = "ttir.subtract"(%arg0, %arg1, %4) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %6 = ttir.empty() : tensor<64x128xf32>
    // CHECK-NOT: = "ttir.div"
    %7 = "ttir.div"(%arg0, %arg1, %6) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %8 = ttir.empty() : tensor<64x128xf32>
    // CHECK-NOT: = "ttir.eq"
    %9 = "ttir.eq"(%arg0, %arg1, %8) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
