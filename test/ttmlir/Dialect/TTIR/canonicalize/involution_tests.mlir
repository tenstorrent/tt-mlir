// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @involution_two_in_the_row(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK-NOT: "ttir.neg"
    %0 = ttir.empty() : tensor<64x64xf32>
    %1 = "ttir.neg"(%arg0, %0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = ttir.empty() : tensor<64x64xf32>
    %3 = "ttir.neg"(%1, %2) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }

  func.func @involution_three_in_the_row(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.neg"
    // CHECK-NOT: "ttir.neg"
    %0 = ttir.empty() : tensor<64x64xf32>
    %1 = "ttir.neg"(%arg0, %0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = ttir.empty() : tensor<64x64xf32>
    %3 = "ttir.neg"(%1, %2) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %4 = ttir.empty() : tensor<64x64xf32>
    %5 = "ttir.neg"(%3, %4) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %5 : tensor<64x64xf32>
  }
}
