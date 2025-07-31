// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_concat_canonicalize(%arg0: tensor<1x2x1xf32>, %arg1: tensor<1x2x0xf32>) -> tensor<1x2x2xf32> {
  // CHECK-LABEL: test_concat_canonicalize
  %0 = ttir.empty() : tensor<1x2x2xf32>
  // CHECK: %{{[0-9]+}} = "ttir.concat"(%arg0, %arg0, %0) <{dim = 2 : si32}> : (tensor<1x2x1xf32>, tensor<1x2x1xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  %1 = "ttir.concat"(%arg0, %arg1, %arg0, %0) <{dim = 2 : si32}> : (tensor<1x2x1xf32>, tensor<1x2x0xf32>, tensor<1x2x1xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  return %1 : tensor<1x2x2xf32>
}
