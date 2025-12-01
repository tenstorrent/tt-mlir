// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_concat_canonicalize_multiple_non_empty_inputs(%arg0: tensor<1x2x1xf32>, %arg1: tensor<1x2x0xf32>) -> tensor<1x2x2xf32> {
  // CHECK-LABEL: @test_concat_canonicalize_multiple_non_empty_inputs
  // CHECK: %{{[0-9]+}} = "ttir.concat"(%arg0, %arg0) <{dim = 2 : si32}> : (tensor<1x2x1xf32>, tensor<1x2x1xf32>) -> tensor<1x2x2xf32>
  %1 = "ttir.concat"(%arg0, %arg1, %arg0) <{dim = 2 : si32}> : (tensor<1x2x1xf32>, tensor<1x2x0xf32>, tensor<1x2x1xf32>) -> tensor<1x2x2xf32>
  return %1 : tensor<1x2x2xf32>
}

func.func @test_concat_canonicalize_one_non_empty_input(%arg0: tensor<1x2x0xf32>, %arg1: tensor<1x2x2xf32>) -> tensor<1x2x2xf32> {
  // CHECK-LABEL: @test_concat_canonicalize_one_non_empty_input
  // CHECK-NOT: "ttir.concat"
  %1 = "ttir.concat"(%arg0, %arg1, %arg0) <{dim = 2 : si32}> : (tensor<1x2x0xf32>, tensor<1x2x2xf32>, tensor<1x2x0xf32>) -> tensor<1x2x2xf32>
  // CHECK: return %arg1 : tensor<1x2x2xf32>
  return %1 : tensor<1x2x2xf32>
}

func.func @test_concat_canonicalize_all_empty_inputs(%arg0: tensor<1x2x0xf32>, %arg1: tensor<1x2x0xf32>) -> tensor<1x2x0xf32> {
  // CHECK-LABEL: @test_concat_canonicalize_all_empty
  // CHECK-NOT: "ttir.concat"
  %1 = "ttir.concat"(%arg0, %arg1, %arg0) <{dim = 2 : si32}> : (tensor<1x2x0xf32>, tensor<1x2x0xf32>, tensor<1x2x0xf32>) -> tensor<1x2x0xf32>
  // CHECK: return %arg0 : tensor<1x2x0xf32>
  return %1 : tensor<1x2x0xf32>
}

func.func @test_concat_canonicalize_one_input(%arg0: tensor<1x2x2xf32>) -> tensor<1x2x2xf32> {
  // CHECK-LABEL: @test_concat_canonicalize_one_input
  // CHECK-NOT: "ttir.concat"
  %1 = "ttir.concat"(%arg0) <{dim = 2 : si32}> : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  // CHECK: return %arg0 : tensor<1x2x2xf32>
  return %1 : tensor<1x2x2xf32>
}
