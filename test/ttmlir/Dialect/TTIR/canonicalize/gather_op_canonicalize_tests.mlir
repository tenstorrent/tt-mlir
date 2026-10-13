// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Do not reassociate gathers with different dims.
  func.func @gather_chain_different_dims(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xui32>, %arg2: tensor<3x8xui32>) -> tensor<3x8xf32> {
    // CHECK-LABEL: @gather_chain_different_dims
    // CHECK: "ttir.gather"(%arg0, %arg1) <{dim = 1 : i32}>
    // CHECK: "ttir.gather"(%{{.*}}, %arg2) <{dim = 0 : i32}>
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 1 : i32}> : (tensor<8x8xf32>, tensor<8x8xui32>) -> tensor<8x8xf32>
    %1 = "ttir.gather"(%0, %arg2) <{dim = 0 : i32}> : (tensor<8x8xf32>, tensor<3x8xui32>) -> tensor<3x8xf32>
    return %1 : tensor<3x8xf32>
  }

  // Reassociate with 3D tensors.
  func.func @gather_chain_3d(%arg0: tensor<4x8x6xf32>, %arg1: tensor<4x8x6xui32>, %arg2: tensor<4x3x6xui32>) -> tensor<4x3x6xf32> {
    // CHECK-LABEL: @gather_chain_3d
    // CHECK: "ttir.gather"(%arg1, %arg2) <{dim = 1 : i32}> : (tensor<4x8x6xui32>, tensor<4x3x6xui32>) -> tensor<4x3x6xui32>
    // CHECK: "ttir.gather"(%arg0, %{{.*}}) <{dim = 1 : i32}> : (tensor<4x8x6xf32>, tensor<4x3x6xui32>) -> tensor<4x3x6xf32>
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 1 : i32}> : (tensor<4x8x6xf32>, tensor<4x8x6xui32>) -> tensor<4x8x6xf32>
    %1 = "ttir.gather"(%0, %arg2) <{dim = 1 : i32}> : (tensor<4x8x6xf32>, tensor<4x3x6xui32>) -> tensor<4x3x6xf32>
    return %1 : tensor<4x3x6xf32>
  }

  // Three parallel chains sharing indices: 6 gathers become 4 after reassociation.
  func.func @gather_parallel_chains_shared_indices(
      %data0: tensor<32x8xbf16>,
      %data1: tensor<32x8xui8>,
      %data2: tensor<32x8xui8>,
      %indices_a: tensor<32x8xsi32>,
      %indices_b: tensor<4x8xsi32>) -> (tensor<4x8xbf16>, tensor<4x8xui8>, tensor<4x8xui8>) {
    // CHECK-LABEL: @gather_parallel_chains_shared_indices
    // CHECK-COUNT-4: "ttir.gather"
    // CHECK-NOT: "ttir.gather"
    %0 = "ttir.gather"(%data0, %indices_a) <{dim = 0 : i32}> : (tensor<32x8xbf16>, tensor<32x8xsi32>) -> tensor<32x8xbf16>
    %1 = "ttir.gather"(%data1, %indices_a) <{dim = 0 : i32}> : (tensor<32x8xui8>, tensor<32x8xsi32>) -> tensor<32x8xui8>
    %2 = "ttir.gather"(%data2, %indices_a) <{dim = 0 : i32}> : (tensor<32x8xui8>, tensor<32x8xsi32>) -> tensor<32x8xui8>
    %3 = "ttir.gather"(%0, %indices_b) <{dim = 0 : i32}> : (tensor<32x8xbf16>, tensor<4x8xsi32>) -> tensor<4x8xbf16>
    %4 = "ttir.gather"(%1, %indices_b) <{dim = 0 : i32}> : (tensor<32x8xui8>, tensor<4x8xsi32>) -> tensor<4x8xui8>
    %5 = "ttir.gather"(%2, %indices_b) <{dim = 0 : i32}> : (tensor<32x8xui8>, tensor<4x8xsi32>) -> tensor<4x8xui8>
    return %3, %4, %5 : tensor<4x8xbf16>, tensor<4x8xui8>, tensor<4x8xui8>
  }
}
