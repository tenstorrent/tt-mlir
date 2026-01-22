// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module {
  // Test cumsum along dimension 0 (size 3 -> 3 iterations unrolled)
  func.func @cumsum_dim0(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
    // CHECK-LABEL: func.func @cumsum_dim0
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK-NOT: scf.for
    %0 = "ttir.cumsum"(%arg0) <{dim = 0 : i64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }

  // Test cumsum along dimension 1 (size 4 -> 4 iterations unrolled)
  func.func @cumsum_dim1(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
    // CHECK-LABEL: func.func @cumsum_dim1
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK-NOT: scf.for
    %0 = "ttir.cumsum"(%arg0) <{dim = 1 : i64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }

  // Test cumsum with negative dimension (dim=-1 -> dim=2, size 4)
  func.func @cumsum_negative_dim(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    // CHECK-LABEL: func.func @cumsum_negative_dim
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK-NOT: scf.for
    %0 = "ttir.cumsum"(%arg0) <{dim = -1 : i64}> : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }

  // Test cumsum with 3D tensor along middle dimension (size 5 -> 5 iterations)
  func.func @cumsum_3d_middle(%arg0: tensor<2x5x3xf32>) -> tensor<2x5x3xf32> {
    // CHECK-LABEL: func.func @cumsum_3d_middle
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK-NOT: scf.for
    %0 = "ttir.cumsum"(%arg0) <{dim = 1 : i64}> : (tensor<2x5x3xf32>) -> tensor<2x5x3xf32>
    return %0 : tensor<2x5x3xf32>
  }
}
