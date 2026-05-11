// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that an embedding whose index tensor contains any unit dimension(s)
// is rewritten to: reshape(squeeze units) + embedding + reshape(restore shape).

module {
  // [A, 1] – trailing unit dimension.
  func.func @embedding_ax1(%input: tensor<4xi32>, %weight: tensor<10x8xf32>) -> tensor<4x8xf32> {
    // CHECK-LABEL: func.func @embedding_ax1
    // CHECK-NOT:   "ttir.reshape"
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<4xi32>, tensor<10x8xf32>) -> tensor<4x8xf32>
    %0 = "ttir.reshape"(%input) <{shape = [4 : i32, 1 : i32]}> : (tensor<4xi32>) -> tensor<4x1xi32>
    %1 = "ttir.embedding"(%0, %weight) : (tensor<4x1xi32>, tensor<10x8xf32>) -> tensor<4x1x8xf32>
    %2 = "ttir.reshape"(%1) <{shape = [4 : i32, 8 : i32]}> : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
    return %2 : tensor<4x8xf32>
  }

  // [1, A] – leading unit dimension.
  func.func @embedding_1xa(%input: tensor<4xi32>, %weight: tensor<10x8xf32>) -> tensor<4x8xf32> {
    // CHECK-LABEL: func.func @embedding_1xa
    // CHECK-NOT:   "ttir.reshape"
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<4xi32>, tensor<10x8xf32>) -> tensor<4x8xf32>
    %0 = "ttir.reshape"(%input) <{shape = [1 : i32, 4 : i32]}> : (tensor<4xi32>) -> tensor<1x4xi32>
    %1 = "ttir.embedding"(%0, %weight) : (tensor<1x4xi32>, tensor<10x8xf32>) -> tensor<1x4x8xf32>
    %2 = "ttir.reshape"(%1) <{shape = [4 : i32, 8 : i32]}> : (tensor<1x4x8xf32>) -> tensor<4x8xf32>
    return %2 : tensor<4x8xf32>
  }

  // Large dimensions from customer model.
  func.func @embedding_ax1_large(%input: tensor<1836732xi64>, %weight: tensor<1993728x80xf32>) -> tensor<1836732x80xf32> {
    // CHECK-LABEL: func.func @embedding_ax1_large
    // CHECK-NOT:   "ttir.reshape"
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<1836732xi64>, tensor<1993728x80xf32>) -> tensor<1836732x80xf32>
    %0 = "ttir.reshape"(%input) <{shape = [1836732 : i32, 1 : i32]}> : (tensor<1836732xi64>) -> tensor<1836732x1xi64>
    %1 = "ttir.embedding"(%0, %weight) : (tensor<1836732x1xi64>, tensor<1993728x80xf32>) -> tensor<1836732x1x80xf32>
    %2 = "ttir.reshape"(%1) <{shape = [1836732 : i32, 80 : i32]}> : (tensor<1836732x1x80xf32>) -> tensor<1836732x80xf32>
    return %2 : tensor<1836732x80xf32>
  }


  // 1-D index tensor with no unit dims – pattern must NOT fire.
  func.func @embedding_1d_no_canonicalize(%input: tensor<2x2xi32>, %weight: tensor<10x8xf32>) -> tensor<2x2x8xf32> {
    // CHECK-LABEL: func.func @embedding_1d_no_canonicalize
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<4xi32>, tensor<10x8xf32>) -> tensor<4x8xf32>
    %0 = "ttir.reshape"(%input) <{shape = [4 : i32]}> : (tensor<2x2xi32>) -> tensor<4xi32>
    %1 = "ttir.embedding"(%0, %weight) : (tensor<4xi32>, tensor<10x8xf32>) -> tensor<4x8xf32>
    %2 = "ttir.reshape"(%1) <{shape = [2 : i32, 2 : i32, 8 : i32]}> : (tensor<4x8xf32>) -> tensor<2x2x8xf32>
    return %2 : tensor<2x2x8xf32>
  }

  // Indices do not come from reshape, output does not go to reshape, pattern must NOT fire.
  func.func @embedding_1d_no_canonicalize_reshapes(%input: tensor<4x1xi32>, %weight: tensor<10x8xf32>) -> tensor<4x1x8xf32> {
    // CHECK-LABEL: func.func @embedding_1d_no_canonicalize_reshapes
    // CHECK-NOT:   "ttir.reshape"
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<4x1xi32>, tensor<10x8xf32>) -> tensor<4x1x8xf32>
    %0 = "ttir.embedding"(%input, %weight) : (tensor<4x1xi32>, tensor<10x8xf32>) -> tensor<4x1x8xf32>
    return %0 : tensor<4x1x8xf32>
  }

  // All-ones input [1, 1] – would produce scalar index.
  func.func @embedding_1x1_canonicalize(%input: tensor<1xi32>, %weight: tensor<10x8xf32>) -> tensor<1x8xf32> {
    // CHECK-LABEL: func.func @embedding_1x1_canonicalize
    // CHECK-NOT:       "ttir.reshape"
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<1xi32>, tensor<10x8xf32>) -> tensor<1x8xf32>
    %0 = "ttir.reshape"(%input) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>) -> tensor<1x1xi32>
    %1 = "ttir.embedding"(%0, %weight) : (tensor<1x1xi32>, tensor<10x8xf32>) -> tensor<1x1x8xf32>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 8 : i32]}> : (tensor<1x1x8xf32>) -> tensor<1x8xf32>
    return %2 : tensor<1x8xf32>
  }
}
