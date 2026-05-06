// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that an embedding whose index tensor contains any unit dimension(s)
// is rewritten to: reshape(squeeze units) + embedding + reshape(restore shape).

module {
  // [A, 1] – trailing unit dimension.
  func.func @embedding_ax1(%input: tensor<4x1xi32>, %weight: tensor<10x8xf32>) -> tensor<4x1x8xf32> {
    // CHECK-LABEL: func.func @embedding_ax1
    // CHECK:       "ttir.reshape"
    // CHECK-SAME:  tensor<4x1xi32>) -> tensor<4xi32>
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<4xi32>, tensor<10x8xf32>) -> tensor<4x8xf32>
    // CHECK:       "ttir.reshape"
    // CHECK-SAME:  tensor<4x8xf32>) -> tensor<4x1x8xf32>
    %0 = "ttir.embedding"(%input, %weight) : (tensor<4x1xi32>, tensor<10x8xf32>) -> tensor<4x1x8xf32>
    return %0 : tensor<4x1x8xf32>
  }

  // [1, A] – leading unit dimension.
  func.func @embedding_1xa(%input: tensor<1x4xi32>, %weight: tensor<10x8xf32>) -> tensor<1x4x8xf32> {
    // CHECK-LABEL: func.func @embedding_1xa
    // CHECK:       "ttir.reshape"
    // CHECK-SAME:  tensor<1x4xi32>) -> tensor<4xi32>
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<4xi32>, tensor<10x8xf32>) -> tensor<4x8xf32>
    // CHECK:       "ttir.reshape"
    // CHECK-SAME:  tensor<4x8xf32>) -> tensor<1x4x8xf32>
    %0 = "ttir.embedding"(%input, %weight) : (tensor<1x4xi32>, tensor<10x8xf32>) -> tensor<1x4x8xf32>
    return %0 : tensor<1x4x8xf32>
  }

  // Large realistic dimensions – same pattern applies.
  func.func @embedding_ax1_large(%input: tensor<1836732x1xi64>, %weight: tensor<1993728x80xf32>) -> tensor<1836732x1x80xf32> {
    // CHECK-LABEL: func.func @embedding_ax1_large
    // CHECK:       "ttir.reshape"
    // CHECK-SAME:  tensor<1836732x1xi64>) -> tensor<1836732xi64>
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<1836732xi64>, tensor<1993728x80xf32>) -> tensor<1836732x80xf32>
    // CHECK:       "ttir.reshape"
    // CHECK-SAME:  tensor<1836732x80xf32>) -> tensor<1836732x1x80xf32>
    %0 = "ttir.embedding"(%input, %weight) : (tensor<1836732x1xi64>, tensor<1993728x80xf32>) -> tensor<1836732x1x80xf32>
    return %0 : tensor<1836732x1x80xf32>
  }

  // 1-D index tensor with no unit dims – pattern must NOT fire.
  func.func @embedding_1d_no_canonicalize(%input: tensor<4xi32>, %weight: tensor<10x8xf32>) -> tensor<4x8xf32> {
    // CHECK-LABEL: func.func @embedding_1d_no_canonicalize
    // CHECK-NOT:   "ttir.reshape"
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<4xi32>, tensor<10x8xf32>) -> tensor<4x8xf32>
    %0 = "ttir.embedding"(%input, %weight) : (tensor<4xi32>, tensor<10x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  // 2-D index tensor with no unit dims – pattern must NOT fire.
  func.func @embedding_ax3_no_canonicalize(%input: tensor<4x3xi32>, %weight: tensor<10x8xf32>) -> tensor<4x3x8xf32> {
    // CHECK-LABEL: func.func @embedding_ax3_no_canonicalize
    // CHECK-NOT:   "ttir.reshape"
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<4x3xi32>, tensor<10x8xf32>) -> tensor<4x3x8xf32>
    %0 = "ttir.embedding"(%input, %weight) : (tensor<4x3xi32>, tensor<10x8xf32>) -> tensor<4x3x8xf32>
    return %0 : tensor<4x3x8xf32>
  }

  // All-ones input [1, 1] – would produce scalar index, pattern must NOT fire.
  func.func @embedding_1x1_no_canonicalize(%input: tensor<1x1xi32>, %weight: tensor<10x8xf32>) -> tensor<1x1x8xf32> {
    // CHECK-LABEL: func.func @embedding_1x1_no_canonicalize
    // CHECK-NOT:   "ttir.reshape"
    // CHECK:       "ttir.embedding"
    // CHECK-SAME:  (tensor<1x1xi32>, tensor<10x8xf32>) -> tensor<1x1x8xf32>
    %0 = "ttir.embedding"(%input, %weight) : (tensor<1x1xi32>, tensor<10x8xf32>) -> tensor<1x1x8xf32>
    return %0 : tensor<1x1x8xf32>
  }
}
