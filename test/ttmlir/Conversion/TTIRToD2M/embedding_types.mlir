// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --canonicalize %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @embedding_bf16_table_ui32_indices
  func.func @embedding_bf16_table_ui32_indices(
      %indices: tensor<2x3xui32>,
      %weight: tensor<16x8xbf16>) -> tensor<2x3x8xbf16> {
    // CHECK: d2m.generic
    // CHECK-SAME: indexing_maps = [
    // CHECK-SAME: iterator_types = [#parallel, #parallel]
    // CHECK-SAME: threads = [#d2m.thread<datamovement>]
    // CHECK: d2m.embedding
    // CHECK-SAME: <6, 8>
    // CHECK-SAME: {indicesShape = array<i64: 2, 3>}
    // CHECK-SAME: tensor<{{.*}}xui32
    // CHECK-SAME: tensor<{{.*}}xbf16
    %0 = "ttir.embedding"(%indices, %weight) : (tensor<2x3xui32>, tensor<16x8xbf16>) -> tensor<2x3x8xbf16>
    return %0 : tensor<2x3x8xbf16>
  }

  // CHECK-LABEL: func.func @embedding_i32_table_ui32_indices
  func.func @embedding_i32_table_ui32_indices(
      %indices: tensor<3x1xui32>,
      %weight: tensor<16x1xi32>) -> tensor<3x1x1xi32> {
    // CHECK: d2m.generic
    // CHECK-SAME: indexing_maps = [
    // CHECK-SAME: iterator_types = [#parallel, #parallel]
    // CHECK-SAME: threads = [#d2m.thread<datamovement>]
    // CHECK: d2m.embedding
    // CHECK-SAME: <3, 1>
    // CHECK-SAME: {indicesShape = array<i64: 3, 1>}
    // CHECK-SAME: tensor<{{.*}}xui32
    // CHECK-SAME: tensor<{{.*}}xi32
    %0 = "ttir.embedding"(%indices, %weight) : (tensor<3x1xui32>, tensor<16x1xi32>) -> tensor<3x1x1xi32>
    return %0 : tensor<3x1x1xi32>
  }
}
