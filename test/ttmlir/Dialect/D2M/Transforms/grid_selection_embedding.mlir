// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --canonicalize %s | FileCheck %s --check-prefix=BEFORE
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize %s | FileCheck %s --check-prefix=AFTER
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize --ttcore-one-shot-bufferize %s | FileCheck %s --check-prefix=BUFFER

module {
  func.func @embedding_grid(%indices: tensor<8x4xi32>, %weight: tensor<16x32xf32>) -> tensor<8x4x32xf32> {
    // BEFORE-LABEL: func.func @embedding_grid
    // BEFORE: %[[GENERIC:.*]] = d2m.generic
    // BEFORE-SAME: grid = #ttcore.grid<1x1>
    // BEFORE: %[[EMBED:.*]] = d2m.embedding {{.*}}<32, 32>
    // BEFORE-SAME: {indicesShape = array<i64: 8, 4>}
    // BEFORE-SAME: -> tensor<1x1x256x32xf32
    // BEFORE: d2m.yield %[[EMBED]] : (tensor<1x1x256x32xf32

    // AFTER-LABEL: func.func @embedding_grid
    // AFTER: %[[GENERIC:.*]] = d2m.generic
    // AFTER-SAME: grid = #ttcore.grid<8x1>
    // AFTER: %[[EMBED:.*]] = d2m.embedding {{.*}}<32, 32>
    // AFTER-SAME: {indicesShape = array<i64: 8, 4>}
    // AFTER-SAME: -> tensor<8x1x32x32xf32
    // AFTER: d2m.yield %[[EMBED]] : (tensor<8x1x32x32xf32

    // BUFFER-LABEL: func.func @embedding_grid
    // BUFFER: d2m.generic
    // BUFFER-SAME: grid = #ttcore.grid<8x1>
    // BUFFER: d2m.indexed_row_copy {{.*}} scratch {{.*}}<32, 32>
    // BUFFER-SAME: {indicesShape = array<i64: 8, 4>}
    // BUFFER-SAME: : memref
    %0 = "ttir.embedding"(%indices, %weight) : (tensor<8x4xi32>, tensor<16x32xf32>) -> tensor<8x4x32xf32>
    return %0 : tensor<8x4x32xf32>
  }

  func.func @embedding_single_row(%indices: tensor<1x4xi32>, %weight: tensor<16x16xf32>) -> tensor<1x4x16xf32> {
    // BEFORE-LABEL: func.func @embedding_single_row
    // BEFORE: %[[EMBED:.*]] = d2m.embedding {{.*}}<4, 16>
    // BEFORE-SAME: {indicesShape = array<i64: 1, 4>}
    // BEFORE-SAME: -> tensor<
    // BEFORE: d2m.yield %[[EMBED]]

    // AFTER-LABEL: func.func @embedding_single_row
    // AFTER: %[[EMBED:.*]] = d2m.embedding {{.*}}<4, 16>
    // AFTER-SAME: {indicesShape = array<i64: 1, 4>}
    // AFTER-SAME: -> tensor<
    // AFTER: d2m.yield %[[EMBED]]

    // BUFFER-LABEL: func.func @embedding_single_row
    // BUFFER: d2m.indexed_row_copy {{.*}} scratch {{.*}}<4, 16>
    // BUFFER-SAME: {indicesShape = array<i64: 1, 4>}
    // BUFFER-SAME: : memref
    %0 = "ttir.embedding"(%indices, %weight) : (tensor<1x4xi32>, tensor<16x16xf32>) -> tensor<1x4x16xf32>
    return %0 : tensor<1x4x16xf32>
  }
}
