// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --canonicalize %s | FileCheck %s --check-prefixes=CHECK,L1
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="default-input-memspace=dram default-output-memspace=dram" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,DRAM

// L1-DAG: #[[L1_INDICES_BF16:.*]] = #ttcore.metal_layout<logical_shape = 2x3{{.*}}l1, sharded>
// L1-DAG: #[[L1_WEIGHT_BF16:.*]] = #ttcore.metal_layout<logical_shape = 16x8{{.*}}l1, sharded>
// L1-DAG: #[[L1_OUTPUT_BF16:.*]] = #ttcore.metal_layout<logical_shape = 2x3x8{{.*}}l1, sharded>
// L1-DAG: #[[L1_INDICES_I32:.*]] = #ttcore.metal_layout<logical_shape = 3x1{{.*}}l1, sharded>
// L1-DAG: #[[L1_WEIGHT_I32:.*]] = #ttcore.metal_layout<logical_shape = 16x1{{.*}}l1, sharded>
// L1-DAG: #[[L1_OUTPUT_I32:.*]] = #ttcore.metal_layout<logical_shape = 3x1x1{{.*}}l1, sharded>
// DRAM-DAG: #[[DRAM_WEIGHT_BF16:.*]] = #ttcore.metal_layout<logical_shape = 16x8{{.*}}dram, sharded>
// DRAM-DAG: #[[DRAM_L1_INDICES_BF16:.*]] = #ttcore.metal_layout<logical_shape = 2x3{{.*}}l1, sharded>
// DRAM-DAG: #[[DRAM_OUTPUT_BF16:.*]] = #ttcore.metal_layout<logical_shape = 2x3x8{{.*}}dram, sharded>
// DRAM-DAG: #[[DRAM_WEIGHT_I32:.*]] = #ttcore.metal_layout<logical_shape = 16x1{{.*}}dram, sharded>
// DRAM-DAG: #[[DRAM_L1_INDICES_I32:.*]] = #ttcore.metal_layout<logical_shape = 3x1{{.*}}l1, sharded>
// DRAM-DAG: #[[DRAM_OUTPUT_I32:.*]] = #ttcore.metal_layout<logical_shape = 3x1x1{{.*}}dram, sharded>

module {
  // CHECK-LABEL: func.func @embedding_bf16_table_ui32_indices
  func.func @embedding_bf16_table_ui32_indices(
      %indices: tensor<2x3xui32>,
      %weight: tensor<16x8xbf16>) -> tensor<2x3x8xbf16> {
    // L1-DAG: d2m.to_layout %arg0, %{{.*}} : tensor<2x3xui32> into tensor<{{.*}}xui32, #[[L1_INDICES_BF16]]>
    // L1-DAG: d2m.to_layout %arg1, %{{.*}} : tensor<16x8xbf16> into tensor<{{.*}}xbf16, #[[L1_WEIGHT_BF16]]>
    // DRAM-DAG: d2m.to_layout %arg1, %{{.*}} : tensor<16x8xbf16> into tensor<{{.*}}xbf16, #[[DRAM_WEIGHT_BF16]]>
    // DRAM-DAG: d2m.to_layout %arg0, %{{.*}} : tensor<2x3xui32> into tensor<{{.*}}xui32, #[[DRAM_L1_INDICES_BF16]]>
    // CHECK: d2m.generic
    // CHECK-SAME: indexing_maps = [
    // CHECK-SAME: iterator_types = [#parallel, #parallel]
    // CHECK-SAME: threads = [#d2m.thread<datamovement>]
    // CHECK: d2m.embedding
    // CHECK-SAME: <6, 8>
    // CHECK-SAME: {indicesShape = array<i64: 2, 3>}
    // L1-SAME: tensor<{{[^,]*}}xui32, #[[L1_INDICES_BF16]]
    // L1-SAME: tensor<{{[^,]*}}xbf16, #[[L1_WEIGHT_BF16]]
    // L1-SAME: tensor<{{[^,]*}}xbf16, #[[L1_OUTPUT_BF16]]
    // DRAM-SAME: tensor<{{[^,]*}}xui32, #[[DRAM_L1_INDICES_BF16]]
    // DRAM-SAME: tensor<{{[^,]*}}xbf16, #[[DRAM_WEIGHT_BF16]]
    // DRAM-SAME: tensor<{{[^,]*}}xbf16, #[[DRAM_OUTPUT_BF16]]
    %0 = "ttir.embedding"(%indices, %weight) : (tensor<2x3xui32>, tensor<16x8xbf16>) -> tensor<2x3x8xbf16>
    return %0 : tensor<2x3x8xbf16>
  }

  // CHECK-LABEL: func.func @embedding_i32_table_ui32_indices
  func.func @embedding_i32_table_ui32_indices(
      %indices: tensor<3x1xui32>,
      %weight: tensor<16x1xi32>) -> tensor<3x1x1xi32> {
    // L1-DAG: d2m.to_layout %arg0, %{{.*}} : tensor<3x1xui32> into tensor<{{.*}}xui32, #[[L1_INDICES_I32]]>
    // L1-DAG: d2m.to_layout %arg1, %{{.*}} : tensor<16x1xi32> into tensor<{{.*}}xi32, #[[L1_WEIGHT_I32]]>
    // DRAM-DAG: d2m.to_layout %arg1, %{{.*}} : tensor<16x1xi32> into tensor<{{.*}}xi32, #[[DRAM_WEIGHT_I32]]>
    // DRAM-DAG: d2m.to_layout %arg0, %{{.*}} : tensor<3x1xui32> into tensor<{{.*}}xui32, #[[DRAM_L1_INDICES_I32]]>
    // CHECK: d2m.generic
    // CHECK-SAME: indexing_maps = [
    // CHECK-SAME: iterator_types = [#parallel, #parallel]
    // CHECK-SAME: threads = [#d2m.thread<datamovement>]
    // CHECK: d2m.embedding
    // CHECK-SAME: <3, 1>
    // CHECK-SAME: {indicesShape = array<i64: 3, 1>}
    // L1-SAME: tensor<{{[^,]*}}xui32, #[[L1_INDICES_I32]]
    // L1-SAME: tensor<{{[^,]*}}xi32, #[[L1_WEIGHT_I32]]
    // L1-SAME: tensor<{{[^,]*}}xi32, #[[L1_OUTPUT_I32]]
    // DRAM-SAME: tensor<{{[^,]*}}xui32, #[[DRAM_L1_INDICES_I32]]
    // DRAM-SAME: tensor<{{[^,]*}}xi32, #[[DRAM_WEIGHT_I32]]
    // DRAM-SAME: tensor<{{[^,]*}}xi32, #[[DRAM_OUTPUT_I32]]
    %0 = "ttir.embedding"(%indices, %weight) : (tensor<3x1xui32>, tensor<16x1xi32>) -> tensor<3x1x1xi32>
    return %0 : tensor<3x1x1xi32>
  }
}
