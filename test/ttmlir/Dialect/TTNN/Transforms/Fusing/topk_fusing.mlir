// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-fusing="enable-op-constraints=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#layout_2x6_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_2x3_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_2x6_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_2x3_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_4x8_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_4x5_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_4x8_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_4x5_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_10x4_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_7x4_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_10x4_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_7x4_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_3x5_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_3x2_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_3x5_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_3x2_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_2x3x8_f32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 3 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_2x3x4_f32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 3 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_2x3x8_si32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 3 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_2x3x4_si32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 3 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_2x8_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_2x8_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_4x6_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_2x6_f32_2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_4x6_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#layout_2x6_si32_2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

// Test fusing sort + slice + slice into topk operation (descending sort)
module {
    func.func @sort_slice_to_topk_descending(%arg0: tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 3 : i32, largest = true, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x6xf32, #layout_2x6_f32>, tensor<2x6xsi32, #layout_2x6_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32, #layout_2x6_f32>) -> tensor<2x3xf32, #layout_2x3_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32, #layout_2x6_si32>) -> tensor<2x3xsi32, #layout_2x3_si32>
        return %0, %1 : tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>
    }
}

// Test fusing sort + slice + slice into topk operation (ascending sort)
module {
    func.func @sort_slice_to_topk_ascending(%arg0: tensor<4x8xf32, #layout_4x8_f32>) -> (tensor<4x5xf32, #layout_4x5_f32>, tensor<4x5xsi32, #layout_4x5_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 5 : i32, largest = false, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = false, dim = 1 : si8, stable = false}> : (tensor<4x8xf32, #layout_4x8_f32>) -> (tensor<4x8xf32, #layout_4x8_f32>, tensor<4x8xsi32, #layout_4x8_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [4 : i32, 5 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x8xf32, #layout_4x8_f32>) -> tensor<4x5xf32, #layout_4x5_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [4 : i32, 5 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x8xsi32, #layout_4x8_si32>) -> tensor<4x5xsi32, #layout_4x5_si32>
        return %0, %1 : tensor<4x5xf32, #layout_4x5_f32>, tensor<4x5xsi32, #layout_4x5_si32>
    }
}

// Test fusing with different dimension (dim=0)
module {
    func.func @sort_slice_to_topk_dim0(%arg0: tensor<10x4xf32, #layout_10x4_f32>) -> (tensor<7x4xf32, #layout_7x4_f32>, tensor<7x4xsi32, #layout_7x4_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 0 : i32, k = 7 : i32, largest = true, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 0 : si8, stable = false}> : (tensor<10x4xf32, #layout_10x4_f32>) -> (tensor<10x4xf32, #layout_10x4_f32>, tensor<10x4xsi32, #layout_10x4_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [7 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<10x4xf32, #layout_10x4_f32>) -> tensor<7x4xf32, #layout_7x4_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [7 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<10x4xsi32, #layout_10x4_si32>) -> tensor<7x4xsi32, #layout_7x4_si32>
        return %0, %1 : tensor<7x4xf32, #layout_7x4_f32>, tensor<7x4xsi32, #layout_7x4_si32>
    }
}

// Test with negative dimension (should be handled correctly)
module {
    func.func @sort_slice_to_topk_negative_dim(%arg0: tensor<3x5xf32, #layout_3x5_f32>) -> (tensor<3x2xf32, #layout_3x2_f32>, tensor<3x2xsi32, #layout_3x2_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 2 : i32, largest = true, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = -1 : si8, stable = false}> : (tensor<3x5xf32, #layout_3x5_f32>) -> (tensor<3x5xf32, #layout_3x5_f32>, tensor<3x5xsi32, #layout_3x5_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [3 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<3x5xf32, #layout_3x5_f32>) -> tensor<3x2xf32, #layout_3x2_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [3 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<3x5xsi32, #layout_3x5_si32>) -> tensor<3x2xsi32, #layout_3x2_si32>
        return %0, %1 : tensor<3x2xf32, #layout_3x2_f32>, tensor<3x2xsi32, #layout_3x2_si32>
    }
}

// Test with 3D tensor
module {
    func.func @sort_slice_to_topk_3d(%arg0: tensor<2x3x8xf32, #layout_2x3x8_f32>) -> (tensor<2x3x4xf32, #layout_2x3x4_f32>, tensor<2x3x4xsi32, #layout_2x3x4_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 2 : i32, k = 4 : i32, largest = false, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = false, dim = 2 : si8, stable = false}> : (tensor<2x3x8xf32, #layout_2x3x8_f32>) -> (tensor<2x3x8xf32, #layout_2x3x8_f32>, tensor<2x3x8xsi32, #layout_2x3x8_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x3x8xf32, #layout_2x3x8_f32>) -> tensor<2x3x4xf32, #layout_2x3x4_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x3x8xsi32, #layout_2x3x8_si32>) -> tensor<2x3x4xsi32, #layout_2x3x4_si32>
        return %0, %1 : tensor<2x3x4xf32, #layout_2x3x4_f32>, tensor<2x3x4xsi32, #layout_2x3x4_si32>
    }
}

// Test fusing sort + slice from end into topk (descending sort, slice from end -> largest=false)
module {
    func.func @sort_slice_to_topk_from_end_descending(%arg0: tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 3 : i32, largest = false, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x6xf32, #layout_2x6_f32>, tensor<2x6xsi32, #layout_2x6_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 3 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32, #layout_2x6_f32>) -> tensor<2x3xf32, #layout_2x3_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 3 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32, #layout_2x6_si32>) -> tensor<2x3xsi32, #layout_2x3_si32>
        return %0, %1 : tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>
    }
}

// Test fusing sort + slice from end into topk (ascending sort, slice from end -> largest=true)
module {
    func.func @sort_slice_to_topk_from_end_ascending(%arg0: tensor<4x8xf32, #layout_4x8_f32>) -> (tensor<4x5xf32, #layout_4x5_f32>, tensor<4x5xsi32, #layout_4x5_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 5 : i32, largest = true, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = false, dim = 1 : si8, stable = false}> : (tensor<4x8xf32, #layout_4x8_f32>) -> (tensor<4x8xf32, #layout_4x8_f32>, tensor<4x8xsi32, #layout_4x8_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 3 : i32], ends = [4 : i32, 8 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x8xf32, #layout_4x8_f32>) -> tensor<4x5xf32, #layout_4x5_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 3 : i32], ends = [4 : i32, 8 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x8xsi32, #layout_4x8_si32>) -> tensor<4x5xsi32, #layout_4x5_si32>
        return %0, %1 : tensor<4x5xf32, #layout_4x5_f32>, tensor<4x5xsi32, #layout_4x5_si32>
    }
}

// Test fusing sort + slice into topk when only indices are used (values unused)
module {
    func.func @sort_slice_to_topk_only_indices(%arg0: tensor<2x6xf32, #layout_2x6_f32>) -> tensor<2x3xsi32, #layout_2x3_si32> {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 3 : i32, largest = true, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x6xf32, #layout_2x6_f32>, tensor<2x6xsi32, #layout_2x6_si32>)
        %0 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32, #layout_2x6_si32>) -> tensor<2x3xsi32, #layout_2x3_si32>
        return %0 : tensor<2x3xsi32, #layout_2x3_si32>
    }
}

// Test fusing sort + slice into topk when only values are used (indices unused)
module {
    func.func @sort_slice_to_topk_only_values(%arg0: tensor<2x6xf32, #layout_2x6_f32>) -> tensor<2x3xf32, #layout_2x3_f32> {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 3 : i32, largest = true, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x6xf32, #layout_2x6_f32>, tensor<2x6xsi32, #layout_2x6_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32, #layout_2x6_f32>) -> tensor<2x3xf32, #layout_2x3_f32>
        return %0 : tensor<2x3xf32, #layout_2x3_f32>
    }
}

// Negative test: slices don't start at 0 and don't end at dim size (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_nonzero_begin(%arg0: tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x6xf32, #layout_2x6_f32>, tensor<2x6xsi32, #layout_2x6_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 1 : i32], ends = [2 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32, #layout_2x6_f32>) -> tensor<2x3xf32, #layout_2x3_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 1 : i32], ends = [2 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32, #layout_2x6_si32>) -> tensor<2x3xsi32, #layout_2x3_si32>
        return %0, %1 : tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>
    }
}

// Negative test: slices have different parameters (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_different_params(%arg0: tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x3xf32, #layout_2x3_f32>, tensor<2x2xsi32, #layout_2x3_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x6xf32, #layout_2x6_f32>, tensor<2x6xsi32, #layout_2x6_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32, #layout_2x6_f32>) -> tensor<2x3xf32, #layout_2x3_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32, #layout_2x6_si32>) -> tensor<2x2xsi32, #layout_2x3_si32>
        return %0, %1 : tensor<2x3xf32, #layout_2x3_f32>, tensor<2x2xsi32, #layout_2x3_si32>
    }
}

// Negative test: slices have non-unit step (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_nonunit_step(%arg0: tensor<2x8xf32, #layout_2x8_f32>) -> (tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x8xf32, #layout_2x8_f32>) -> (tensor<2x8xf32, #layout_2x8_f32>, tensor<2x8xsi32, #layout_2x8_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 2 : i32]}> : (tensor<2x8xf32, #layout_2x8_f32>) -> tensor<2x3xf32, #layout_2x3_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 2 : i32]}> : (tensor<2x8xsi32, #layout_2x8_si32>) -> tensor<2x3xsi32, #layout_2x3_si32>
        return %0, %1 : tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>
    }
}

// Negative test: sort has multiple users (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_multiple_users(%arg0: tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x6xf32, #layout_2x6_f32>, tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[VALUES]], %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32, #layout_2x6_f32>) -> (tensor<2x6xf32, #layout_2x6_f32>, tensor<2x6xsi32, #layout_2x6_si32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32, #layout_2x6_f32>) -> tensor<2x3xf32, #layout_2x3_f32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32, #layout_2x6_si32>) -> tensor<2x3xsi32, #layout_2x3_si32>
        return %values, %0, %1 : tensor<2x6xf32, #layout_2x6_f32>, tensor<2x3xf32, #layout_2x3_f32>, tensor<2x3xsi32, #layout_2x3_si32>
    }
}

// Negative test: slicing on different dimension than sorted (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_wrong_dim(%arg0: tensor<4x6xf32, #layout_4x6_f32>) -> (tensor<2x6xf32, #layout_2x6_f32_2>, tensor<2x6xsi32, #layout_2x6_si32_2>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<4x6xf32, #layout_4x6_f32>) -> (tensor<4x6xf32, #layout_4x6_f32>, tensor<4x6xsi32, #layout_4x6_si32>)
        // Slicing on dim 0 instead of dim 1
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x6xf32, #layout_4x6_f32>) -> tensor<2x6xf32, #layout_2x6_f32_2>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x6xsi32, #layout_4x6_si32>) -> tensor<2x6xsi32, #layout_2x6_si32_2>
        return %0, %1 : tensor<2x6xf32, #layout_2x6_f32_2>, tensor<2x6xsi32, #layout_2x6_si32_2>
    }
}
