// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-fusing="enable-op-constraints=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test fusing sort + slice + slice into topk operation (descending sort)
module {
    func.func @sort_slice_to_topk_descending(%arg0: tensor<2x6xf32>) -> (tensor<2x3xf32>, tensor<2x3xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 3 : i32, largest = true, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xsi32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32>) -> tensor<2x3xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32>) -> tensor<2x3xsi32>
        return %0, %1 : tensor<2x3xf32>, tensor<2x3xsi32>
    }
}

// Test fusing sort + slice + slice into topk operation (ascending sort)
module {
    func.func @sort_slice_to_topk_ascending(%arg0: tensor<4x8xf32>) -> (tensor<4x5xf32>, tensor<4x5xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 5 : i32, largest = false, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = false, dim = 1 : si8, stable = false}> : (tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xsi32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [4 : i32, 5 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x8xf32>) -> tensor<4x5xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [4 : i32, 5 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x8xsi32>) -> tensor<4x5xsi32>
        return %0, %1 : tensor<4x5xf32>, tensor<4x5xsi32>
    }
}

// Test fusing with different dimension (dim=0)
module {
    func.func @sort_slice_to_topk_dim0(%arg0: tensor<10x4xf32>) -> (tensor<7x4xf32>, tensor<7x4xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 0 : i32, k = 7 : i32, largest = true, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 0 : si8, stable = false}> : (tensor<10x4xf32>) -> (tensor<10x4xf32>, tensor<10x4xsi32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [7 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<10x4xf32>) -> tensor<7x4xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [7 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<10x4xsi32>) -> tensor<7x4xsi32>
        return %0, %1 : tensor<7x4xf32>, tensor<7x4xsi32>
    }
}

// Test with negative dimension (should be handled correctly)
module {
    func.func @sort_slice_to_topk_negative_dim(%arg0: tensor<3x5xf32>) -> (tensor<3x2xf32>, tensor<3x2xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 1 : i32, k = 2 : i32, largest = true, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = -1 : si8, stable = false}> : (tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xsi32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [3 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<3x5xf32>) -> tensor<3x2xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [3 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<3x5xsi32>) -> tensor<3x2xsi32>
        return %0, %1 : tensor<3x2xf32>, tensor<3x2xsi32>
    }
}

// Test with 3D tensor
module {
    func.func @sort_slice_to_topk_3d(%arg0: tensor<2x3x8xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x4xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%arg0)
        // CHECK-SAME: <{dim = 2 : i32, k = 4 : i32, largest = false, sorted = true}>
        // CHECK-NOT: ttnn.sort
        // CHECK-NOT: ttnn.slice_static
        // CHECK: return %[[VALUES]], %[[INDICES]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = false, dim = 2 : si8, stable = false}> : (tensor<2x3x8xf32>) -> (tensor<2x3x8xf32>, tensor<2x3x8xsi32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x3x8xf32>) -> tensor<2x3x4xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x3x8xsi32>) -> tensor<2x3x4xsi32>
        return %0, %1 : tensor<2x3x4xf32>, tensor<2x3x4xsi32>
    }
}

// Negative test: slices don't start at 0 (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_nonzero_begin(%arg0: tensor<2x6xf32>) -> (tensor<2x3xf32>, tensor<2x3xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xsi32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 1 : i32], ends = [2 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32>) -> tensor<2x3xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 1 : i32], ends = [2 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32>) -> tensor<2x3xsi32>
        return %0, %1 : tensor<2x3xf32>, tensor<2x3xsi32>
    }
}

// Negative test: slices have different parameters (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_different_params(%arg0: tensor<2x6xf32>) -> (tensor<2x3xf32>, tensor<2x2xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xsi32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32>) -> tensor<2x3xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32>) -> tensor<2x2xsi32>
        return %0, %1 : tensor<2x3xf32>, tensor<2x2xsi32>
    }
}

// Negative test: slices have non-unit step (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_nonunit_step(%arg0: tensor<2x8xf32>) -> (tensor<2x3xf32>, tensor<2x3xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x8xf32>) -> (tensor<2x8xf32>, tensor<2x8xsi32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 2 : i32]}> : (tensor<2x8xf32>) -> tensor<2x3xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 2 : i32]}> : (tensor<2x8xsi32>) -> tensor<2x3xsi32>
        return %0, %1 : tensor<2x3xf32>, tensor<2x3xsi32>
    }
}

// Negative test: sort has multiple users (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_multiple_users(%arg0: tensor<2x6xf32>) -> (tensor<2x6xf32>, tensor<2x3xf32>, tensor<2x3xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[VALUES]], %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<2x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xsi32>)
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32>) -> tensor<2x3xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xsi32>) -> tensor<2x3xsi32>
        return %values, %0, %1 : tensor<2x6xf32>, tensor<2x3xf32>, tensor<2x3xsi32>
    }
}

// Negative test: slicing on different dimension than sorted (should NOT be fused)
module {
    func.func @sort_slice_no_fusion_wrong_dim(%arg0: tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xsi32>) {
        // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"(%arg0)
        // CHECK: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[VALUES]])
        // CHECK: %[[SLICE2:.*]] = "ttnn.slice_static"(%[[INDICES]])
        // CHECK-NOT: ttnn.topk
        // CHECK: return %[[SLICE1]], %[[SLICE2]]
        %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<4x6xf32>) -> (tensor<4x6xf32>, tensor<4x6xsi32>)
        // Slicing on dim 0 instead of dim 1
        %0 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x6xf32>) -> tensor<2x6xf32>
        %1 = "ttnn.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 6 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x6xsi32>) -> tensor<2x6xsi32>
        return %0, %1 : tensor<2x6xf32>, tensor<2x6xsi32>
    }
}
