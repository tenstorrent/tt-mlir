// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,4" -o %t %s
// RUN: FileCheck %s --input-file=%t

// The ttnn reduce_scatter kernel HANGS on cluster axes wider than 2 chips
// (observed on a 1x4 Blackhole mesh running Qwen3-32B vLLM TP, where an
// all_reduce(sum) on tensor<32x25600> was lowered to reduce_scatter +
// all_gather and the reduce_scatter deadlocked). For >2-chip axes we instead
// lower all_reduce to all_gather + a local reduce, which only relies on the
// working all_gather collective.

// -----

// The exact shape from the hanging Qwen3-32B up/gate-projection TP reduction:
// 25600 is divisible by 4, so the legacy path would have picked
// reduce_scatter; the 4-chip axis must now use all_gather + local sum.
module attributes {} {
  // CHECK-LABEL: all_reduce_sum_4chip_avoids_reduce_scatter
  func.func @all_reduce_sum_4chip_avoids_reduce_scatter(%arg0: tensor<32x25600xf32>) -> tensor<32x25600xf32> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x25600xf32>) -> tensor<32x25600xf32>
    // CHECK-NOT: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.sum"
    return %0 : tensor<32x25600xf32>
  }
}

// -----

// A divisible 4D shape that the legacy path would have routed to
// reduce_scatter (last dim divisible by 4) now uses all_gather + local sum.
module attributes {} {
  // CHECK-LABEL: all_reduce_sum_4chip_divisible_4d
  func.func @all_reduce_sum_4chip_divisible_4d(%arg0: tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    // CHECK-NOT: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.sum"
    return %0 : tensor<1x1x4096x16384xf32>
  }
}

// -----

// cluster_axis=0 here spans a single device (mesh row count is 1), so the all
// reduce folds away entirely (no collective, no reduce). This guards that the
// >2-chip gate does not fire on degenerate axes.
module attributes {} {
  // CHECK-LABEL: all_reduce_sum_single_device_axis_folds
  func.func @all_reduce_sum_single_device_axis_folds(%arg0: tensor<32x25600xf32>) -> tensor<32x25600xf32> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x25600xf32>) -> tensor<32x25600xf32>
    // CHECK-NOT: "ttnn.reduce_scatter"
    // CHECK-NOT: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.all_reduce"
    return %0 : tensor<32x25600xf32>
  }
}
