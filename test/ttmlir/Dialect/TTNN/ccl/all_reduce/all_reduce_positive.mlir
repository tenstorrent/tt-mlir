// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for ttnn all_reduce op

// -----

// Verify lowering of ttir all_reduce to ttnn ops
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_with_reshapes
  func.func @all_reduce_positive_with_reshapes(%arg0: tensor<4096x16384xf32>) -> tensor<4096x16384xf32> {
    %0 = ttir.empty() : tensor<4096x16384xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{all_gather_dim = 0 : si32, cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<4096x16384xf32>, tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    // CHECK: = "ttnn.reshape"
    // CHECK: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    return %1 : tensor<4096x16384xf32>
  }
}

// -----

// Verify lowering of ttir all_reduce to ttnn ops
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_without_reshapes
  func.func @all_reduce_positive_without_reshapes(%arg0: tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32> {
    %0 = ttir.empty() : tensor<1x1x4096x16384xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{all_gather_dim = 0 : si32, cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<1x1x4096x16384xf32>, tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    // CHECK-NOT: = "ttnn.reshape"
    // CHECK: "ttnn.reduce_scatter"
    // CHECK-NOT: = "ttnn.reshape"
    // CHECK: "ttnn.all_gather"
    return %1 : tensor<1x1x4096x16384xf32>
  }
}

// -----

// Verify op folding for single mesh device communication
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_with_reshapes_folding
  func.func @all_reduce_positive_with_reshapes_folding(%arg0: tensor<4096x16384xf32>) -> tensor<4096x16384xf32> {
    %0 = ttir.empty() : tensor<4096x16384xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<4096x16384xf32>, tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    // CHECK-NOT: = "ttnn.reshape"
    // CHECK-NOT: "ttnn.reduce_scatter"
    // CHECK-NOT: = "ttnn.reshape"
    // CHECK-NOT: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.all_reduce"
    return %1 : tensor<4096x16384xf32>
  }
}

// -----

// Verify op folding for single mesh device communication
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_without_reshapes_folding
  func.func @all_reduce_positive_without_reshapes_folding(%arg0: tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32> {
    %0 = ttir.empty() : tensor<1x1x4096x16384xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<1x1x4096x16384xf32>, tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    // CHECK-NOT: = "ttnn.reshape"
    // CHECK-NOT: "ttnn.reduce_scatter"
    // CHECK-NOT: = "ttnn.reshape"
    // CHECK-NOT: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.all_reduce"
    return %1 : tensor<1x1x4096x16384xf32>
  }
}

// -----

// Verify transform for all_reduce with non-divisible dimension
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_with_non_divisible_dimensions
  func.func @all_reduce_positive_with_non_divisible_dimensions(%arg0: tensor<1x1x31x31xf32>) -> tensor<1x1x31x31xf32> {
    %0 = ttir.empty() : tensor<1x1x31x31xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x31x31xf32>, tensor<1x1x31x31xf32>) -> tensor<1x1x31x31xf32>
    // CHECK-NOT: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.sum"
    return %1 : tensor<1x1x31x31xf32>
  }
}


// -----

// Verify breakdown of all_reduce into all_gather and reduce with reduce_type Min
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_with_non_divisible_dimensions_min
  func.func @all_reduce_positive_with_non_divisible_dimensions_min(%arg0: tensor<1x1x31x31xf32>) -> tensor<1x1x31x31xf32> {
    %0 = ttir.empty() : tensor<1x1x31x31xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<min>}> : (tensor<1x1x31x31xf32>, tensor<1x1x31x31xf32>) -> tensor<1x1x31x31xf32>
    // CHECK-NOT: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.min"
    return %1 : tensor<1x1x31x31xf32>
  }
}

// -----

// Verify breakdown of all_reduce into all_gather and reduce with reduce_type Max
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_with_non_divisible_dimensions_max
  func.func @all_reduce_positive_with_non_divisible_dimensions_max(%arg0: tensor<1x1x31x31xf32>) -> tensor<1x1x31x31xf32> {
    %0 = ttir.empty() : tensor<1x1x31x31xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<max>}> : (tensor<1x1x31x31xf32>, tensor<1x1x31x31xf32>) -> tensor<1x1x31x31xf32>
    // CHECK-NOT: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.max"
    return %1 : tensor<1x1x31x31xf32>
  }
}


// -----

// Verify breakdown of all_reduce into all_gather and local reduce memory limit
// Reduce_Scatter + All_Gather breakdown need to be used for this case due to memory limitation
// 1x38x128x515xf32 tensor needs additional 646279680 bytes of memory when we use all_gather + local reduce breakdown
// It exceedes 5% of DRAM capacity limit : 644244480 bytes with default system descriptor

module attributes {} {
  // CHECK-LABEL: all_reduce_positive_with_non_divisible_dimensions_over_memory_limit
  func.func @all_reduce_positive_with_non_divisible_dimensions_over_memory_limit(%arg0: tensor<1x38x128x515xf32>) -> tensor<1x38x128x515xf32> {
    %0 = ttir.empty() : tensor<1x38x128x515xf32>
    %1 = "ttir.all_reduce"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x38x128x515xf32>, tensor<1x38x128x515xf32>) -> tensor<1x38x128x515xf32>
    // CHECK: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.sum"
    return %1 : tensor<1x38x128x515xf32>
  }
}
