// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2 mesh-topology=ring,linear" -o %t %s
// RUN: FileCheck %s --input-file=%t
// Tests for ttnn-set-ccl-topology pass

// -----

// Verify that all_gather on cluster_axis=1 gets linear topology (axis 1 is linear)
module attributes {} {
  // CHECK-LABEL: all_gather_linear_topology
  func.func @all_gather_linear_topology(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttir.all_gather"(%arg0) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16>
    // CHECK: "ttnn.all_gather"
    // CHECK-SAME: topology = #ttcore.topology<linear>
    return %0 : tensor<1x1x32x64xbf16>
  }
}

// -----

// Verify that reduce_scatter on cluster_axis=1 gets linear topology
module attributes {} {
  // CHECK-LABEL: reduce_scatter_linear_topology
  func.func @reduce_scatter_linear_topology(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x128xf32> {
    %0 = "ttir.reduce_scatter"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x128xf32>
    // CHECK: "ttnn.reduce_scatter"
    // CHECK-SAME: topology = #ttcore.topology<linear>
    return %0 : tensor<1x1x8192x128xf32>
  }
}

// -----

// Verify that all_reduce on cluster_axis=1 gets linear topology on its constituent ops
module attributes {} {
  // CHECK-LABEL: all_reduce_linear_topology
  func.func @all_reduce_linear_topology(%arg0: tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    // CHECK: "ttnn.reduce_scatter"
    // CHECK-SAME: topology = #ttcore.topology<linear>
    // CHECK: "ttnn.all_gather"
    // CHECK-SAME: topology = #ttcore.topology<linear>
    return %0 : tensor<1x1x4096x16384xf32>
  }
}
