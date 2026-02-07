// RUN: ttmlir-opt --ttir-all-reduce-decomposition %s | FileCheck %s

// Test: Two all_reduce ops with the same cluster_axis flowing into a multiply
// should be decomposed to reduce_scatter + all_gather, and the all_gathers
// should be fused.

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
      func.func @test_decompose_and_fuse(%arg0: tensor<4096x4608xbf16>, %arg1: tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16> {
        // CHECK-LABEL: func.func @test_decompose_and_fuse
        // CHECK: %[[RS0:.*]] = "ttir.reduce_scatter"(%arg0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}>
        // CHECK: %[[RS1:.*]] = "ttir.reduce_scatter"(%arg1) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}>
        // CHECK: %[[MUL:.*]] = "ttir.multiply"(%[[RS0]], %[[RS1]])
        // CHECK: %[[AG:.*]] = "ttir.all_gather"(%[[MUL]]) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}>
        // CHECK: return %[[AG]]
        %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16>
        %1 = "ttir.all_reduce"(%arg1) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16>
        %2 = "ttir.multiply"(%0, %1) : (tensor<4096x4608xbf16>, tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16>
        return %2 : tensor<4096x4608xbf16>
      }
    }
  }
}
