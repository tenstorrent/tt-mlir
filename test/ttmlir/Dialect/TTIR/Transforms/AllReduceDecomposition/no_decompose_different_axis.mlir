// RUN: ttmlir-opt --ttir-all-reduce-decomposition %s | FileCheck %s

// Test: Two all_reduce ops with DIFFERENT cluster_axis should NOT be
// decomposed or fused.

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
      func.func @test_no_decompose_different_axis(%arg0: tensor<4096x4608xbf16>, %arg1: tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16> {
        // CHECK-LABEL: func.func @test_no_decompose_different_axis
        // Different cluster_axis values should prevent fusion.
        // CHECK: "ttir.all_reduce"{{.*}}cluster_axis = 0
        // CHECK: "ttir.all_reduce"{{.*}}cluster_axis = 1
        // CHECK-NOT: "ttir.reduce_scatter"
        // CHECK-NOT: "ttir.all_gather"
        %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16>
        %1 = "ttir.all_reduce"(%arg1) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16>
        %2 = "ttir.multiply"(%0, %1) : (tensor<4096x4608xbf16>, tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16>
        return %2 : tensor<4096x4608xbf16>
      }
    }
  }
}
