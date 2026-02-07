// RUN: ttmlir-opt --ttir-all-reduce-decomposition %s | FileCheck %s

// Test: A single all_reduce without a fusable pair should NOT be decomposed.

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
      func.func @test_no_decompose_single(%arg0: tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16> {
        // CHECK-LABEL: func.func @test_no_decompose_single
        // A single all_reduce without a fusable pair should remain unchanged.
        // CHECK: "ttir.all_reduce"
        // CHECK-NOT: "ttir.reduce_scatter"
        // CHECK-NOT: "ttir.all_gather"
        %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16>
        return %0 : tensor<4096x4608xbf16>
      }
    }
  }
}
