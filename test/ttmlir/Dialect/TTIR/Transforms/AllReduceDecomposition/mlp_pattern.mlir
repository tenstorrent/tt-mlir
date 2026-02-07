// RUN: ttmlir-opt --ttir-all-reduce-decomposition %s | FileCheck %s

// Test: MLP-style pattern with reshape and silu operations between
// all_reduce and the join point. The all_gathers should be commuted through
// reshape and silu, then fused at the multiply join point.

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
      func.func @test_mlp_pattern(%arg0: tensor<4096x4608xbf16>, %arg1: tensor<4096x4608xbf16>) -> tensor<128x32x4608xbf16> {
        // CHECK-LABEL: func.func @test_mlp_pattern
        // CHECK: %[[RS0:.*]] = "ttir.reduce_scatter"(%arg0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<4096x4608xbf16>) -> tensor<4096x2304xbf16>
        // CHECK: %[[RESHAPE0:.*]] = "ttir.reshape"(%[[RS0]]) <{shape = [128 : i32, 32 : i32, 2304 : i32]}> : (tensor<4096x2304xbf16>) -> tensor<128x32x2304xbf16>
        // CHECK: %[[SILU:.*]] = "ttir.silu"(%[[RESHAPE0]]) : (tensor<128x32x2304xbf16>) -> tensor<128x32x2304xbf16>
        // CHECK: %[[RS1:.*]] = "ttir.reduce_scatter"(%arg1) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<4096x4608xbf16>) -> tensor<4096x2304xbf16>
        // CHECK: %[[RESHAPE1:.*]] = "ttir.reshape"(%[[RS1]]) <{shape = [128 : i32, 32 : i32, 2304 : i32]}> : (tensor<4096x2304xbf16>) -> tensor<128x32x2304xbf16>
        // CHECK: %[[MUL:.*]] = "ttir.multiply"(%[[SILU]], %[[RESHAPE1]]) : (tensor<128x32x2304xbf16>, tensor<128x32x2304xbf16>) -> tensor<128x32x2304xbf16>
        // CHECK: %[[AG:.*]] = "ttir.all_gather"(%[[MUL]]) <{all_gather_dim = 2 : si32, cluster_axis = 0 : ui32}> : (tensor<128x32x2304xbf16>) -> tensor<128x32x4608xbf16>
        // CHECK: return %[[AG]]
        %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16>
        %1 = "ttir.reshape"(%0) <{shape = [128 : i32, 32 : i32, 4608 : i32]}> : (tensor<4096x4608xbf16>) -> tensor<128x32x4608xbf16>
        %2 = "ttir.silu"(%1) : (tensor<128x32x4608xbf16>) -> tensor<128x32x4608xbf16>
        %3 = "ttir.all_reduce"(%arg1) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096x4608xbf16>) -> tensor<4096x4608xbf16>
        %4 = "ttir.reshape"(%3) <{shape = [128 : i32, 32 : i32, 4608 : i32]}> : (tensor<4096x4608xbf16>) -> tensor<128x32x4608xbf16>
        %5 = "ttir.multiply"(%2, %4) : (tensor<128x32x4608xbf16>, tensor<128x32x4608xbf16>) -> tensor<128x32x4608xbf16>
        return %5 : tensor<128x32x4608xbf16>
      }
    }
  }
}
