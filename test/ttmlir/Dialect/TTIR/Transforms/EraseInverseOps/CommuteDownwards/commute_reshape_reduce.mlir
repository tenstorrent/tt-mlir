// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
    // Motivating LLaMA-70B RMS norm case: a leading reshape that only
    // rearranges outer dims while preserving the reduce dim gets pushed past
    // the reduce, leaving the reshape on the much smaller post-reduce tensor.
    func.func @test_reshape_sum_commute_downwards(%arg0: tensor<1x32x4096xbf16>) -> tensor<32x1x1xbf16> {
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%arg0
        // CHECK-SAME: dim_arg = [2 : i32]
        // CHECK-SAME: keep_dim = true
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SUM]]
        // CHECK-SAME: shape = [32 : i32, 1 : i32, 1 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>) -> tensor<32x1x4096xbf16>
        %2 = "ttir.sum"(%1) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<32x1x4096xbf16>) -> tensor<32x1x1xbf16>
        return %2 : tensor<32x1x1xbf16>
    }

    // keep_dim=false: reduce dim is dropped from the post-reduce shape, then
    // the trailing reshape compensates.
    func.func @test_reshape_sum_commute_downwards_keepdim_false(%arg0: tensor<1x32x4096xbf16>) -> tensor<32x1xbf16> {
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%arg0
        // CHECK-SAME: dim_arg = [2 : i32]
        // CHECK-SAME: keep_dim = false
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SUM]]
        // CHECK-SAME: shape = [32 : i32, 1 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>) -> tensor<32x1x4096xbf16>
        %2 = "ttir.sum"(%1) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x4096xbf16>) -> tensor<32x1xbf16>
        return %2 : tensor<32x1xbf16>
    }

    // Multi-dim reduce: every reduce dim must map back to the pre-reshape
    // shape; here both inner dims are preserved by the reshape.
    func.func @test_reshape_sum_commute_downwards_multi_dim(%arg0: tensor<1x32x4x16x64xbf16>) -> tensor<32x4x1x1xbf16> {
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%arg0
        // CHECK-SAME: dim_arg = [3 : i32, 4 : i32]
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SUM]]
        // CHECK-SAME: shape = [32 : i32, 4 : i32, 1 : i32, 1 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 4 : i32, 16 : i32, 64 : i32]}> : (tensor<1x32x4x16x64xbf16>) -> tensor<32x4x16x64xbf16>
        %2 = "ttir.sum"(%1) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<32x4x16x64xbf16>) -> tensor<32x4x1x1xbf16>
        return %2 : tensor<32x4x1x1xbf16>
    }

    // Negative dim_arg must be normalized before mapping.
    func.func @test_reshape_sum_commute_downwards_negative_dim(%arg0: tensor<1x32x4096xbf16>) -> tensor<32x1x1xbf16> {
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%arg0
        // CHECK-SAME: dim_arg = [2 : i32]
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SUM]]
        // CHECK-SAME: shape = [32 : i32, 1 : i32, 1 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>) -> tensor<32x1x4096xbf16>
        %2 = "ttir.sum"(%1) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<32x1x4096xbf16>) -> tensor<32x1x1xbf16>
        return %2 : tensor<32x1x1xbf16>
    }

    // Reduce dim was created by splitting an input dim — cannot map back, so
    // commute is rejected.
    func.func @test_reshape_sum_no_commute_split_reduce_dim(%arg0: tensor<32x4096xbf16>) -> tensor<32x64x1xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK-SAME: shape = [32 : i32, 64 : i32, 64 : i32]
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%[[RESHAPE]]
        // CHECK-SAME: dim_arg = [2 : i32]
        // CHECK: return %[[SUM]]
        %1 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 64 : i32, 64 : i32]}> : (tensor<32x4096xbf16>) -> tensor<32x64x64xbf16>
        %2 = "ttir.sum"(%1) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<32x64x64xbf16>) -> tensor<32x64x1xbf16>
        return %2 : tensor<32x64x1xbf16>
    }

    // Reduce dim is the merge of two input dims — cannot map back.
    func.func @test_reshape_sum_no_commute_merged_reduce_dim(%arg0: tensor<1x160x160x128xbf16>) -> tensor<1x1x128xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK-SAME: shape = [1 : i32, 25600 : i32, 128 : i32]
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%[[RESHAPE]]
        // CHECK-SAME: dim_arg = [1 : i32]
        // CHECK: return %[[SUM]]
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16>
        %2 = "ttir.sum"(%1) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<1x25600x128xbf16>) -> tensor<1x1x128xbf16>
        return %2 : tensor<1x1x128xbf16>
    }

    // The pattern is templated over multiple reduce ops — spot-check mean.
    func.func @test_reshape_mean_commute_downwards(%arg0: tensor<1x32x4096xbf16>) -> tensor<32x1x1xbf16> {
        // CHECK: %[[MEAN:[0-9]+]] = "ttir.mean"(%arg0
        // CHECK-SAME: dim_arg = [2 : i32]
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[MEAN]]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>) -> tensor<32x1x4096xbf16>
        %2 = "ttir.mean"(%1) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<32x1x4096xbf16>) -> tensor<32x1x1xbf16>
        return %2 : tensor<32x1x1xbf16>
    }

    // ...and max.
    func.func @test_reshape_max_commute_downwards(%arg0: tensor<1x32x4096xbf16>) -> tensor<32x1x1xbf16> {
        // CHECK: %[[MAX:[0-9]+]] = "ttir.max"(%arg0
        // CHECK-SAME: dim_arg = [2 : i32]
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[MAX]]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>) -> tensor<32x1x4096xbf16>
        %2 = "ttir.max"(%1) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<32x1x4096xbf16>) -> tensor<32x1x1xbf16>
        return %2 : tensor<32x1x1xbf16>
    }

    // No dim_arg means reduce-all. The pattern expands this into an explicit
    // dim_arg covering every input dim, so the resulting sum operates on the
    // pre-reshape input. Output shape matches the original — the trailing
    // identity reshape is folded away.
    func.func @test_reshape_sum_commute_downwards_reduce_all(%arg0: tensor<1x32x4096xbf16>) -> tensor<1x1x1xbf16> {
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%arg0
        // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32]
        // CHECK: return %[[SUM]]
        %1 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>) -> tensor<32x1x4096xbf16>
        %2 = "ttir.sum"(%1) <{keep_dim = true}> : (tensor<32x1x4096xbf16>) -> tensor<1x1x1xbf16>
        return %2 : tensor<1x1x1xbf16>
    }
}
