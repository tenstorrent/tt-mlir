// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
    // A reduce with reduction length > 1 changes total element count, so the
    // new reshape of the pre-reduce input cannot produce the original
    // post-reshape shape; the pattern must refuse the commute.
    func.func @test_reshape_sum_no_commute_upwards(%arg0: tensor<1x32x4096xbf16>) -> tensor<32x1xbf16> {
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%arg0
        // CHECK-SAME: dim_arg = [2 : i32]
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SUM]]
        // CHECK-SAME: shape = [32 : i32, 1 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x32x4096xbf16>) -> tensor<1x32x1xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 1 : i32]}> : (tensor<1x32x1xbf16>) -> tensor<32x1xbf16>
        return %2 : tensor<32x1xbf16>
    }

    // keep_dim=false version of the same scenario — still cannot commute.
    func.func @test_reshape_sum_no_commute_upwards_keepdim_false(%arg0: tensor<1x32x4096xbf16>) -> tensor<32xbf16> {
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%arg0
        // CHECK-SAME: keep_dim = false
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SUM]]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x32x4096xbf16>) -> tensor<1x32xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32]}> : (tensor<1x32xbf16>) -> tensor<32xbf16>
        return %2 : tensor<32xbf16>
    }

    // Trivial reduce (size-1 reduce dim) preserves element count; if the
    // reshape also preserves a compatibly-strided size-1 dim, the pattern can
    // commute the reshape upwards.
    func.func @test_reshape_sum_commute_upwards_trivial(%arg0: tensor<2x1x1x8xbf16>) -> tensor<2x1x8xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK-SAME: shape = [2 : i32, 1 : i32, 8 : i32]
        // CHECK: %[[SUM:[0-9]+]] = "ttir.sum"(%[[RESHAPE]]
        // CHECK: return %[[SUM]]
        %1 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<2x1x1x8xbf16>) -> tensor<2x1x1x8xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [2 : i32, 1 : i32, 8 : i32]}> : (tensor<2x1x1x8xbf16>) -> tensor<2x1x8xbf16>
        return %2 : tensor<2x1x8xbf16>
    }
}
