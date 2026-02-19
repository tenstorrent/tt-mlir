// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    // After commuting permute upwards through broadcast, the permute becomes
    // permute(2048x1x1, [0,2,1]) which swaps dims 1 and 2 (both size 1).
    // This is an identity permute and gets folded away, leaving just broadcast.
    func.func @test_permute_broadcast_commute_upwards(%arg0: tensor<2048x1x1xbf16>) -> tensor<2048x1x2048xbf16> {
        // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 1, 2048>}>
        // CHECK-NEXT: return %[[BROADCAST]]
        %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 2048, 1>}> : (tensor<2048x1x1xbf16>) -> tensor<2048x2048x1xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 0, 2, 1>}> : (tensor<2048x2048x1xbf16>) -> tensor<2048x1x2048xbf16>
        return %3: tensor<2048x1x2048xbf16>
    }

    func.func @test_permute_broadcast_commute_upwards_non_identity_permute(%arg0: tensor<2x1x3xbf16>) -> tensor<4x2x3xbf16> {
        // CHECK-LABEL: @test_permute_broadcast_commute_upwards_non_identity_permute
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0)
        // CHECK-SAME: permutation = array<i64: 1, 0, 2>
        // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%[[PERMUTE]])
        // CHECK-SAME: broadcast_dimensions = array<i64: 4, 1, 1>
        // CHECK-NEXT: return %[[BROADCAST]]
        %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<2x1x3xbf16>) -> tensor<2x4x3xbf16>
        %2 = "ttir.permute"(%1) <{permutation = array<i64: 1, 0, 2>}> : (tensor<2x4x3xbf16>) -> tensor<4x2x3xbf16>
        return %2: tensor<4x2x3xbf16>
    }

    func.func @test_permute_broadcast_not_commute_when_dim6_changes(%arg0: tensor<1x1x1x1x1x1x2xbf16>) -> tensor<1x16x1x1x1x1x2xbf16> {
        // CHECK-LABEL: @test_permute_broadcast_not_commute_when_dim6_changes
        // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 16, 1, 1, 1, 1, 1, 1>}>
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[BROADCAST]])
        %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 16, 1, 1, 1, 1, 1, 1>}> : (tensor<1x1x1x1x1x1x2xbf16>) -> tensor<16x1x1x1x1x1x2xbf16>
        %2 = "ttir.permute"(%1) <{permutation = array<i64: 1, 0, 2, 3, 4, 5, 6>}> : (tensor<16x1x1x1x1x1x2xbf16>) -> tensor<1x16x1x1x1x1x2xbf16>
        return %2: tensor<1x16x1x1x1x1x2xbf16>
    }

}
