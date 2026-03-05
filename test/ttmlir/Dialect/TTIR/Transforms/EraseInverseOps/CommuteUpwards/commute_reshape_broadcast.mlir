// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_reshape_broadcast_commute_upwards(%arg0: tensor<2048x1x1xbf16>) -> tensor<2048x32x1x64xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%[[RESHAPE]]
        %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 2048, 1>}> : (tensor<2048x1x1xbf16>) -> tensor<2048x2048x1xbf16>
        %3 = "ttir.reshape"(%1) <{shape = [2048:i32, 32: i32, 1: i32, 64: i32]}> : (tensor<2048x2048x1xbf16>) -> tensor<2048x32x1x64xbf16>
        return %3: tensor<2048x32x1x64xbf16>
    }

    // Broadcast changes dim 6 (RTL), so commute should not happen.
    func.func @test_reshape_broadcast_not_commute_when_high_dim_changes(%arg0: tensor<1x1x1x1x1x1x1xbf16>) -> tensor<2x8x1x1x1x1x1xbf16> {
        // CHECK-LABEL: @test_reshape_broadcast_not_commute_when_high_dim_changes
        // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 16, 1, 1, 1, 1, 1, 1>}>
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[BROADCAST]])
        %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 16, 1, 1, 1, 1, 1, 1>}> : (tensor<1x1x1x1x1x1x1xbf16>) -> tensor<16x1x1x1x1x1x1xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [2:i32, 8:i32, 1:i32, 1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<16x1x1x1x1x1x1xbf16>) -> tensor<2x8x1x1x1x1x1xbf16>
        return %2: tensor<2x8x1x1x1x1x1xbf16>
    }
}
