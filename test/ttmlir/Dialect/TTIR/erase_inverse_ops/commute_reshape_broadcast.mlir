// RUN: ttmlir-opt --ttir-erase-inverse-ops %s | FileCheck %s

module {
    func.func @test_commute_reshape_through_broadcast(%arg0: tensor<2048x1x1xbf16>) -> tensor<2048x1x2048x1xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0,
        // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%[[RESHAPE]]
        %0 = tensor.empty() : tensor<2048x2048x1xbf16>
        %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 2048, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x2048x1xbf16>) -> tensor<2048x2048x1xbf16>
        %2 = tensor.empty() : tensor<2048x1x2048x1xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [2048:i32, 1: i32, 2048: i32, 1: i32]}> : (tensor<2048x2048x1xbf16>, tensor<2048x1x2048x1xbf16>) -> tensor<2048x1x2048x1xbf16>
        return %3: tensor<2048x1x2048x1xbf16>
    }
}
