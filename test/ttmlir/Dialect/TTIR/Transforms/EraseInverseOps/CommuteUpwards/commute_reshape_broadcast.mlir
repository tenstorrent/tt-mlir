// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-downwards=false" %s | FileCheck %s

module {
    func.func @test_reshape_broadcast_commute_upwards(%arg0: tensor<2048x1x1xbf16>) -> tensor<2048x32x1x64xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0,
        // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%[[RESHAPE]]
        %0 = tensor.empty() : tensor<2048x2048x1xbf16>
        %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 2048, 1>}> : (tensor<2048x1x1xbf16>, tensor<2048x2048x1xbf16>) -> tensor<2048x2048x1xbf16>
        %2 = tensor.empty() : tensor<2048x32x1x64xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [2048:i32, 32: i32, 1: i32, 64: i32]}> : (tensor<2048x2048x1xbf16>, tensor<2048x32x1x64xbf16>) -> tensor<2048x32x1x64xbf16>
        return %3: tensor<2048x32x1x64xbf16>
    }
}
