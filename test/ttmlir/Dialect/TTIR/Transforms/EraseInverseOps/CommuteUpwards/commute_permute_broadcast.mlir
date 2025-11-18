// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_permute_broadcast_commute_upwards(%arg0: tensor<2048x1x1xbf16>) -> tensor<2048x1x2048xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.permute"(%arg0
        // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%[[RESHAPE]]
        %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 2048, 1>}> : (tensor<2048x1x1xbf16>) -> tensor<2048x2048x1xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 0, 2, 1>}> : (tensor<2048x2048x1xbf16>) -> tensor<2048x1x2048xbf16>
        return %3: tensor<2048x1x2048xbf16>
    }
}
