// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_permute_broadcast_commute_upwards(%arg0: tensor<2048x1x1xbf16>) -> tensor<2048x1x2048xbf16> {
        // After commuting upwards, the permute operates on dims of size 1, making it
        // an identity permute that gets folded away, leaving just the broadcast.
        // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 1, 2048>}>
        // CHECK-NOT: "ttir.permute"
        %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 2048, 1>}> : (tensor<2048x1x1xbf16>) -> tensor<2048x2048x1xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 0, 2, 1>}> : (tensor<2048x2048x1xbf16>) -> tensor<2048x1x2048xbf16>
        return %3: tensor<2048x1x2048xbf16>
    }
}
