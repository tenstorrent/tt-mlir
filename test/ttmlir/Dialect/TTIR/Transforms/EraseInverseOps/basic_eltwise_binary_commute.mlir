// RUN: ttmlir-opt --ttir-erase-inverse-ops %s | FileCheck %s

module {
    func.func @test_commute_permute(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16> {
        // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"
        // CHECK: %[[PERMUTE2:[0-9]+]] = "ttir.permute"
        // CHECK: %[[ADD:[0-9]+]] = "ttir.add"(%[[PERMUTE1]], %[[PERMUTE2]]
        // CHECK: return %[[ADD]]
        %0 = tensor.empty() : tensor<1x3x224x224xbf16>
        %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x3x224x224xbf16>, tensor<1x3x224x224xbf16>, tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16>
        %2 = tensor.empty() : tensor<1x224x224x3xbf16>
        %3 = "ttir.permute"(%1, %2) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>, tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
        return %3: tensor<1x224x224x3xbf16>
    }
}
