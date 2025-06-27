// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" %s | FileCheck %s

module {
    func.func @test_commute_permute(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16> {
        // CHECK: %[[ADD:[0-9]+]] = "ttir.add"(%arg0, %arg1
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"
        // CHECK: permutation = array<i64: 0, 2, 3, 1>
        // CHECK: return %[[PERMUTE]]
        %0 = tensor.empty() : tensor<1x224x224x3xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>, tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
        %2 = tensor.empty() : tensor<1x224x224x3xbf16>
        %3 = "ttir.permute"(%arg1, %2) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>, tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
        %4 = tensor.empty() : tensor<1x224x224x3xbf16>
        %5 = "ttir.add"(%1, %3, %4) : (tensor<1x224x224x3xbf16>, tensor<1x224x224x3xbf16>, tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
        return %5: tensor<1x224x224x3xbf16>
    }
}
