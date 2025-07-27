// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_permute_binary_commute_upwards(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16> {
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

module {
    func.func @test_reshape_binary_commute_upwards(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x1x3x50176xbf16> {
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"
    // CHECK: %[[ADD:[0-9]+]] = "ttir.add"(%[[RESHAPE1]], %[[RESHAPE2]]
    // CHECK: return %[[ADD]]
    %0 = tensor.empty() : tensor<1x3x224x224xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x3x224x224xbf16>, tensor<1x3x224x224xbf16>, tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16>
    %2 = tensor.empty() : tensor<1x1x3x50176xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [1:i32, 1: i32, 3: i32, 50176: i32]}> : (tensor<1x3x224x224xbf16>, tensor<1x1x3x50176xbf16>) -> tensor<1x1x3x50176xbf16>
    return %3: tensor<1x1x3x50176xbf16>
    }
}
