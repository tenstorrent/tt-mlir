// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_permute_binary_commute_downwards(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16> {
        // CHECK: %[[ADD:[0-9]+]] = "ttir.add"(%arg0, %arg1
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[ADD]]
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

module {
    func.func @test_reshape_binary_commute_downwards(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x1x3x50176xbf16> {
        // CHECK: %[[ADD:[0-9]+]] = "ttir.add"(%arg0, %arg1
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[ADD]]
        // CHECK: return %[[RESHAPE]]
        %0 = ttir.empty() : tensor<1x1x3x50176xbf16>
        %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 1 : i32, 3 : i32, 50176 : i32]}> : (tensor<1x3x224x224xbf16>, tensor<1x1x3x50176xbf16>) -> tensor<1x1x3x50176xbf16>
        %2 = ttir.empty() : tensor<1x1x3x50176xbf16>
        %3 = "ttir.reshape"(%arg1, %2) <{shape = [1 : i32, 1 : i32, 3 : i32, 50176 : i32]}> : (tensor<1x3x224x224xbf16>, tensor<1x1x3x50176xbf16>) -> tensor<1x1x3x50176xbf16>
        %4 = ttir.empty() : tensor<1x1x3x50176xbf16>
        %5 = "ttir.add"(%1, %3, %4) : (tensor<1x1x3x50176xbf16>, tensor<1x1x3x50176xbf16>, tensor<1x1x3x50176xbf16>) -> tensor<1x1x3x50176xbf16>
        return %5 : tensor<1x1x3x50176xbf16>
    }
}
