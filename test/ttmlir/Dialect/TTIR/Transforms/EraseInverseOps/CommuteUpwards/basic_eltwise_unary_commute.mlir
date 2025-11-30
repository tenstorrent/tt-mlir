// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
    func.func @test_commute_identical_users(%arg0: tensor<32x64xbf16>) -> (tensor<64x32xbf16>, tensor<64x32xbf16>) {
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[PERMUTE]]
        // CHECK: return %[[EXP]], %[[EXP]]
        %1 = "ttir.exp"(%arg0) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
        %5 = "ttir.permute"(%1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
        return %3, %5 : tensor<64x32xbf16>, tensor<64x32xbf16>
    }
}

module {
    func.func @test_dont_commute_different_users(%arg0: tensor<32x64xbf16>) -> (tensor<64x32xbf16>, tensor<1x2048xbf16>) {
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[EXP]]
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[EXP]]
        // CHECK: return %[[PERMUTE]], %[[RESHAPE]]
        %1 = "ttir.exp"(%arg0) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
        %5 = "ttir.reshape"(%1) <{shape = [1: i32, 2048: i32]}> : (tensor<32x64xbf16>) -> tensor<1x2048xbf16>
        return %3, %5 : tensor<64x32xbf16>, tensor<1x2048xbf16>
    }
}

module {
    func.func @test_commute_reshape(%arg0: tensor<32x64xbf16>) -> tensor<1x2048xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[RESHAPE]]
        // CHECK: return %[[EXP]]
        %1 = "ttir.exp"(%arg0) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
        %3 = "ttir.reshape"(%1) <{shape = [1: i32, 2048: i32]}> : (tensor<32x64xbf16>) -> tensor<1x2048xbf16>
        return %3: tensor<1x2048xbf16>
    }
}

module {
    func.func @test_commute_permute(%arg0: tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16> {
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[PERMUTE]]
        // CHECK: return %[[EXP]]
        %1 = "ttir.exp"(%arg0) : (tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16>
        return %3: tensor<1x224x224x3xbf16>
    }
}
