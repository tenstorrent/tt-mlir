// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-upwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test to verify if where-reshape commute is implemented.
// Currently, ElementwiseTernary is NOT included in populateElementwiseCommutePatterns,
// so this test expects the commute NOT to happen (reshape -> where order preserved).

module {
    func.func @test_reshape_where_commute_downwards(%arg0: tensor<1x3x224x224xi1>, %arg1: tensor<1x3x224x224xbf16>, %arg2: tensor<1x3x224x224xbf16>) -> tensor<1x1x3x50176xbf16> {
        // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%arg1
        // CHECK: %[[RESHAPE3:[0-9]+]] = "ttir.reshape"(%arg2
        // CHECK: %[[WHERE:[0-9]+]] = "ttir.where"(%[[RESHAPE1]], %[[RESHAPE2]], %[[RESHAPE3]]
        // CHECK: return %[[WHERE]]
        // If commute is implemented, uncomment below and comment above:
        // CHECK: %[[WHERE:[0-9]+]] = "ttir.where"(%arg0, %arg1, %arg2
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[WHERE]]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 3 : i32, 50176 : i32]}> : (tensor<1x3x224x224xi1>) -> tensor<1x1x3x50176xi1>
        %3 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 3 : i32, 50176 : i32]}> : (tensor<1x3x224x224xbf16>) -> tensor<1x1x3x50176xbf16>
        %5 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 3 : i32, 50176 : i32]}> : (tensor<1x3x224x224xbf16>) -> tensor<1x1x3x50176xbf16>
        %7 = "ttir.where"(%1, %3, %5) : (tensor<1x1x3x50176xi1>, tensor<1x1x3x50176xbf16>, tensor<1x1x3x50176xbf16>) -> tensor<1x1x3x50176xbf16>
        return %7 : tensor<1x1x3x50176xbf16>
    }
}

module {
    func.func @test_reshape_where_commute_downwards_simple(%arg0: tensor<32x64xi1>, %arg1: tensor<32x64xbf16>, %arg2: tensor<32x64xbf16>) -> tensor<64x32xbf16> {
        // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%arg1
        // CHECK: %[[RESHAPE3:[0-9]+]] = "ttir.reshape"(%arg2
        // CHECK: %[[WHERE:[0-9]+]] = "ttir.where"(%[[RESHAPE1]], %[[RESHAPE2]], %[[RESHAPE3]]
        // CHECK: return %[[WHERE]]
        // If commute is implemented, uncomment below and comment above:
        // CHECK: %[[WHERE:[0-9]+]] = "ttir.where"(%arg0, %arg1, %arg2
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[WHERE]]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [64 : i32, 32 : i32]}> : (tensor<32x64xi1>) -> tensor<64x32xi1>
        %3 = "ttir.reshape"(%arg1) <{shape = [64 : i32, 32 : i32]}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
        %5 = "ttir.reshape"(%arg2) <{shape = [64 : i32, 32 : i32]}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
        %7 = "ttir.where"(%1, %3, %5) : (tensor<64x32xi1>, tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
        return %7 : tensor<64x32xbf16>
    }
}
