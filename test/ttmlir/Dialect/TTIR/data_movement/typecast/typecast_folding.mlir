// RUN: ttmlir-opt --canonicalize %s | FileCheck %s

module attributes {} {
    // Test case to verify the folding of typecast operation.
    func.func @typecast_folding(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
        // Verify that we fold the typecast when we try to cast to the same data type.
        // CHECK: return %arg0 : tensor<64x128xf32>
        %0 = "ttir.empty"() : () -> tensor<64x128xf32>
        %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
        return %1 : tensor<64x128xf32>
    }

    // Test case to verify consecutive typecast op folding.
    func.func @typecast_folding_consecutive_typecasts(%arg0: tensor<64x128xf32>) -> tensor<64x128xbf16> {
        // Verify that we fold two consecutive typecast ops into a single one.
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xbf16>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xi32>
        %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xi32>) -> tensor<64x128xi32>
        %2 = "ttir.empty"() : () -> tensor<64x128xbf16>
        %3 = "ttir.typecast"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xi32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
        return %3 : tensor<64x128xbf16>
    }

    // Test case to verify that we do not fold consecutive typecast ops if the first typecast have more than a single use.
    func.func @typecast_folding_consecutive_typecasts_with_multiple_uses(%arg0: tensor<64x128xf32>) -> (tensor<64x128xbf16>, tensor<64x128xi32>) {
        // Verify that both typecasts exists.
        // CHECK: ttir.typecast
        // CHECK: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xi32>
        %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xi32>) -> tensor<64x128xi32>
        %2 = "ttir.empty"() : () -> tensor<64x128xbf16>
        %3 = "ttir.typecast"(%0, %2) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xi32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
        %4 = "ttir.empty"() : () -> tensor<64x128xi32>
        %5 = "ttir.add"(%1, %1, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
        return %3, %5 : tensor<64x128xbf16>, tensor<64x128xi32>
    }
}
