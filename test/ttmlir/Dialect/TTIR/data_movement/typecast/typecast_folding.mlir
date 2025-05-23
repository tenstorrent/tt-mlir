// RUN: ttmlir-opt --canonicalize %s | FileCheck %s

module attributes {} {
    // Test case to verify the folding of typecast operation.
    // CHECK-LABEL: typecast_folding_identity
    func.func @typecast_folding_identity(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
        // Verify that we fold the typecast when we try to cast to the same data type.
        // CHECK: return %arg0 : tensor<64x128xf32>
        %0 = "ttir.empty"() : () -> tensor<64x128xf32>
        %1 = "ttir.typecast"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
        return %1 : tensor<64x128xf32>
    }

    // Test case to verify consecutive narrowing typecast op folding.
    // CHECK-LABEL: typecast_folding_consecutive_narrowing_typecasts
    func.func @typecast_folding_consecutive_narrowing_typecasts(%arg0: tensor<64x128xf32>) -> tensor<64x128xi32> {
        // Verify that we fold two consecutive narrowing typecast ops into a single one.
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xi32>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xbf16>
        %1 = "ttir.typecast"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
        %2 = "ttir.empty"() : () -> tensor<64x128xi32>
        %3 = "ttir.typecast"(%1, %2) : (tensor<64x128xbf16>, tensor<64x128xi32>) -> tensor<64x128xi32>
        return %3 : tensor<64x128xi32>
    }

    // Test case to verify consecutive widening typecast op folding.
    // CHECK-LABEL: typecast_folding_consecutive_widening_typecasts
    func.func @typecast_folding_consecutive_widening_typecasts(%arg0: tensor<64x128xi8>) -> tensor<64x128xf32> {
        // Verify that we fold two consecutive narrowing typecast ops into a single one.
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xf32>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xbf16>
        %1 = "ttir.typecast"(%arg0, %0) : (tensor<64x128xi8>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
        %2 = "ttir.empty"() : () -> tensor<64x128xf32>
        %3 = "ttir.typecast"(%1, %2) : (tensor<64x128xbf16>, tensor<64x128xf32>) -> tensor<64x128xf32>
        return %3 : tensor<64x128xf32>
    }

    // Test case to verify widening-then-narrowing typecast op folding.
    // CHECK-LABEL: typecast_folding_widening_then_narrowing_typecasts
    func.func @typecast_folding_widening_then_narrowing_typecasts(%arg0: tensor<64x128xbf16>) -> tensor<64x128xf16> {
        // Verify that we fold widening-then-narrowing typecast ops into a single one.
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xf16>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xf32>
        %1 = "ttir.typecast"(%arg0, %0) : (tensor<64x128xbf16>, tensor<64x128xf32>) -> tensor<64x128xf32>
        %2 = "ttir.empty"() : () -> tensor<64x128xf16>
        %3 = "ttir.typecast"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf16>) -> tensor<64x128xf16>
        return %3 : tensor<64x128xf16>
    }

    // Test case to verify that we do fold consecutive FP -> Int -> FP typecast ops.
    // CHECK-LABEL: typecast_folding_decimal_truncating_typecasts
    func.func @typecast_folding_decimal_truncating_typecasts(%arg0: tensor<64x128xbf16>) -> tensor<64x128xf32> {
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xf32>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xi32>
        %1 = "ttir.typecast"(%arg0, %0) : (tensor<64x128xbf16>, tensor<64x128xi32>) -> tensor<64x128xi32>
        %2 = "ttir.empty"() : () -> tensor<64x128xf32>
        %3 = "ttir.typecast"(%1, %2) : (tensor<64x128xi32>, tensor<64x128xf32>) -> tensor<64x128xf32>
        return %3 : tensor<64x128xf32>
    }

    // Test case to verify that we do fold consecutive narrowing-then-widening typecast ops.
    // CHECK-LABEL: typecast_folding_narrowing_then_widening_typecasts
    func.func @typecast_folding_narrowing_then_widening_typecasts(%arg0: tensor<64x128xf16>) -> tensor<64x128xf32> {
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xf32>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xbf16>
        %1 = "ttir.typecast"(%arg0, %0) : (tensor<64x128xf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
        %2 = "ttir.empty"() : () -> tensor<64x128xf32>
        %3 = "ttir.typecast"(%1, %2) : (tensor<64x128xbf16>, tensor<64x128xf32>) -> tensor<64x128xf32>
        return %3 : tensor<64x128xf32>
    }

    // Test case to verify that we do not fold consecutive typecast ops if it's FP -> Int -> FP and we're in conservative mode.
    // CHECK-LABEL: typecast_folding_decimal_truncating_typecasts_conservative
    func.func @typecast_folding_decimal_truncating_typecasts_conservative(%arg0: tensor<64x128xbf16>) -> tensor<64x128xf32> {
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xi32>
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xf32>
        %0 = "ttir.empty"() : () -> tensor<64x128xi32>
        %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = true}> : (tensor<64x128xbf16>, tensor<64x128xi32>) -> tensor<64x128xi32>
        %2 = "ttir.empty"() : () -> tensor<64x128xf32>
        %3 = "ttir.typecast"(%1, %2) : (tensor<64x128xi32>, tensor<64x128xf32>) -> tensor<64x128xf32>
        return %3 : tensor<64x128xf32>
    }

    // Test case to verify that we do not fold consecutive typecast ops if it's narrowing-then-widening and we're in conservative mode.
    // CHECK-LABEL: typecast_folding_narrowing_then_widening_typecasts_conservative
    func.func @typecast_folding_narrowing_then_widening_typecasts_conservative(%arg0: tensor<64x128xf16>) -> tensor<64x128xf32> {
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xbf16>
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xf32>
        %0 = "ttir.empty"() : () -> tensor<64x128xbf16>
        %1 = "ttir.typecast"(%arg0, %0) : (tensor<64x128xf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
        %2 = "ttir.empty"() : () -> tensor<64x128xf32>
        %3 = "ttir.typecast"(%1, %2) <{conservative_folding = true}> : (tensor<64x128xbf16>, tensor<64x128xf32>) -> tensor<64x128xf32>
        return %3 : tensor<64x128xf32>
    }

    // Test case to verify that we do fold consecutive typecast ops if it's narrowing-then-narrowing and we're in conservative mode.
    // CHECK-LABEL: typecast_folding_narrowing_then_narrowing_typecasts_conservative
    func.func @typecast_folding_narrowing_then_narrowing_typecasts_conservative(%arg0: tensor<64x128xui32>) -> tensor<64x128xui8> {
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xui8>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xui16>
        %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = true}> : (tensor<64x128xui32>, tensor<64x128xui16>) -> tensor<64x128xui16>
        %2 = "ttir.empty"() : () -> tensor<64x128xui8>
        %3 = "ttir.typecast"(%1, %2) : (tensor<64x128xui16>, tensor<64x128xui8>) -> tensor<64x128xui8>
        return %3 : tensor<64x128xui8>
    }

    // Test case to verify that we do fold consecutive typecast ops if it's widening-then-widening and we're in conservative mode.
    // CHECK-LABEL: typecast_folding_widening_then_widening_typecasts_conservative
    func.func @typecast_folding_widening_then_widening_typecasts_conservative(%arg0: tensor<64x128xui8>) -> tensor<64x128xi32> {
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xi32>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xui16>
        %1 = "ttir.typecast"(%arg0, %0) : (tensor<64x128xui8>, tensor<64x128xui16>) -> tensor<64x128xui16>
        %2 = "ttir.empty"() : () -> tensor<64x128xi32>
        %3 = "ttir.typecast"(%1, %2) <{conservative_folding = true}> : (tensor<64x128xui16>, tensor<64x128xi32>) -> tensor<64x128xi32>
        return %3 : tensor<64x128xi32>
    }

    // Test case to verify that we do fold consecutive typecast ops if it's widening-then-narrowing and we're in conservative mode.
    // CHECK-LABEL: typecast_folding_widening_then_narrowing_typecasts_conservative
    func.func @typecast_folding_widening_then_narrowing_typecasts_conservative(%arg0: tensor<64x128xui16>) -> tensor<64x128xui8> {
        // CHECK: ttir.typecast
        // CHECK-SAME: -> tensor<64x128xui8>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xi32>
        %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = true}> : (tensor<64x128xui16>, tensor<64x128xi32>) -> tensor<64x128xi32>
        %2 = "ttir.empty"() : () -> tensor<64x128xui8>
        %3 = "ttir.typecast"(%1, %2) : (tensor<64x128xi32>, tensor<64x128xui8>) -> tensor<64x128xui8>
        return %3 : tensor<64x128xui8>
    }

    // Test case to verify that two conservative typecast ops folds into a conservative typecast op.
    // CHECK-LABEL: typecast_folding_consecutive_conservative_typecasts
    func.func @typecast_folding_consecutive_conservative_typecasts(%arg0: tensor<64x128xui8>) -> tensor<64x128xf32> {
        // CHECK: ttir.typecast
        // CHECK-SAME: <{conservative_folding = true}>
        // CHECK-SAME: -> tensor<64x128xf32>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xf16>
        %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = true}> : (tensor<64x128xui8>, tensor<64x128xf16>) -> tensor<64x128xf16>
        %2 = "ttir.empty"() : () -> tensor<64x128xf32>
        %3 = "ttir.typecast"(%1, %2) <{conservative_folding = true}> : (tensor<64x128xf16>, tensor<64x128xf32>) -> tensor<64x128xf32>
        return %3 : tensor<64x128xf32>
    }

    // Test case to verify that a conservative and a non-conservative typecast ops folds into a non-conservative typecast op.
    // CHECK-LABEL: typecast_folding_consecutive_conservative_and_non_conservative_typecasts
    func.func @typecast_folding_consecutive_conservative_and_non_conservative_typecasts(%arg0: tensor<64x128xf32>) -> tensor<64x128xui16> {
        // CHECK: ttir.typecast
        // CHECK-SAME: <{conservative_folding = false}>
        // CHECK-SAME: -> tensor<64x128xui16>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xbf16>
        %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = true}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
        %2 = "ttir.empty"() : () -> tensor<64x128xui16>
        %3 = "ttir.typecast"(%1, %2) <{conservative_folding = false}> : (tensor<64x128xbf16>, tensor<64x128xui16>) -> tensor<64x128xui16>
        return %3 : tensor<64x128xui16>
    }

    // Test case to verify that a non-conservative and a conservative typecast ops folds into a non-conservative typecast op.
    // CHECK-LABEL: typecast_folding_consecutive_non_conservative_and_conservative_typecasts
    func.func @typecast_folding_consecutive_non_conservative_and_conservative_typecasts(%arg0: tensor<64x128xf32>) -> tensor<64x128xui16> {
        // CHECK: ttir.typecast
        // CHECK-SAME: <{conservative_folding = false}>
        // CHECK-SAME: -> tensor<64x128xui16>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xi32>
        %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = false}> : (tensor<64x128xf32>, tensor<64x128xi32>) -> tensor<64x128xi32>
        %2 = "ttir.empty"() : () -> tensor<64x128xui16>
        %3 = "ttir.typecast"(%1, %2) <{conservative_folding = true}> : (tensor<64x128xi32>, tensor<64x128xui16>) -> tensor<64x128xui16>
        return %3 : tensor<64x128xui16>
    }

    // Test cast to very that two non-conservative typecast ops folds into a non-conservative typecast op.
    // CHECK-LABEL: typecast_folding_consecutive_non_conservative_typecasts
    func.func @typecast_folding_consecutive_non_conservative_typecasts(%arg0: tensor<64x128xbf16>) -> tensor<64x128xui8> {
        // CHECK: ttir.typecast
        // CHECK-SAME: <{conservative_folding = true}>
        // CHECK-SAME: -> tensor<64x128xui8>
        // CHECK-NOT: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xf16>
        %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = true}> : (tensor<64x128xbf16>, tensor<64x128xf16>) -> tensor<64x128xf16>
        %2 = "ttir.empty"() : () -> tensor<64x128xui8>
        %3 = "ttir.typecast"(%1, %2) <{conservative_folding = true}> : (tensor<64x128xf16>, tensor<64x128xui8>) -> tensor<64x128xui8>
        return %3 : tensor<64x128xui8>
    }

    // Test case to verify that we do not fold consecutive typecast ops if the first typecast have more than a single use.
    // CHECK-LABEL: typecast_folding_consecutive_typecasts_with_multiple_uses
    func.func @typecast_folding_consecutive_typecasts_with_multiple_uses(%arg0: tensor<64x128xf32>) -> (tensor<64x128xbf16>, tensor<64x128xi32>) {
        // Verify that both typecasts exists.
        // CHECK: ttir.typecast
        // CHECK: ttir.typecast
        %0 = "ttir.empty"() : () -> tensor<64x128xi32>
        %1 = "ttir.typecast"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xi32>) -> tensor<64x128xi32>
        %2 = "ttir.empty"() : () -> tensor<64x128xbf16>
        %3 = "ttir.typecast"(%0, %2) : (tensor<64x128xi32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
        %4 = "ttir.empty"() : () -> tensor<64x128xi32>
        %5 = "ttir.add"(%1, %1, %4) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
        return %3, %5 : tensor<64x128xbf16>, tensor<64x128xi32>
    }
}
