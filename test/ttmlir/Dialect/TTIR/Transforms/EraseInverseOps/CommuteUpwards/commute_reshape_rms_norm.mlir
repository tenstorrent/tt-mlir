// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    // Basic case: inverse reshapes around rms_norm are eliminated.
    // reshape(32x2048 -> 32x1x2048) -> rms_norm -> reshape(32x1x2048 -> 32x2048)
    // The reshape pair should be folded away.
    func.func @test_reshape_rms_norm_inverse_reshape_upwards(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<32x2048xbf16> {
        // CHECK: "ttir.rms_norm"(%arg0, %arg1)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK-NOT: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.rms_norm"(%0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 2048>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        return %2 : tensor<32x2048xbf16>
    }

    // Case with multiple identical reshape users - all should be replaced.
    func.func @test_reshape_rms_norm_multiple_identical_users(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> (tensor<32x2048xbf16>, tensor<32x2048xbf16>) {
        // CHECK: "ttir.rms_norm"(%arg0, %arg1)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK-NOT: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.rms_norm"(%0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 2048>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        %3 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        return %2, %3 : tensor<32x2048xbf16>, tensor<32x2048xbf16>
    }

    // Negative case: reshape changes the last dimension (normalization dim).
    // Should NOT commute because the last dim is different.
    func.func @test_reshape_rms_norm_last_dim_changes(%arg0: tensor<1x32x64xbf16>, %arg1: tensor<2048xbf16>) -> tensor<1x32x64xbf16> {
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.rms_norm"
        // CHECK: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 2048 : i32]}> : (tensor<1x32x64xbf16>) -> tensor<1x2048xbf16>
        %1 = "ttir.rms_norm"(%0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 2048>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x2048xbf16>, tensor<2048xbf16>) -> tensor<1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 64 : i32]}> : (tensor<1x2048xbf16>) -> tensor<1x32x64xbf16>
        return %2 : tensor<1x32x64xbf16>
    }

    // Negative case: reshapes are not inverses (different output shape).
    // Should NOT commute because shapes don't match.
    func.func @test_reshape_rms_norm_not_inverse(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<16x2x2048xbf16> {
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.rms_norm"
        // CHECK: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.rms_norm"(%0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 2048>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [16 : i32, 2 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<16x2x2048xbf16>
        return %2 : tensor<16x2x2048xbf16>
    }

    // Negative case: no input reshape.
    // Should NOT commute because there's no input reshape to eliminate.
    func.func @test_reshape_rms_norm_no_input_reshape(%arg0: tensor<32x1x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<32x2048xbf16> {
        // CHECK: "ttir.rms_norm"
        // CHECK: "ttir.reshape"
        %1 = "ttir.rms_norm"(%arg0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 2048>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        return %2 : tensor<32x2048xbf16>
    }

    // Negative case: rms_norm has non-reshape user.
    // Should NOT commute because not all users are identical reshapes.
    func.func @test_reshape_rms_norm_non_reshape_user(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> (tensor<32x2048xbf16>, tensor<32x1x2048xbf16>) {
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.rms_norm"
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.relu"
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.rms_norm"(%0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 2048>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        %3 = "ttir.relu"(%1) : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        return %2, %3 : tensor<32x2048xbf16>, tensor<32x1x2048xbf16>
    }
}

