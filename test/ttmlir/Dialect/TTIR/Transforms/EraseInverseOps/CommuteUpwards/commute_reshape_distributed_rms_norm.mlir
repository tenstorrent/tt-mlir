// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    // Basic case: inverse reshapes around distributed_rms_norm are eliminated.
    // reshape(32x2048 -> 32x1x2048) -> distributed_rms_norm -> reshape(32x1x2048 -> 32x2048)
    // The reshape pair should be folded away.
    func.func @test_reshape_distributed_rms_norm_inverse_reshape_upwards(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<32x2048xbf16> {
        // CHECK: "ttir.distributed_rms_norm"(%arg0, %arg1)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK-NOT: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        return %2 : tensor<32x2048xbf16>
    }

    // Case with residual: both input and residual must be reshaped when the
    // reshape commutes above distributed_rms_norm.
    func.func @test_reshape_distributed_rms_norm_with_residual(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>, %arg2: tensor<32x2048xbf16>) -> tensor<32x2048xbf16> {
        // CHECK-LABEL: @test_reshape_distributed_rms_norm_with_residual
        // CHECK: "ttir.distributed_rms_norm"(%arg0, %arg1, %arg2)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<2048xbf16>, tensor<32x2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK-NOT: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.reshape"(%arg2) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.distributed_rms_norm"(%0, %arg1, %1) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        %3 = "ttir.reshape"(%2) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        return %3 : tensor<32x2048xbf16>
    }

    // Case with multiple identical reshape users - all should be replaced.
    func.func @test_reshape_distributed_rms_norm_multiple_identical_users(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> (tensor<32x2048xbf16>, tensor<32x2048xbf16>) {
        // CHECK-LABEL: @test_reshape_distributed_rms_norm_multiple_identical_users
        // CHECK: "ttir.distributed_rms_norm"(%arg0, %arg1)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK-NOT: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        %3 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        return %2, %3 : tensor<32x2048xbf16>, tensor<32x2048xbf16>
    }

    // Negative case: reshape changes the last dimension (normalization dim).
    // Should NOT commute because the last dim is different.
    func.func @test_reshape_distributed_rms_norm_last_dim_changes(%arg0: tensor<1x32x64xbf16>, %arg1: tensor<2048xbf16>) -> tensor<1x32x64xbf16> {
        // CHECK-LABEL: @test_reshape_distributed_rms_norm_last_dim_changes
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.distributed_rms_norm"
        // CHECK: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 2048 : i32]}> : (tensor<1x32x64xbf16>) -> tensor<1x2048xbf16>
        %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x2048xbf16>, tensor<2048xbf16>) -> tensor<1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 64 : i32]}> : (tensor<1x2048xbf16>) -> tensor<1x32x64xbf16>
        return %2 : tensor<1x32x64xbf16>
    }

    // Negative case: distributed_rms_norm has a non-reshape user.
    // Should NOT commute because not all users are identical reshapes.
    func.func @test_reshape_distributed_rms_norm_non_reshape_user(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> (tensor<32x2048xbf16>, tensor<32x1x2048xbf16>) {
        // CHECK-LABEL: @test_reshape_distributed_rms_norm_non_reshape_user
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.distributed_rms_norm"
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.relu"
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        %3 = "ttir.relu"(%1) : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        return %2, %3 : tensor<32x2048xbf16>, tensor<32x1x2048xbf16>
    }
}
