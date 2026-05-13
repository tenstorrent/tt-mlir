// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    // Inverse reshapes around distributed_rms_norm should be eliminated.
    func.func @test_distributed_rms_norm_inverse_reshape(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<32x2048xbf16> {
        // CHECK: "ttir.distributed_rms_norm"(%arg0, %arg1)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK-NOT: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{cluster_axis = 0 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        return %2 : tensor<32x2048xbf16>
    }

    // Pattern observed in llama_3_1_8b prefill: add feeds distributed_rms_norm,
    // bracketed by inverse reshapes that should commute upwards through both
    // ops and cancel.
    func.func @test_distributed_rms_norm_with_add_residual(%arg0: tensor<32x2048xbf16>, %residual: tensor<32x1x2048xbf16>, %weight: tensor<2048xbf16>) -> tensor<32x2048xbf16> {
        // CHECK-LABEL: func.func @test_distributed_rms_norm_with_add_residual
        // CHECK: %[[RESIDUAL_RS:.*]] = "ttir.reshape"(%arg1)
        // CHECK-SAME: -> tensor<32x2048xbf16>
        // CHECK: %[[ADD:.*]] = "ttir.add"(%[[RESIDUAL_RS]], %arg0)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<32x2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK: "ttir.distributed_rms_norm"(%[[ADD]], %arg2)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK-NOT: tensor<32x1x2048xbf16>
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.add"(%residual, %0) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        %2 = "ttir.distributed_rms_norm"(%1, %weight) <{cluster_axis = 0 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        %3 = "ttir.reshape"(%2) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        return %3 : tensor<32x2048xbf16>
    }

    // Negative: reshape changes the last (normalization) dim, must NOT commute.
    func.func @test_distributed_rms_norm_last_dim_changes(%arg0: tensor<1x32x64xbf16>, %arg1: tensor<2048xbf16>) -> tensor<1x32x64xbf16> {
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.distributed_rms_norm"
        // CHECK: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 2048 : i32]}> : (tensor<1x32x64xbf16>) -> tensor<1x2048xbf16>
        %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{cluster_axis = 0 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x2048xbf16>, tensor<2048xbf16>) -> tensor<1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 64 : i32]}> : (tensor<1x2048xbf16>) -> tensor<1x32x64xbf16>
        return %2 : tensor<1x32x64xbf16>
    }
}
