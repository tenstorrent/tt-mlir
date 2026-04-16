// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-upwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    // Basic case: the input reshape is pushed below distributed_rms_norm, so
    // distributed_rms_norm runs at the un-reshaped (smaller-rank) shape and an
    // output reshape is added afterward.
    func.func @test_reshape_distributed_rms_norm_commute_down(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<32x1x2048xbf16> {
        // CHECK-LABEL: @test_reshape_distributed_rms_norm_commute_down
        // CHECK: "ttir.distributed_rms_norm"(%arg0, %arg1)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK: "ttir.reshape"
        // CHECK-SAME: -> tensor<32x1x2048xbf16>
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        return %1 : tensor<32x1x2048xbf16>
    }

    // Case with residual: commute down adds an inverse reshape on the residual
    // so it continues to match the input shape fed to distributed_rms_norm.
    func.func @test_reshape_distributed_rms_norm_commute_down_with_residual(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>, %arg2: tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16> {
        // CHECK-LABEL: @test_reshape_distributed_rms_norm_commute_down_with_residual
        // CHECK: "ttir.reshape"(%arg2)
        // CHECK-SAME: -> tensor<32x2048xbf16>
        // CHECK: "ttir.distributed_rms_norm"(%arg0, %arg1, %{{.+}})
        // CHECK-SAME: -> tensor<32x2048xbf16>
        // CHECK: "ttir.reshape"
        // CHECK-SAME: -> tensor<32x1x2048xbf16>
        %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %1 = "ttir.distributed_rms_norm"(%0, %arg1, %arg2) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<32x1x2048xbf16>, tensor<2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        return %1 : tensor<32x1x2048xbf16>
    }

    // Negative case: reshape changes the last dimension. Should NOT commute down
    // because the normalization dim would change.
    func.func @test_reshape_distributed_rms_norm_down_last_dim_changes(%arg0: tensor<1x32x64xbf16>, %arg1: tensor<2048xbf16>) -> tensor<1x2048xbf16> {
        // CHECK-LABEL: @test_reshape_distributed_rms_norm_down_last_dim_changes
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.distributed_rms_norm"
        %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 2048 : i32]}> : (tensor<1x32x64xbf16>) -> tensor<1x2048xbf16>
        %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x2048xbf16>, tensor<2048xbf16>) -> tensor<1x2048xbf16>
        return %1 : tensor<1x2048xbf16>
    }
}
