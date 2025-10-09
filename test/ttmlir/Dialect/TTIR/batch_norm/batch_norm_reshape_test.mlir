// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

// Test 1: 4D input with dimension=1 (NCHW) - parameters should be reshaped to [1, C, 1, 1]
module {
  // CHECK-LABEL: func.func public @test_batch_norm_4d_dim1
  func.func public @test_batch_norm_4d_dim1(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
    %0 = ttir.empty() : tensor<2x3x4x5xf32>
    // Input: 2x3x4x5, dimension=1 means normalize over dimension with size 3
    // Parameters (mean, var, scale, offset) are 1D with size 3
    // They should be reshaped to [1, 3, 1, 1] to match NCHW format
    %1 = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32}> : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    // CHECK-DAG: "ttir.reshape"(%arg1, {{.*}}) <{shape = [1 : i32, 3 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg2, {{.*}}) <{shape = [1 : i32, 3 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg3, {{.*}}) <{shape = [1 : i32, 3 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg4, {{.*}}) <{shape = [1 : i32, 3 : i32, 1 : i32, 1 : i32]}>
    // CHECK: "ttir.batch_norm"({{.*}}) <{dimension = 1 : i32, epsilon = {{.*}}}>
    return %1 : tensor<2x3x4x5xf32>
  }

  // Test 2: 3D input with dimension=1 - should normalize to NCHW, reshape params, then denormalize
  // CHECK-LABEL: func.func public @test_batch_norm_3d_dim1
  func.func public @test_batch_norm_3d_dim1(%arg0: tensor<2x3x4xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<2x3x4xf32> {
    %0 = ttir.empty() : tensor<2x3x4xf32>
    // 3D input: [2, 3, 4], dimension=1
    // Should be normalized to 4D: [2, 3, 4, 1] (NCHW format)
    // Parameters reshaped to [1, 3, 1, 1]
    %1 = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32}> : (tensor<2x3x4xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    // CHECK: "ttir.reshape"(%arg0, {{.*}}) <{shape = [2 : i32, 3 : i32, 4 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg1, {{.*}}) <{shape = [1 : i32, 3 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg2, {{.*}}) <{shape = [1 : i32, 3 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg3, {{.*}}) <{shape = [1 : i32, 3 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg4, {{.*}}) <{shape = [1 : i32, 3 : i32, 1 : i32, 1 : i32]}>
    // CHECK: "ttir.batch_norm"({{.*}}) <{dimension = 1 : i32, epsilon = {{.*}}}>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [2 : i32, 3 : i32, 4 : i32]}>
    return %1 : tensor<2x3x4xf32>
  }

  // Test 3: 2D input with dimension=1 - edge case
  // CHECK-LABEL: func.func public @test_batch_norm_2d_dim1
  func.func public @test_batch_norm_2d_dim1(%arg0: tensor<4x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<4x8xf32> {
    %0 = ttir.empty() : tensor<4x8xf32>
    // 2D input: [4, 8], dimension=1
    // Should be normalized to 4D: [4, 8, 1, 1] (NCHW format)
    // Parameters reshaped to [1, 8, 1, 1]
    %1 = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32}> : (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    // CHECK: "ttir.reshape"(%arg0, {{.*}}) <{shape = [4 : i32, 8 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg1, {{.*}}) <{shape = [1 : i32, 8 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg2, {{.*}}) <{shape = [1 : i32, 8 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg3, {{.*}}) <{shape = [1 : i32, 8 : i32, 1 : i32, 1 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg4, {{.*}}) <{shape = [1 : i32, 8 : i32, 1 : i32, 1 : i32]}>
    // CHECK: "ttir.batch_norm"({{.*}}) <{dimension = 1 : i32, epsilon = {{.*}}}>
    // CHECK: "ttir.reshape"({{.*}}) <{shape = [4 : i32, 8 : i32]}>
    return %1 : tensor<4x8xf32>
  }
}
