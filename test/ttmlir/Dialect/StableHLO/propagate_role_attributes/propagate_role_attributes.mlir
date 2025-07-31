// REQUIRES: stablehlo
// RUN: ttmlir-opt --propagate-role-attributes %s | FileCheck %s

// CHECK-LABEL: module @jit__lambda_
module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, tt.mark_function_defined_10_f32 = true, tt.mark_function_defined_128_f32 = true, tt.mark_function_defined_128x10_f32 = true, tt.mark_function_defined_784x128_f32 = true} {
  func.func private @tt.mark_128x10_f32(%arg0: tensor<128x10xf32>) -> tensor<128x10xf32> {
    return %arg0 : tensor<128x10xf32>
  }
  func.func private @tt.mark_10_f32(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    return %arg0 : tensor<10xf32>
  }
  func.func private @tt.mark_784x128_f32(%arg0: tensor<784x128xf32>) -> tensor<784x128xf32> {
    return %arg0 : tensor<784x128xf32>
  }
  func.func private @tt.mark_128_f32(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }
  // CHECK: func.func public @main(%arg0: tensor<128xf32> {mhlo.sharding = "{replicated}", tt.role = "weight"}, %arg1: tensor<784x128xf32> {mhlo.sharding = "{replicated}", tt.role = "weight"}, %arg2: tensor<10xf32> {mhlo.sharding = "{replicated}", tt.role = "weight"}, %arg3: tensor<128x10xf32> {mhlo.sharding = "{replicated}", tt.role = "weight"}, %arg4: tensor<32x28x28x1xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<32x10xf32> {jax.result_info = "result"}) {
  func.func public @main(%arg0: tensor<128xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<784x128xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<10xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<128x10xf32> {mhlo.sharding = "{replicated}"}, %arg4: tensor<32x28x28x1xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<32x10xf32> {jax.result_info = "result"}) {
    %0 = call @"<lambda>"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<128xf32>, tensor<784x128xf32>, tensor<10xf32>, tensor<128x10xf32>, tensor<32x28x28x1xf32>) -> tensor<32x10xf32>
    return %0 : tensor<32x10xf32>
  }
  // CHECK: func.func private @"<lambda>"(%arg0: tensor<128xf32> {tt.role = "weight"}, %arg1: tensor<784x128xf32> {tt.role = "weight"}, %arg2: tensor<10xf32> {tt.role = "weight"}, %arg3: tensor<128x10xf32> {tt.role = "weight"}, %arg4: tensor<32x28x28x1xf32>) -> tensor<32x10xf32> {
  func.func private @"<lambda>"(%arg0: tensor<128xf32>, %arg1: tensor<784x128xf32>, %arg2: tensor<10xf32>, %arg3: tensor<128x10xf32>, %arg4: tensor<32x28x28x1xf32>) -> tensor<32x10xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = call @tt.mark_128_f32(%arg0) {tt.role = "weight"} : (tensor<128xf32>) -> tensor<128xf32>
    %1 = call @tt.mark_784x128_f32(%arg1) {tt.role = "weight"} : (tensor<784x128xf32>) -> tensor<784x128xf32>
    %2 = call @tt.mark_10_f32(%arg2) {tt.role = "weight"} : (tensor<10xf32>) -> tensor<10xf32>
    %3 = call @tt.mark_128x10_f32(%arg3) {tt.role = "weight"} : (tensor<128x10xf32>) -> tensor<128x10xf32>
    %4 = stablehlo.reshape %arg4 : (tensor<32x28x28x1xf32>) -> tensor<32x784xf32>
    %5 = stablehlo.dot_general %4, %1, contracting_dims = [1] x [0] : (tensor<32x784xf32>, tensor<784x128xf32>) -> tensor<32x128xf32>
    %6 = stablehlo.reshape %0 : (tensor<128xf32>) -> tensor<1x128xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<32x128xf32>
    %8 = stablehlo.add %5, %7 : tensor<32x128xf32>
    %9 = call @relu(%8) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %10 = stablehlo.dot_general %9, %3, contracting_dims = [1] x [0] : (tensor<32x128xf32>, tensor<128x10xf32>) -> tensor<32x10xf32>
    %11 = stablehlo.reshape %2 : (tensor<10xf32>) -> tensor<1x10xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x10xf32>) -> tensor<32x10xf32>
    %13 = stablehlo.add %10, %12 : tensor<32x10xf32>
    %14 = stablehlo.reduce(%13 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %16 = stablehlo.maximum %15, %14 : tensor<32xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x10xf32>
    %19 = stablehlo.subtract %13, %18 : tensor<32x10xf32>
    %20 = stablehlo.exponential %19 : tensor<32x10xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x10xf32>
    %24 = stablehlo.divide %20, %23 : tensor<32x10xf32>
    return %24 : tensor<32x10xf32>
  }
  func.func private @relu(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }
}

// -----

// Test with simpler case
// CHECK-LABEL: module @simple_case
module @simple_case {
  func.func private @tt.mark_256_f32(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    return %arg0 : tensor<256xf32>
  }
  // CHECK: func.func private @lambda(%arg0: tensor<256xf32> {tt.role = "weight"}) -> tensor<256xf32> {
  func.func private @lambda(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %0 = call @tt.mark_256_f32(%arg0) {tt.role = "weight"} : (tensor<256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  // CHECK: func.func public @main(%arg0: tensor<256xf32> {tt.role = "weight"}) -> tensor<256xf32> {
  func.func public @main(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %0 = call @lambda(%arg0) : (tensor<256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
}

// -----

// Test with multiple functions and mixed roles
// CHECK-LABEL: module @mixed_roles
module @mixed_roles {
  func.func private @tt.mark_tensor(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    return %arg0 : tensor<64xf32>
  }
  func.func private @process_weights(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    // CHECK: func.func private @process_weights(%arg0: tensor<64xf32> {tt.role = "weight"}, %arg1: tensor<64xf32>) -> tensor<64xf32>
    %0 = call @tt.mark_tensor(%arg0) {tt.role = "weight"} : (tensor<64xf32>) -> tensor<64xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  // CHECK: func.func public @main(%arg0: tensor<64xf32> {tt.role = "weight"}, %arg1: tensor<64xf32>) -> tensor<64xf32> {
  func.func public @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = call @process_weights(%arg0, %arg1) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}
