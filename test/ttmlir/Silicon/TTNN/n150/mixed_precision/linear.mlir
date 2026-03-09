// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="experimental-weight-dtype=bfp_bf8 enable-optimizer=true system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @linear(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %bias: tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32> {
    // CHECK-LABEL: func.func @linear
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<2x34x1024xf32>, tensor<1024x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    return %1 : tensor<2x34x1024xf32>
  }
  func.func @linear_with_implicit_broadcast(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %bias: tensor<1024xf32>) -> tensor<2x34x1024xf32> {
    // CHECK-LABEL: func.func @linear_with_implicit_broadcast
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.linear"(%arg0, %[[BFP8_WEIGHT]], %arg2)
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<2x34x1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>) -> tensor<2x34x1024xf32>
    return %1 : tensor<2x34x1024xf32>
  }

  func.func @linear_with_implicit_broadcast_2(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %bias: tensor<2x2x34x1024xf32>) -> tensor<2x2x34x1024xf32> {
    // CHECK-LABEL: func.func @linear_with_implicit_broadcast_2
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<2x34x1024xf32>, tensor<1024x1024xf32>, tensor<2x2x34x1024xf32>) -> tensor<2x2x34x1024xf32>
    return %1 : tensor<2x2x34x1024xf32>
  }

  func.func @linear_with_batched_rhs_and_bias(%arg0: tensor<2x33x1024xf32>, %arg1: tensor<2x1024x1024xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %bias: tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32> {
    // CHECK-LABEL: func.func @linear_with_batched_rhs_and_bias
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    // this will be lowered to a matmul + add
    %1 = "ttir.linear"(%arg0, %arg1, %bias) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<2x33x1024xf32>, tensor<2x1024x1024xf32>, tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>
    return %1 : tensor<2x33x1024xf32>
  }

  func.func @linear_bias_broadcast(%arg0: tensor<4x3x64x128xbf16>, %arg1: tensor<4x3x128x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    // CHECK-LABEL: func.func @linear_bias_broadcast
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    // this will be lowered to a matmul + add
    // Bias broadcasts from [14, 4, 3, 64, 32] (adds leading batch dim 14)
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<4x3x64x128xbf16>, tensor<4x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %1 : tensor<14x4x3x64x32xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast(%arg0: tensor<1x1x64x128xbf16>, %arg1: tensor<1x1x128x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %bias: tensor<4x3x64x32xbf16>) -> tensor<4x3x64x32xbf16> {
    // CHECK-LABEL: func.func @linear_nd_nd_bias_broadcast
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    // The expected output shape is [1, 1, 64, 32] with leading batch dims broadcasted to [4, 3, 64, 32] due to bias.
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<1x1x64x128xbf16>, tensor<1x1x128x32xbf16>, tensor<4x3x64x32xbf16>) -> tensor<4x3x64x32xbf16>
    return %1 : tensor<4x3x64x32xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast_matmul(%arg0: tensor<1x3x64x128xbf16>, %arg1: tensor<1x3x128x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    // CHECK-LABEL: func.func @linear_nd_nd_bias_broadcast_matmul
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<1x3x64x128xbf16>, tensor<1x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %1 : tensor<14x4x3x64x32xbf16>
  }

  func.func @linear_no_parameter(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32> {
    // CHECK-LABEL: func.func @linear_no_parameter
    // CHECK-NOT: ttcore.load_cached
    // CHECK-NOT: bfp_bf8
    // CHECK: "ttnn.matmul"(%arg0, %arg1)
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<2x34x1024xf32>, tensor<1024x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    return %1 : tensor<2x34x1024xf32>
  }
}
