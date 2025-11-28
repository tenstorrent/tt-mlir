// RUN: ttmlir-opt --canonicalize --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @scaled_sum_to_mean(%input: tensor<1x32x112x112xbf16>) -> tensor<1x32x1x1xbf16> {
    // CHECK-LABEL: func.func @scaled_sum_to_mean
    %0 = "ttir.constant"() <{value = dense<7.97193861E-5> : tensor<1x32xbf16>}> : () -> tensor<1x32xbf16>
    // CHECK-NOT: "ttir.sum"
    %2 = "ttir.sum"(%input) <{dim_arg = [3 : i32, 2 : i32], keep_dim = false}> : (tensor<1x32x112x112xbf16>) -> tensor<1x32xbf16>
    // CHECK-NOT: "ttir.multiply"
    %4 = "ttir.multiply"(%2, %0) : (tensor<1x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    // CHECK: %[[MEAN:.*]] = "ttir.mean"(%arg0)
    // CHECK-SAME: dim_arg = [3 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-NOT: "ttir.reshape"
    %6 = "ttir.reshape"(%4) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32xbf16>) -> tensor<1x32x1x1xbf16>
    // CHECK: return %[[MEAN]]
    return %6 : tensor<1x32x1x1xbf16>
}

func.func @scaled_sum_to_mean_no_reshape(%input: tensor<1x32x112x112xbf16>) -> tensor<1x32xbf16> {
    // CHECK-LABEL: func.func @scaled_sum_to_mean_no_reshape
    %0 = "ttir.constant"() <{value = dense<7.97193861E-5> : tensor<1x32xbf16>}> : () -> tensor<1x32xbf16>
    // CHECK-NOT: "ttir.sum"
    %2 = "ttir.sum"(%input) <{dim_arg = [3 : i32, 2 : i32], keep_dim = false}> : (tensor<1x32x112x112xbf16>) -> tensor<1x32xbf16>
    // CHECK-NOT: "ttir.multiply"
    %4 = "ttir.multiply"(%2, %0) : (tensor<1x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    // CHECK: %[[MEAN:.*]] = "ttir.mean"(%arg0)
    // CHECK-SAME: dim_arg = [3 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK: return %[[MEAN]]
    return %4 : tensor<1x32xbf16>
}

func.func @scaled_sum_to_mean_keep_dim(%input: tensor<1x32x112x112xbf16>) -> tensor<1x32x1x1xbf16> {
    // CHECK-LABEL: func.func @scaled_sum_to_mean_keep_dim
    %0 = "ttir.constant"() <{value = dense<7.97193861E-5> : tensor<1x32x1x1xbf16>}> : () -> tensor<1x32x1x1xbf16>
    // CHECK-NOT: "ttir.sum"
    %2 = "ttir.sum"(%input) <{dim_arg = [3 : i32, 2 : i32], keep_dim = true}> : (tensor<1x32x112x112xbf16>) -> tensor<1x32x1x1xbf16>
    // CHECK-NOT: "ttir.multiply"
    %4 = "ttir.multiply"(%2, %0) : (tensor<1x32x1x1xbf16>, tensor<1x32x1x1xbf16>) -> tensor<1x32x1x1xbf16>
    // CHECK: %[[MEAN:.*]] = "ttir.mean"(%arg0)
    // CHECK-SAME: dim_arg = [3 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK: return %[[MEAN]]
    return %4 : tensor<1x32x1x1xbf16>
}

// Test with different dimensions [1,2] that also triggers SpatialMeanOptimizationPattern
func.func @scaled_sum_to_mean_with_spatial_optimization(%input: tensor<8x16x32x64xbf16>) -> tensor<8x64x1x1xbf16> {
    // CHECK-LABEL: func.func @scaled_sum_to_mean_with_spatial_optimization
    %0 = "ttir.constant"() <{value = dense<1.953125E-3> : tensor<8x64xbf16>}> : () -> tensor<8x64xbf16>
    // CHECK-NOT: "ttir.sum"
    %2 = "ttir.sum"(%input) <{dim_arg = [1 : i32, 2 : i32], keep_dim = false}> : (tensor<8x16x32x64xbf16>) -> tensor<8x64xbf16>
    // CHECK-NOT: "ttir.multiply"
    %4 = "ttir.multiply"(%2, %0) : (tensor<8x64xbf16>, tensor<8x64xbf16>) -> tensor<8x64xbf16>
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.mean"
    // CHECK-SAME: dim_arg = [2 : i32]
    // CHECK: "ttir.reshape"
    %6 = "ttir.reshape"(%4) <{shape = [8 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<8x64xbf16>) -> tensor<8x64x1x1xbf16>
    // CHECK: return
    return %6 : tensor<8x64x1x1xbf16>
}

// Test with single dimension reduction: sum on dim [2] 
func.func @scaled_sum_to_mean_single_dim(%input: tensor<4x8x16xbf16>) -> tensor<4x8xbf16> {
    // CHECK-LABEL: func.func @scaled_sum_to_mean_single_dim
    %0 = "ttir.constant"() <{value = dense<6.25E-2> : tensor<4x8xbf16>}> : () -> tensor<4x8xbf16>
    // CHECK-NOT: "ttir.sum"
    %2 = "ttir.sum"(%input) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<4x8x16xbf16>) -> tensor<4x8xbf16>
    // CHECK-NOT: "ttir.multiply"
    %4 = "ttir.multiply"(%2, %0) : (tensor<4x8xbf16>, tensor<4x8xbf16>) -> tensor<4x8xbf16>
    // CHECK: %[[MEAN:.*]] = "ttir.mean"(%arg0)
    // CHECK-SAME: dim_arg = [2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK: return %[[MEAN]]
    return %4 : tensor<4x8xbf16>
}

func.func @scaled_sum_to_mean_three_dims(%input: tensor<2x4x8x16xbf16>) -> tensor<2x1x1x1xbf16> {
    // CHECK-LABEL: func.func @scaled_sum_to_mean_three_dims
    %0 = "ttir.constant"() <{value = dense<1.953125E-3> : tensor<2x1x1x1xbf16>}> : () -> tensor<2x1x1x1xbf16>
    // CHECK-NOT: "ttir.sum"
    %2 = "ttir.sum"(%input) <{dim_arg = [1 : i32, 2 : i32, 3 : i32], keep_dim = true}> : (tensor<2x4x8x16xbf16>) -> tensor<2x1x1x1xbf16>
    // CHECK-NOT: "ttir.multiply"
    %4 = "ttir.multiply"(%2, %0) : (tensor<2x1x1x1xbf16>, tensor<2x1x1x1xbf16>) -> tensor<2x1x1x1xbf16>
    // CHECK: %[[MEAN:.*]] = "ttir.mean"(%arg0)
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK: return %[[MEAN]]
    return %4 : tensor<2x1x1x1xbf16>
}
