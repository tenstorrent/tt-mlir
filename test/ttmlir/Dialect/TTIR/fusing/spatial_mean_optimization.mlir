// RUN: ttmlir-opt --canonicalize --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @spatial_mean_optimization(%input: tensor<12x7x7x1152xbf16>) -> tensor<12x1x1x1152xbf16> {
    // CHECK-LABEL: func.func @spatial_mean_optimization
    %0 = "ttir.mean"(%input) <{dim_arg = [1 : i32, 2 : i32], keep_dim = true}> : (tensor<12x7x7x1152xbf16>) -> tensor<12x1x1x1152xbf16>
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.mean" 
    // CHECK-SAME: dim_arg = [2 : i32]
    return %0 : tensor<12x1x1x1152xbf16>
}

func.func @spatial_mean_optimization_no_keep_dim(%input: tensor<12x7x7x1152xbf16>) -> tensor<12x1152xbf16> {
    // CHECK-LABEL: func.func @spatial_mean_optimization_no_keep_dim
    %0 = "ttir.mean"(%input) <{dim_arg = [1 : i32, 2 : i32], keep_dim = false}> : (tensor<12x7x7x1152xbf16>) -> tensor<12x1152xbf16>
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.mean" 
    // CHECK-SAME: dim_arg = [2 : i32]
    // CHECK: "ttir.reshape"
    return %0 : tensor<12x1152xbf16>
}

