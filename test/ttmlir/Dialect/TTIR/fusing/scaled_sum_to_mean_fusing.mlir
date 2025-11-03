// RUN: ttmlir-opt --canonicalize --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @scaled_sum_to_mean(%input: tensor<1x32x112x112xbf16>) -> tensor<1x32x1x1xbf16> {
    %0 = "ttir.constant"() <{value = dense<7.97193861E-5> : tensor<1x32xbf16>}> : () -> tensor<1x32xbf16>
    %1 = ttir.empty() : tensor<1x32xbf16>
    // CHECK-NOT: "ttir.sum"
    %2 = "ttir.sum"(%input, %1) <{dim_arg = [3 : i32, 2 : i32], keep_dim = false}> : (tensor<1x32x112x112xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %3 = ttir.empty() : tensor<1x32xbf16>
    // CHECK-NOT: "ttir.multiply"
    %4 = "ttir.multiply"(%2, %0, %3) : (tensor<1x32xbf16>, tensor<1x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %5 = ttir.empty() : tensor<1x32x1x1xbf16>
    // CHECK: "ttir.mean"
    // CHECK-NOT: "ttir.reshape"
    %6 = "ttir.reshape"(%4, %5) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32xbf16>, tensor<1x32x1x1xbf16>) -> tensor<1x32x1x1xbf16>
    return %6 : tensor<1x32x1x1xbf16>
}

func.func @scaled_sum_to_mean_no_reshape(%input: tensor<1x32x112x112xbf16>) -> tensor<1x32xbf16> {
    %0 = "ttir.constant"() <{value = dense<7.97193861E-5> : tensor<1x32xbf16>}> : () -> tensor<1x32xbf16>
    %1 = ttir.empty() : tensor<1x32xbf16>
    // CHECK-NOT: "ttir.sum"
    %2 = "ttir.sum"(%input, %1) <{dim_arg = [3 : i32, 2 : i32], keep_dim = false}> : (tensor<1x32x112x112xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %3 = ttir.empty() : tensor<1x32xbf16>
    // CHECK-NOT: "ttir.multiply"
    %4 = "ttir.multiply"(%2, %0, %3) : (tensor<1x32xbf16>, tensor<1x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    // CHECK: "ttir.mean"
    return %4 : tensor<1x32xbf16>
}

func.func @scaled_sum_to_mean_keep_dim(%input: tensor<1x32x112x112xbf16>) -> tensor<1x32x1x1xbf16> {
    %0 = "ttir.constant"() <{value = dense<7.97193861E-5> : tensor<1x32x1x1xbf16>}> : () -> tensor<1x32x1x1xbf16>
    %1 = ttir.empty() : tensor<1x32x1x1xbf16>
    // CHECK-NOT: "ttir.sum"
    %2 = "ttir.sum"(%input, %1) <{dim_arg = [3 : i32, 2 : i32], keep_dim = true}> : (tensor<1x32x112x112xbf16>, tensor<1x32x1x1xbf16>) -> tensor<1x32x1x1xbf16>
    %3 = ttir.empty() : tensor<1x32x1x1xbf16>
    // CHECK-NOT: "ttir.multiply"
    %4 = "ttir.multiply"(%2, %0, %3) : (tensor<1x32x1x1xbf16>, tensor<1x32x1x1xbf16>, tensor<1x32x1x1xbf16>) -> tensor<1x32x1x1xbf16>
    // CHECK: "ttir.mean"
    return %4 : tensor<1x32x1x1xbf16>
}
