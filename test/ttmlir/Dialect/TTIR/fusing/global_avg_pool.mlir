// RUN: ttmlir-opt --ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_efficientnet_avg_pool(%input: tensor<1x32x112x112xbf16>) -> tensor<1x32x1x1xbf16> {
    %10 = "ttir.constant"() <{value = dense<7.97193861E-5> : tensor<1x32xbf16>}> : () -> tensor<1x32xbf16>
    %0 = ttir.empty() : tensor<1x32xbf16>
    %1 = "ttir.sum"(%input, %0) <{dim_arg = [3 : i32, 2 : i32], keep_dim = false}> : (tensor<1x32x112x112xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %2 = ttir.empty() : tensor<1x32xbf16>
    %3 = "ttir.multiply"(%1, %10, %2) : (tensor<1x32xbf16>, tensor<1x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %4 = ttir.empty() : tensor<1x32x1x1xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32xbf16>, tensor<1x32x1x1xbf16>) -> tensor<1x32x1x1xbf16>
    return %5 : tensor<1x32x1x1xbf16>
}

