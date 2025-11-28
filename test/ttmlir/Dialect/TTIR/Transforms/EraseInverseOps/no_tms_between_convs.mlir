// RUN: ttmlir-opt --ttir-erase-inverse-ops -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @main(%arg0: tensor<128x128x3x3xbf16>, %arg1: tensor<1x128x1x1xbf16>, %arg2: tensor<1x128x28x28xbf16>, %arg3: tensor<1x128x1x1xbf16>, %arg4: tensor<1x128x1x1xbf16>, %arg5: tensor<128x128x3x3xbf16>, %arg6: tensor<128xbf16>) -> tensor<1x128x28x28xbf16> {
    // CHECK: %[[CONV0:[0-9]+]] = "ttir.conv2d"
    // CHECK: %[[MUL:[0-9]+]] = "ttir.multiply"(%[[CONV0]],
    // CHECK: %[[ADD:[0-9]+]] = "ttir.add"(%[[MUL]],
    // CHECK: %[[MAXIMUM:[0-9]+]] = "ttir.maximum"(%[[ADD]]
    // CHECK: %[[CONV1:[0-9]+]] = "ttir.conv2d"(%[[MAXIMUM]],
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x128x28x28xbf16>}> : () -> tensor<1x128x28x28xbf16>
    %3 = "ttir.permute"(%arg2) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x128x28x28xbf16>) -> tensor<1x28x28x128xbf16>
    %5 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 784 : i32, 128 : i32]}> : (tensor<1x28x28x128xbf16>) -> tensor<1x1x784x128xbf16>
    %7 = "ttir.conv2d"(%5, %arg0) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 28, input_width = 28,>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> : (tensor<1x1x784x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x1x784x128xbf16>
    %9 = "ttir.reshape"(%7) <{shape = [1 : i32, 28 : i32, 28 : i32, 128 : i32]}> : (tensor<1x1x784x128xbf16>) -> tensor<1x28x28x128xbf16>
    %10 = "ttir.permute"(%9) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x28x28x128xbf16>) -> tensor<1x128x28x28xbf16>
    %12 = "ttir.broadcast"(%arg3) <{broadcast_dimensions = array<i64: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %14 = "ttir.multiply"(%10, %12) : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %16 = "ttir.broadcast"(%arg4) <{broadcast_dimensions = array<i64: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %18 = "ttir.add"(%14, %16) : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %20 = "ttir.maximum"(%18, %0) : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %23 = "ttir.permute"(%20) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x128x28x28xbf16>) -> tensor<1x28x28x128xbf16>
    %25 = "ttir.reshape"(%23) <{shape = [1 : i32, 1 : i32, 784 : i32, 128 : i32]}> : (tensor<1x28x28x128xbf16>) -> tensor<1x1x784x128xbf16>
    %27 = "ttir.conv2d"(%25, %arg5) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 28, input_width = 28,>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> : (tensor<1x1x784x128xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x1x784x128xbf16>
    %29 = "ttir.reshape"(%27) <{shape = [1 : i32, 28 : i32, 28 : i32, 128 : i32]}> : (tensor<1x1x784x128xbf16>) -> tensor<1x28x28x128xbf16>
    %30 = "ttir.permute"(%29) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x28x28x128xbf16>) -> tensor<1x128x28x28xbf16>
    %32 = "ttir.reshape"(%arg6) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128xbf16>) -> tensor<1x128x1x1xbf16>
    %34 = "ttir.broadcast"(%32) <{broadcast_dimensions = array<i64: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>) -> tensor<1x128x28x28xbf16>
    %36 = "ttir.add"(%30, %34) : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    return %36 : tensor<1x128x28x28xbf16>
  }
}
