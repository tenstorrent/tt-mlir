// RUN: ttmlir-opt --ttir-erase-inverse-ops -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @main(%arg0: tensor<128x128x3x3xbf16>, %arg1: tensor<1x128x1x1xbf16>, %arg2: tensor<1x128x28x28xbf16>, %arg3: tensor<1x128x1x1xbf16>, %arg4: tensor<1x128x1x1xbf16>, %arg5: tensor<128x128x3x3xbf16>, %arg6: tensor<128xbf16>) -> tensor<1x128x28x28xbf16> {
    // CHECK: %[[CONV0:[0-9]+]] = "ttir.conv2d"
    // CHECK: %[[MUL:[0-9]+]] = "ttir.multiply"(%[[CONV0]],
    // CHECK: %[[ADD:[0-9]+]] = "ttir.add"(%[[MUL]],
    // CHECK: %[[MAXIMUM:[0-9]+]] = "ttir.maximum"(%[[ADD]],
    // CHECK: %[[CONV1:[0-9]+]] = "ttir.conv2d"(%[[MAXIMUM]],
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x128x28x28xbf16>}> : () -> tensor<1x128x28x28xbf16>
    %1 = tensor.empty() : tensor<1x128x28x28xbf16>
    %2 = tensor.empty() : tensor<1x28x28x128xbf16>
    %3 = "ttir.permute"(%arg2, %2) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %4 = tensor.empty() : tensor<1x1x784x128xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 1 : i32, 784 : i32, 128 : i32]}> : (tensor<1x28x28x128xbf16>, tensor<1x1x784x128xbf16>) -> tensor<1x1x784x128xbf16>
    %6 = tensor.empty() : tensor<1x1x784x128xbf16>
    %7 = "ttir.conv2d"(%5, %arg0, %6) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 28, input_width = 28,>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> : (tensor<1x1x784x128xbf16>, tensor<128x128x3x3xbf16>, tensor<1x1x784x128xbf16>) -> tensor<1x1x784x128xbf16>
    %8 = tensor.empty() : tensor<1x28x28x128xbf16>
    %9 = "ttir.reshape"(%7, %8) <{shape = [1 : i32, 28 : i32, 28 : i32, 128 : i32]}> : (tensor<1x1x784x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %10 = "ttir.permute"(%9, %1) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x28x28x128xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %11 = tensor.empty() : tensor<1x128x28x28xbf16>
    %12 = "ttir.broadcast"(%arg3, %11) <{broadcast_dimensions = array<i64: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %13 = tensor.empty() : tensor<1x128x28x28xbf16>
    %14 = "ttir.multiply"(%10, %12, %13) : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %15 = tensor.empty() : tensor<1x128x28x28xbf16>
    %16 = "ttir.broadcast"(%arg4, %15) <{broadcast_dimensions = array<i64: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %17 = tensor.empty() : tensor<1x128x28x28xbf16>
    %18 = "ttir.add"(%14, %16, %17) : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %19 = tensor.empty() : tensor<1x128x28x28xbf16>
    %20 = "ttir.maximum"(%18, %0, %19) : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %21 = tensor.empty() : tensor<1x128x28x28xbf16>
    %22 = tensor.empty() : tensor<1x28x28x128xbf16>
    %23 = "ttir.permute"(%20, %22) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x128x28x28xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %24 = tensor.empty() : tensor<1x1x784x128xbf16>
    %25 = "ttir.reshape"(%23, %24) <{shape = [1 : i32, 1 : i32, 784 : i32, 128 : i32]}> : (tensor<1x28x28x128xbf16>, tensor<1x1x784x128xbf16>) -> tensor<1x1x784x128xbf16>
    %26 = tensor.empty() : tensor<1x1x784x128xbf16>
    %27 = "ttir.conv2d"(%25, %arg5, %26) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 28, input_width = 28,>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> : (tensor<1x1x784x128xbf16>, tensor<128x128x3x3xbf16>, tensor<1x1x784x128xbf16>) -> tensor<1x1x784x128xbf16>
    %28 = tensor.empty() : tensor<1x28x28x128xbf16>
    %29 = "ttir.reshape"(%27, %28) <{shape = [1 : i32, 28 : i32, 28 : i32, 128 : i32]}> : (tensor<1x1x784x128xbf16>, tensor<1x28x28x128xbf16>) -> tensor<1x28x28x128xbf16>
    %30 = "ttir.permute"(%29, %21) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x28x28x128xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %31 = tensor.empty() : tensor<1x128x1x1xbf16>
    %32 = "ttir.reshape"(%arg6, %31) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %33 = tensor.empty() : tensor<1x128x28x28xbf16>
    %34 = "ttir.broadcast"(%32, %33) <{broadcast_dimensions = array<i64: 1, 1, 28, 28>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    %35 = tensor.empty() : tensor<1x128x28x28xbf16>
    %36 = "ttir.add"(%30, %34, %35) : (tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>, tensor<1x128x28x28xbf16>) -> tensor<1x128x28x28xbf16>
    return %36 : tensor<1x128x28x28xbf16>
  }
}
