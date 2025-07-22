// RUN: ttmlir-opt --ttir-explicate-tms --ttir-erase-inverse-ops -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @resnet_erase_redundant_reshapes(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x1x64xf32>, %arg2: tensor<1x1x1x64xf32>, %arg3: tensor<1x1x1x64xf32>, %arg4: tensor<64x64x1x1xf32>, %arg5: tensor<64x64x3x3xf32>) -> tensor<1x56x56x64xf32> {
    %0 = ttir.empty() : tensor<1x56x56x64xf32>
    %1 = ttir.empty() : tensor<1x1x3136x64xf32>
    %2 = "ttir.reshape"(%arg0, %1) <{shape = [1 : i32, 1 : i32, 3136 : i32, 64 : i32]}> : (tensor<1x56x56x64xf32>, tensor<1x1x3136x64xf32>) -> tensor<1x1x3136x64xf32>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: -> tensor<1x1x3136x64xf32>
    %3 = ttir.empty() : tensor<1x1x3136x64xf32>
    %4 = "ttir.conv2d"(%2, %arg4, %3) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 56, input_width = 56>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x3136x64xf32>, tensor<64x64x1x1xf32>, tensor<1x1x3136x64xf32>) -> tensor<1x1x3136x64xf32>
    %5 = ttir.empty() : tensor<1x56x56x64xf32>
    %6 = "ttir.reshape"(%4, %5) <{shape = [1 : i32, 56 : i32, 56 : i32, 64 : i32]}> : (tensor<1x1x3136x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %7 = ttir.empty() : tensor<1x56x56x64xf32>
    %8 = "ttir.multiply"(%6, %arg2, %7) : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %9 = ttir.empty() : tensor<1x56x56x64xf32>
    %10 = "ttir.add"(%8, %arg3, %9) : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %11 = ttir.empty() : tensor<1x56x56x64xf32>
    %12 = "ttir.relu"(%10, %11) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %13 = ttir.empty() : tensor<1x56x56x64xf32>
    %14 = ttir.empty() : tensor<1x1x3136x64xf32>
    %15 = "ttir.reshape"(%12, %14) <{shape = [1 : i32, 1 : i32, 3136 : i32, 64 : i32]}> : (tensor<1x56x56x64xf32>, tensor<1x1x3136x64xf32>) -> tensor<1x1x3136x64xf32>
    %16 = ttir.empty() : tensor<1x1x3136x64xf32>
    %17 = "ttir.conv2d"(%15, %arg5, %16) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 56, input_width = 56>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> : (tensor<1x1x3136x64xf32>, tensor<64x64x3x3xf32>, tensor<1x1x3136x64xf32>) -> tensor<1x1x3136x64xf32>
    %18 = ttir.empty() : tensor<1x56x56x64xf32>
    %19 = "ttir.reshape"(%17, %18) <{shape = [1 : i32, 56 : i32, 56 : i32, 64 : i32]}> : (tensor<1x1x3136x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: -> tensor<1x56x56x64xf32>
    // CHECK-NOT: "ttir.reshape"
    return %19 : tensor<1x56x56x64xf32>
  }
}
