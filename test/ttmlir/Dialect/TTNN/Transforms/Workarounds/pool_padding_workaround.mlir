// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize %s | FileCheck %s

module {
  func.func public @test_uneven_pool_padding_workaround(%arg0: tensor<1x56x56x128xf32>) -> tensor<1x28x28x128xf32> {
    // CHECK: %[[INPUT_RESHAPE:[0-9]+]] = "ttnn.reshape"(%0
    // CHECK: %[[PAD:[0-9]+]] = "ttnn.pad"(%[[INPUT_RESHAPE]]
    // CHECK: %[[OUTPUT_RESHAPE:[0-9]+]] = "ttnn.reshape"(%[[PAD]]
    // CHECK: %[[TOLAYOUT:[0-9]+]] = "ttnn.to_layout"(%[[OUTPUT_RESHAPE]]
    // CHECK: %[[MAX_POOL:[0-9]+]] = "ttnn.max_pool2d"(%[[TOLAYOUT]]
    %1 = ttir.empty() : tensor<1x1x3136x128xf32>
    %2 = "ttir.reshape"(%arg0, %1) <{shape = [1 : i32, 1 : i32, 3136 : i32, 128 : i32]}> : (tensor<1x56x56x128xf32>, tensor<1x1x3136x128xf32>) -> tensor<1x1x3136x128xf32>
    %3 = ttir.empty() : tensor<1x1x784x128xf32>
    %4 = "ttir.max_pool2d"(%2, %3) <{ceil_mode = false, dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 56, input_width = 56>, kernel = array<i32: 3, 3>, padding = array<i32: 0, 0, 1, 1>, stride = array<i32: 2, 2>}> : (tensor<1x1x3136x128xf32>, tensor<1x1x784x128xf32>) -> tensor<1x1x784x128xf32>
    %5 = ttir.empty() : tensor<1x28x28x128xf32>
    %6 = "ttir.reshape"(%4, %5) <{shape = [1 : i32, 28 : i32, 28 : i32, 128 : i32]}> : (tensor<1x1x784x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    return %6 : tensor<1x28x28x128xf32>
  }
}
