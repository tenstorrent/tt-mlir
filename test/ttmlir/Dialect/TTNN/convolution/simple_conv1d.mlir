// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @conv1d_test1(%arg0: tensor<1x256x512xf32>, %arg1: tensor<1024x256x1xf32>, %arg2: tensor<1024xf32>) -> tensor<1x1024x512xf32> {
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 256 : i32, 512 : i32, 1 : i32]
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1024 : i32, 256 : i32, 1 : i32, 1 : i32]
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK: "ttnn.conv2d"
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1024 : i32, 512 : i32]
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 256 : i32, 512 : i32, 1 : i32]}> : (tensor<1x256x512xf32>) -> tensor<1x256x512x1xf32>
    %1 = "ttir.reshape"(%arg1) <{shape = [1024 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x256x1xf32>) -> tensor<1024x256x1x1xf32>
    %2 = "ttir.permute"(%0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x256x512x1xf32>) -> tensor<1x512x1x256xf32>
    %3 = "ttir.permute"(%1) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %4 = "ttir.conv2d"(%2, %3) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x512x1x256xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x512x1x1024xf32>
    %5 = "ttir.permute"(%4) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x512x1x1024xf32>) -> tensor<1x1024x512x1xf32>
    %6 = "ttir.reshape"(%5) <{shape = [1 : i32, 1024 : i32, 512 : i32]}> : (tensor<1x1024x512x1xf32>) -> tensor<1x1024x512xf32>
    return %6 : tensor<1x1024x512xf32>
    // CHECK: return %{{.*}} : tensor<1x1024x512xf32, #ttnn_layout3>
  }

  // Test a different ordering of dimensions
  func.func public @conv1d_test2(%arg0: tensor<1x7x768xbf16>, %arg1: tensor<1x192x768xbf16>) -> (tensor<1x7x768xbf16>) {
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 7 : i32, 768 : i32, 1 : i32]
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 192 : i32, 768 : i32, 1 : i32]
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 1, 3, 2>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 2, 1, 0, 3>
    // CHECK: "ttnn.conv2d"
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 1, 3, 2>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 7 : i32, 768 : i32]
        %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 7 : i32, 768 : i32, 1 : i32]}> : (tensor<1x7x768xbf16>) -> tensor<1x7x768x1xbf16>
    %1 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 192 : i32, 768 : i32, 1 : i32]}> : (tensor<1x192x768xbf16>) -> tensor<1x192x768x1xbf16>
    %2 = "ttir.permute"(%0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x7x768x1xbf16>) -> tensor<1x7x1x768xbf16>
    %3 = "ttir.permute"(%1) <{permutation = array<i64: 2, 1, 0, 3>}> : (tensor<1x192x768x1xbf16>) -> tensor<768x192x1x1xbf16>
    %4 = "ttir.conv2d"(%2, %3) <{dilation = array<i32: 1, 1>, groups = 4 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x7x1x768xbf16>, tensor<768x192x1x1xbf16>) -> tensor<1x7x1x768xbf16>
    %5 = "ttir.permute"(%4) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x7x1x768xbf16>) -> tensor<1x7x768x1xbf16>
    %6 = "ttir.reshape"(%5) <{shape = [1 : i32, 7 : i32, 768 : i32]}> : (tensor<1x7x768x1xbf16>) -> tensor<1x7x768xbf16>
    return %6 : tensor<1x7x768xbf16>
    // CHECK: return %{{.*}} : tensor<1x7x768xbf16, #ttnn_layout{{.*}}>
  }
}
