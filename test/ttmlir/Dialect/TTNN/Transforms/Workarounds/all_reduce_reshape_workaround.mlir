// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for the all_reduce sub-4D tensor reshape workaround.

// Verify that all_reduce on tensors with rank < 4 is reshaped up to 4D, run as
// a native all_reduce, then reshaped back to the original rank. all_reduce
// produces incorrect results for tensors with rank < 4, hence the workaround.

// -----

// 1D input is padded with three leading unit dimensions.
module attributes {} {
  // CHECK-LABEL: all_reduce_reshape_workaround_1d
  func.func @all_reduce_reshape_workaround_1d(%arg0: tensor<128xbf16>) -> tensor<128xbf16> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128xbf16>) -> tensor<128xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]
    // CHECK: "ttnn.all_reduce"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [128 : i32]
    return %0 : tensor<128xbf16>
  }
}

// -----

// 2D input is padded with two leading unit dimensions.
module attributes {} {
  // CHECK-LABEL: all_reduce_reshape_workaround_2d
  func.func @all_reduce_reshape_workaround_2d(%arg0: tensor<32x256xbf16>) -> tensor<32x256xbf16> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x256xbf16>) -> tensor<32x256xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 32 : i32, 256 : i32]
    // CHECK: "ttnn.all_reduce"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [32 : i32, 256 : i32]
    return %0 : tensor<32x256xbf16>
  }
}

// -----

// 3D input is padded with one leading unit dimension.
module attributes {} {
  // CHECK-LABEL: all_reduce_reshape_workaround_3d
  func.func @all_reduce_reshape_workaround_3d(%arg0: tensor<1x32x256xbf16>) -> tensor<1x32x256xbf16> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x32x256xbf16>) -> tensor<1x32x256xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 32 : i32, 256 : i32]
    // CHECK: "ttnn.all_reduce"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 32 : i32, 256 : i32]
    return %0 : tensor<1x32x256xbf16>
  }
}

// -----

// 4D input already has rank >= 4, so no reshape is inserted.
module attributes {} {
  // CHECK-LABEL: all_reduce_reshape_workaround_4d
  func.func @all_reduce_reshape_workaround_4d(%arg0: tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    // CHECK-NOT: "ttnn.reshape"
    // CHECK: "ttnn.all_reduce"
    // CHECK-NOT: "ttnn.reshape"
    return %0 : tensor<1x1x4096x16384xf32>
  }
}
