// REQUIRES: stablehlo
// RUN: ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Unsigned (ui32) reduce_window init value. checkInitValue() used to iterate the
// init constant with value_begin<int32_t>(), which asserts on an unsigned
// ElementsAttr ("does not provide iteration facilities for type `int`") and aborts
// the compiler. torch 2.11's max_pool2d_with_indices lowering emits ui32 tensors,
// so this path is exercised by e.g. efficientdet. See tenstorrent/tt-mlir#9031.
// The init here is the ui32 bit pattern of INT32_MIN (2147483648 == 0x80000000),
// the -inf reduction sentinel for a max reduce.
func.func public @test_maxpool2d_ui32(%arg0: tensor<1x128x128x32xui32>) -> tensor<1x64x64x32xui32> {
  %0 = stablehlo.constant dense<2147483648> : tensor<ui32>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%arg0)
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<1x128x128x32xui32>) -> tensor<1x64x64x32xui32>
  %2 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg2: tensor<ui32>, %arg3: tensor<ui32>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<ui32>
    stablehlo.return %3 : tensor<ui32>
  }) : (tensor<1x128x128x32xui32>, tensor<ui32>) -> tensor<1x64x64x32xui32>
  return %2 : tensor<1x64x64x32xui32>
}
