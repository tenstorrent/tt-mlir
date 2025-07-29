// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func public @test_avgpool2d_workaround(%arg0: tensor<8x256x6x6xf32>) -> tensor<8x256x6x6xf32> {
  %0 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  %1 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %2 = ttir.empty() : tensor<8x256x6x6xf32>
  // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0
  // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
  // CHECK-SAME: (tensor<8x256x6x6xf32>, tensor<8x6x6x256xf32>)
  // CHECK-SAME: -> tensor<8x6x6x256xf32>
  // CHECK: %[[AVGPOOL:[0-9]+]] = "ttir.avg_pool2d"(%[[PERMUTE]],
  // CHECK-SAME: ceil_mode = false,
<<<<<<< HEAD
  // CHECK-SAME: dilation = array<i32: 1, 1>,
  // CHECK-SAME: kernel = array<i32: 1, 1>,
  // CHECK-SAME: padding = array<i32: 0, 0, 0, 0>,
  // CHECK-SAME: stride = array<i32: 1, 1>
=======
  // CHECK-SAME: dilation_height = 1 : si32, dilation_width = 1 : si32,
  // CHECK-SAME: kernel_height = 1 : si32, kernel_width = 1 : si32,
  // CHECK-SAME: padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 1 : si32, padding_top = 0 : si32,
  // CHECK-SAME: stride_height = 1 : si32, stride_width = 1 : si32
>>>>>>> cede6eb71 (Add padding to keep existing tests passing)
  // CHECK-SAME: (tensor<8x6x6x256xf32>, tensor<8x6x6x256xf32>)
  // CHECK-SAME: -> tensor<8x6x6x256xf32>
  %3 = "ttir.pooling"(%arg0, %2) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 1, 0, 0>, pooling_method = #ttir<pooling_method Sum>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> : (tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>) -> tensor<8x256x6x6xf32>
  // CHECK: %[[PERMUTE_1:[0-9]+]] = "ttir.permute"(%[[AVGPOOL]],
  // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
  // CHECK-SAME: (tensor<8x6x6x256xf32>, tensor<8x256x6x6xf32>)
  // CHECK-SAME: -> tensor<8x256x6x6xf32>
  // CHECK: %[[KERNEL:[0-9]+]] = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<8x256x6x6xf32>}> : () -> tensor<8x256x6x6xf32>
  // CHECK: %{{[0-9]+}} = "ttir.multiply"(%[[PERMUTE_1]], %[[KERNEL]], %{{[0-9]+}})
  // CHECK-SAME: (tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>)
  // CHECK-SAME: -> tensor<8x256x6x6xf32>
  %4 = ttir.empty() : tensor<1x1x1x1xf32>
  %5 = "ttir.reshape"(%1, %4) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %6 = ttir.empty() : tensor<8x256x6x6xf32>
  %7 = "ttir.broadcast"(%5, %6) <{broadcast_dimensions = array<i64: 8, 256, 6, 6>}> : (tensor<1x1x1x1xf32>, tensor<8x256x6x6xf32>) -> tensor<8x256x6x6xf32>
  %8 = ttir.empty() : tensor<1xf32>
  %9 = "ttir.typecast"(%0, %8) : (tensor<1xi32>, tensor<1xf32>) -> tensor<1xf32>
  %10 = ttir.empty() : tensor<1x1x1x1xf32>
  %11 = "ttir.reshape"(%9, %10) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %12 = ttir.empty() : tensor<8x256x6x6xf32>
  %13 = "ttir.broadcast"(%11, %12) <{broadcast_dimensions = array<i64: 8, 256, 6, 6>}> : (tensor<1x1x1x1xf32>, tensor<8x256x6x6xf32>) -> tensor<8x256x6x6xf32>
  %14 = ttir.empty() : tensor<8x256x6x6xf32>
  %15 = "ttir.multiply"(%7, %13, %14) : (tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>) -> tensor<8x256x6x6xf32>
  %16 = ttir.empty() : tensor<8x256x6x6xf32>
  %17 = "ttir.div"(%3, %15, %16) : (tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>) -> tensor<8x256x6x6xf32>
  return %17 : tensor<8x256x6x6xf32>
}
