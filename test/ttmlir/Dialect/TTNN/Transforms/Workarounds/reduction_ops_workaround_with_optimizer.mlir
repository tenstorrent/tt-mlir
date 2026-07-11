// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func public @test_reduce_sum_workaround_with_optimizer(%arg0: tensor<128x10xsi32>) -> tensor<128xsi32> {
  // CHECK-LABEL: func.func public @test_reduce_sum_workaround_with_optimizer
  // CHECK: %[[ARG0:.*]] = "ttnn.to_layout"(%arg0)
  // CHECK-SAME: tensor<128x10xsi32,
  // CHECK-SAME: -> tensor<128x10xf32,
  // CHECK: %[[SUM:.*]] = "ttnn.sum"(%[[ARG0]])
  // CHECK-SAME: tensor<128x10xf32,
  // CHECK-SAME: -> tensor<128xf32,
  %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10xsi32>) -> tensor<128xsi32>
  // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[SUM]])
  // CHECK-SAME: tensor<128xf32,
  // CHECK-SAME: -> tensor<128xsi32,
  return %0 : tensor<128xsi32>
}

func.func public @test_reduce_mean_workaround_with_optimizer(%arg0: tensor<128x10xsi32>) -> tensor<128xsi32> {
  // CHECK-LABEL: func.func public @test_reduce_mean_workaround_with_optimizer
  // CHECK: %[[ARG0:.*]] = "ttnn.to_layout"(%arg0)
  // CHECK-SAME: tensor<128x10xsi32,
  // CHECK-SAME: -> tensor<128x10xf32,
  // CHECK: %[[MEAN:.*]] = "ttnn.mean"(%[[ARG0]])
  // CHECK-SAME: tensor<128x10xf32,
  // CHECK-SAME: -> tensor<128xf32,
  %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10xsi32>) -> tensor<128xsi32>
  // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[MEAN]])
  // CHECK-SAME: tensor<128xf32,
  // CHECK-SAME: -> tensor<128xsi32,
  return %0 : tensor<128xsi32>
}

func.func public @test_reduce_max_workaround_with_optimizer(%arg0: tensor<128x32xsi32>) -> tensor<128xsi32> {
  // CHECK-LABEL: func.func public @test_reduce_max_workaround_with_optimizer
  // CHECK: %[[ARG0:.*]] = "ttnn.to_layout"(%arg0)
  // CHECK-SAME: tensor<128x32xsi32,
  // CHECK-SAME: -> tensor<128x32xf32,
  // CHECK: %[[MAX:.*]] = "ttnn.max"(%[[ARG0]])
  // CHECK-SAME: tensor<128x32xf32,
  // CHECK-SAME: -> tensor<128xf32,
  %0 = "ttir.max"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x32xsi32>) -> tensor<128xsi32>
  // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[MAX]])
  // CHECK-SAME: tensor<128xf32,
  // CHECK-SAME: -> tensor<128xsi32,
  return %0 : tensor<128xsi32>
}

func.func public @test_reduce_min_workaround_with_optimizer(%arg0: tensor<128x32xsi32>) -> tensor<128xsi32> {
  // CHECK-LABEL: func.func public @test_reduce_min_workaround_with_optimizer
  // CHECK: %[[ARG0:.*]] = "ttnn.to_layout"(%arg0)
  // CHECK-SAME: tensor<128x32xsi32,
  // CHECK-SAME: -> tensor<128x32xf32,
  // CHECK: %[[MIN:.*]] = "ttnn.min"(%[[ARG0]])
  // CHECK-SAME: tensor<128x32xf32,
  // CHECK-SAME: -> tensor<128xf32,
  %0 = "ttir.min"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x32xsi32>) -> tensor<128xsi32>
  // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[MIN]])
  // CHECK-SAME: tensor<128xf32,
  // CHECK-SAME: -> tensor<128xsi32,
  return %0 : tensor<128xsi32>
}
