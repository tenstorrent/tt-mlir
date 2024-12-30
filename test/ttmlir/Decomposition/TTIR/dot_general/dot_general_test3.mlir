// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module @jit_loss attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<127x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<127x1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x1xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<1xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %0 = "ttir.dot_general"(%arg2, %arg0) <{batchdims_a = array<i64>, batchdims_b = array<i64>, contractdims_a = array<i64: 1>, contractdims_b = array<i64: 0>}> : (tensor<127x2xf32>, tensor<2x1xf32>) -> tensor<127x1xf32>
    %1 = tensor.empty() : tensor<1xf32>
    %2 = "ttir.typecast"(%arg1, %1) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %3 = tensor.empty() : tensor<127x1xf32>
    %4 = "ttir.broadcast"(%2, %3) <{dimension = []}> : (tensor<1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %5 = tensor.empty() : tensor<127x1xf32>
    %6 = "ttir.add"(%0, %4, %5) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<127x1xf32>, tensor<127x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %7 = tensor.empty() : tensor<127x1xf32>
    %8 = "ttir.subtract"(%6, %arg3, %7) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<127x1xf32>, tensor<127x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %9 = call @integer_pow(%8) : (tensor<127x1xf32>) -> tensor<127x1xf32>
    %10 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %11 = tensor.empty() : tensor<127x1xf32>
    %12 = "ttir.broadcast"(%10, %11) <{dimension = []}> : (tensor<1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %13 = tensor.empty() : tensor<127x1xf32>
    %14 = "ttir.multiply"(%12, %9, %13) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<127x1xf32>, tensor<127x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %15 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %16 = "ttir.constant"() <{value = dense<1.270000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %17 = tensor.empty() : tensor<1xf32>
    %18 = "ttir.div"(%15, %16, %17) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %19 = tensor.empty() : tensor<127x1xf32>
    %20 = "ttir.broadcast"(%18, %19) <{dimension = []}> : (tensor<1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %21 = tensor.empty() : tensor<127x1xf32>
    %22 = "ttir.multiply"(%20, %14, %21) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<127x1xf32>, tensor<127x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %23 = tensor.empty() : tensor<1xf32>
    %24 = "ttir.sum"(%22, %23) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<127x1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %25 = tensor.empty() : tensor<1xf32>
    %26 = "ttir.typecast"(%24, %25) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %27 = "ttir.dot_general"(%22, %arg2) <{batchdims_a = array<i64>, batchdims_b = array<i64>, contractdims_a = array<i64: 0>, contractdims_b = array<i64: 0>}> : (tensor<127x1xf32>, tensor<127x2xf32>) -> tensor<1x2xf32>
    // CHECK: "ttir.permute"
    // CHECK: <{permutation = array<i64: 1, 0>}> : (tensor<127x1xf32>, tensor<1x127xf32>) -> tensor<1x127xf32>
    // CHECK: "ttir.matmul"
    // CHECK: (tensor<1x127xf32>, tensor<127x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %28 = tensor.empty() : tensor<2x1xf32>
    %29 = "ttir.permute"(%27, %28) <{permutation = array<i64: 1, 0>}> : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    return %29, %26 : tensor<2x1xf32>, tensor<1xf32>
  }
  func.func private @integer_pow(%arg0: tensor<127x1xf32>) -> tensor<127x1xf32> {
    return %arg0 : tensor<127x1xf32>
  }
}
