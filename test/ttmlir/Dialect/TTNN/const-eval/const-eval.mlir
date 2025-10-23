// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @forward_const_eval_0
  // CHECK: = "ttnn.subtract"(%{{.*}}, %{{.*}})

  // CHECK: func.func @forward(
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@forward_const_eval_0, [%arg2, %arg3])
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: %[[TILED_INPUT1:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK: "ttnn.add"(%[[TILED_INPUT1]], %arg1)
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.subtract"(%arg2, %arg3, %2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttnn.add"(%{{.*}}, %{{.*}})
    %5 = "ttir.add"(%1, %3, %4)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %5 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func @forward_split_const_eval_0
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})

  // CHECK-LABEL: func.func @forward_split_const_eval_1
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})

  // CHECK: func.func @forward_split(
  func.func @forward_split(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@forward_split_const_eval_0, [%arg1, %arg2])
    // CHECK: ttcore.load_cached(@forward_split_const_eval_1, [%arg2, %arg3])
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: %[[TILED_INPUT2:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK: "ttnn.add"(%[[TILED_INPUT2]], %arg1)
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3, %4)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %6 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttnn.subtract"(%{{.*}}, %{{.*}})
    %7 = "ttir.subtract"(%1, %3, %6) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %8 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttnn.multiply"(%{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%5, %7, %8) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func @forward_merge_const_eval_0
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})
  // CHECK: = "ttnn.subtract"(%{{.*}}, %{{.*}})

  // CHECK: func.func @forward_merge(
  func.func @forward_merge(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = ttcore.load_cached(@forward_merge_const_eval_0, [%arg1, %arg2, %arg3])
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: %[[TILED_INPUT:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK: = "ttnn.add"(%[[TILED_INPUT]], %arg1)
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3, %4)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %6 = ttir.empty() : tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5, %6) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %8 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%1, %7, %8) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func @forward_merge_return_multiple_values_const_eval_0
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})
  // CHECK: = "ttnn.subtract"(%{{.*}}, %{{.*}})

  // CHECK: func.func @forward_merge_return_multiple_values(
  func.func @forward_merge_return_multiple_values(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = ttcore.load_cached(@forward_merge_return_multiple_values_const_eval_0, [%arg1, %arg2, %arg3])
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: %[[TILED_INPUT:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK: = "ttnn.add"(%[[TILED_INPUT]], %arg1
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3, %4)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %6 = ttir.empty() : tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5, %6) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %8 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%1, %7, %8) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %10 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %11 = "ttir.multiply"(%9, %3, %10) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %11 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func @forward_reuse_zeros_const_eval_0
  // CHECK: = "ttnn.get_device"
  // CHECK: = "ttnn.zeros"(%{{.*}})
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})

  // CHECK: func.func @forward_reuse_zeros(
  func.func @forward_reuse_zeros(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = ttcore.load_cached(@forward_reuse_zeros_const_eval_0, [%arg1, %arg2])
    %0 = "ttir.zeros"() <{shape = array<i32:32, 32>}> : () -> tensor<32x32xbf16>
    %1 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: %[[TILED_INPUT:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK: = "ttnn.add"(%[[TILED_INPUT]], %{{.*}})
    %2 = "ttir.add"(%arg0, %0, %1)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = ttir.empty() : tensor<32x32xbf16>
    %4 = "ttir.add"(%arg1, %0, %3)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %5 = ttir.empty() : tensor<32x32xbf16>
    %6 = "ttir.add"(%arg2, %0, %5)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %7 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %8 = "ttir.multiply"(%2, %6, %7) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %9 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %10 = "ttir.multiply"(%4, %8, %9) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %10 : tensor<32x32xbf16>
  }


  // CHECK-LABEL: func.func @forward_reuse_constant_merge_const_eval_0
  // CHECK: = "ttnn.get_device"
  // CHECK: = "ttnn.full"(%{{.*}})
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})
  // CHECK: = "ttnn.add"(%{{.*}}, %{{.*}})
  // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})

  // CHECK-LABEL: func.func @forward_reuse_constant_merge(
  func.func @forward_reuse_constant_merge(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = ttcore.load_cached(@forward_reuse_constant_merge_const_eval_0, [%arg1, %arg2])
    %0 = "ttir.constant"() <{value = dense<1.111e+00> : tensor<32x32xbf16>}> : () -> tensor<32x32xbf16>
    %1 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: %[[TILED_INPUT:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK: = "ttnn.add"(%[[TILED_INPUT]], %{{.*}})
    %2 = "ttir.add"(%arg0, %0, %1)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = ttir.empty() : tensor<32x32xbf16>
    %4 = "ttir.add"(%arg1, %0, %3)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %5 = ttir.empty() : tensor<32x32xbf16>
    %6 = "ttir.add"(%arg2, %0, %5)  : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %7 = ttir.empty() : tensor<32x32xbf16>
    %8 = "ttir.multiply"(%4, %6, %7) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %9 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %10 = "ttir.multiply"(%2, %8, %9) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %10 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func @non_splat_constant_const_eval_0
  // CHECK: = "ttnn.get_device"
  // CHECK: = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]>
  // CHECK: = "ttnn.neg"

  // CHECK: func.func @non_splat_constant(
  func.func @non_splat_constant(%arg0: tensor<2x2xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<2x2xbf16> {
    // CHECK: = ttcore.load_cached(@non_splat_constant_const_eval_0, [])
    // Create a non-splat constant with different values
    %0 = "ttir.constant"() <{value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xbf16>}> : () -> tensor<2x2xbf16>
    %1 = ttir.empty() : tensor<2x2xbf16>
    %2 = "ttir.neg"(%0, %1) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    %3 = ttir.empty() : tensor<2x2xbf16>
    // CHECK: %[[TILED_INPUT:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK: = "ttnn.add"(%[[TILED_INPUT]], %{{.*}})
    %4 = "ttir.add"(%arg0, %2, %3) : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    return %4 : tensor<2x2xbf16>
  }

  // CHECK-LABEL: func.func @creation_ops_const_eval_0
  // CHECK: "ttnn.zeros"
  // CHECK: "ttnn.ones"
  // CHECK: "ttnn.add"
  // CHECK: "ttnn.arange"
  // CHECK: "ttnn.add"

  // CHECK-LABEL: func.func @creation_ops(
  func.func @creation_ops() -> tensor<4x4xbf16> {
    // CHECK: = ttcore.load_cached(@creation_ops_const_eval_0, [])
    %0 = "ttir.zeros"() <{shape = array<i32: 4, 4>}> : () -> tensor<4x4xbf16>
    %1 = "ttir.ones"() <{shape = array<i32: 4, 4>}> : () -> tensor<4x4xbf16>

    %2 = ttir.empty() : tensor<4x4xbf16>
    %3 = "ttir.add"(%0, %1, %2) : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>

    %4 = ttir.empty() : tensor<4x4xbf16>
    %5 = "ttir.arange"() {start = 0 : si64, end = 4 : si64, step = 1 : si64, arange_dimension = 0 : i64} : () -> tensor<4x4xbf16>
    %6 = "ttir.add"(%3, %5, %4) : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>

    return %6 : tensor<4x4xbf16>
  }

  // CHECK-LABEL: func.func @forward_all_const_const_eval
  // CHECK: "ttnn.add"


  // CHECK-LABEL: func.func @forward_all_const(
  func.func @forward_all_const(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: %[[LOAD_CACHED_RESULT:.+]] = ttcore.load_cached(@forward_all_const_const_eval_0, [%arg0, %arg1])
    // CHECK-NOT: "ttnn.add"
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: return %[[LOAD_CACHED_RESULT]]
    return %1 : tensor<32x32xbf16>
  }
}
