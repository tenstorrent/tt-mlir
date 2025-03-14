// RUN: ttmlir-opt --const-eval-hoist-transform %s | FileCheck %s

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<input>}, %arg1: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: tt.load_cached(@forward_const_eval_0, [%arg2, %arg3])
    // CHECK: ttir.empty()
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.add"(%arg0, %arg1, %{{.*}})
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.subtract"(%arg2, %arg3, %2)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: ttir.empty()
    %4 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}})
    %5 = "ttir.add"(%1, %3, %4)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %5 : tensor<32x32xbf16>
  }

  func.func @forward_split(%arg0: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<input>}, %arg1: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: tt.load_cached(@forward_split_const_eval_0, [%arg1, %arg2])
    // CHECK: tt.load_cached(@forward_split_const_eval_1, [%arg2, %arg3])
    // CHECK: ttir.empty()
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.add"(%arg0, %arg1, %{{.*}})
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3, %4)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: ttir.empty()
    %6 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.subtract"(%{{.*}}, %{{.*}}, %{{.*}})
    %7 = "ttir.subtract"(%1, %3, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: ttir.empty()
    %8 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.multiply"(%{{.*}}, %{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%5, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }

  func.func @forward_merge(%arg0: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<input>}, %arg1: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = tt.load_cached(@forward_merge_const_eval_0, [%arg1, %arg2, %arg3])
    // CHECK: = ttir.empty()
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttir.add"(%arg0, %arg1, %{{.*}})
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3, %4)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %6 = ttir.empty() : tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: = ttir.empty()
    %8 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttir.multiply"(%{{.*}}, %{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%1, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }

  func.func @forward_merge_return_multiple_values(%arg0: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<input>}, %arg1: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = tt.load_cached(@forward_merge_return_multiple_values_const_eval_0, [%arg1, %arg2, %arg3])
    // CHECK: = ttir.empty()
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttir.add"(%arg0, %arg1, %{{.*}})
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3, %4)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %6 = ttir.empty() : tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: = ttir.empty()
    %8 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttir.multiply"(%{{.*}}, %{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%1, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: = ttir.empty()
    %10 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttir.multiply"(%{{.*}}, %{{.*}}, %{{.*}})
    %11 = "ttir.multiply"(%9, %3, %10) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %11 : tensor<32x32xbf16>
  }

    func.func @forward_reuse_zeros(%arg0: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<input>}, %arg1: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<constant>}) -> tensor<32x32xbf16> {
      // CHECK: = tt.load_cached(@forward_reuse_zeros_const_eval_0, [%arg1, %arg2])
      %0 = "ttir.zeros"() <{shape = array<i32:32, 32>}> : () -> tensor<32x32xbf16>
      // CHECK: = ttir.empty()
      %1 = ttir.empty() : tensor<32x32xbf16>
      // CHECK: = "ttir.add"(%arg0, %{{.*}}, %{{.*}})
      %2 = "ttir.add"(%arg0, %0, %1)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %3 = ttir.empty() : tensor<32x32xbf16>
      %4 = "ttir.add"(%arg1, %0, %3)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %5 = ttir.empty() : tensor<32x32xbf16>
      %6 = "ttir.add"(%arg2, %0, %5)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %7 = ttir.empty() : tensor<32x32xbf16>
      %8 = "ttir.multiply"(%4, %6, %7) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      // CHECK: = ttir.empty()
      %9 = ttir.empty() : tensor<32x32xbf16>
      // CHECK: = "ttir.multiply"(%{{.*}}, %{{.*}}, %{{.*}})
      %10 = "ttir.multiply"(%2, %8, %9) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %10 : tensor<32x32xbf16>
    }

  // CHECK-LABEL: func.func @forward_const_eval_0
  // CHECK: "ttir.subtract"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>

  // CHECK-LABEL: func.func @forward_split_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>

  // CHECK-LABEL: func.func @forward_split_const_eval_1
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>

  // CHECK-LABEL: func.func @forward_merge_const_eval_0
  // CHECK: "ttir.subtract"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>

  // CHECK-LABEL: func.func @forward_merge_return_multiple_values_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>
  // CHECK: "ttir.subtract"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>

  // CHECK-LABEL: func.func @forward_reuse_zeros_const_eval_0
  // CHECK: "ttir.zeros"()
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>
  // CHECK: "ttir.multiply"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}>

}
