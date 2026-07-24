// RUN: ttmlir-opt --const-eval-hoist-transform -o %t %s
// RUN: FileCheck %s --input-file=%t

module {

  // CHECK-LABEL: func.func private @add_neg_const_eval_0
  // CHECK: "ttir.subtract"(%{{.*}}, %{{.*}})
  // CHECK: "ttir.neg"(%{{.*}})
  func.func private @add_neg_const_eval_0(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
    %0 = "ttir.subtract"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }

  // CHECK: func.func @add_neg(
  func.func @add_neg(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@add_neg_const_eval_0, [%arg2, %arg3])
    // CHECK-NOT: "ttcore.load_cached"
    %0 = ttcore.load_cached(@add_neg_const_eval_0, [%arg2, %arg3]) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: "ttir.add"(%arg0, %arg1)
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.neg"(%0) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
    %5 = "ttir.add"(%1, %2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %5 : tensor<32x32xbf16>
  }


  // CHECK-LABEL: func.func private @disjunct_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  func.func private @disjunct_const_eval_0(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
  // CHECK-LABEL: func.func private @disjunct_const_eval_1
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})

  // CHECK: func.func @disjunct(
  func.func @disjunct(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@disjunct_const_eval_0, [%arg1, %arg2])
    // CHECK: ttcore.load_cached(@disjunct_const_eval_1, [%arg2, %arg3])
    // CHECK: "ttir.add"(%arg0, %arg1)
    %0 = ttcore.load_cached(@disjunct_const_eval_0, [%arg1, %arg2]) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg2, %arg3)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: "ttir.subtract"(%{{.*}}, %{{.*}})
    %3 = "ttir.subtract"(%1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: "ttir.multiply"(%{{.*}}, %{{.*}})
    %4 = "ttir.multiply"(%2, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %4 : tensor<32x32xbf16>
  }


  // CHECK-LABEL: func.func private @additional_args_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  // CHECK: "ttir.subtract"(%{{.*}}, %{{.*}})
  func.func private @additional_args_const_eval_0(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }

  // CHECK: func.func @additional_args(
  func.func @additional_args(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = ttcore.load_cached(@additional_args_const_eval_0, [%arg1, %arg2, %arg3])
    %0 = ttcore.load_cached(@additional_args_const_eval_0, [%arg2, %arg3]) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: = "ttir.add"(%arg0, %arg1)
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg1, %arg2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.subtract"(%2, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: = "ttir.multiply"(%{{.*}}, %{{.*}})
    %4 = "ttir.multiply"(%1, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %4 : tensor<32x32xbf16>
  }


  // CHECK-LABEL: func.func private @return_multiple_values_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  // CHECK: "ttir.subtract"(%{{.*}}, %{{.*}})
  // CHECK: return %{{.*}}, %{{.*}}
  func.func private @return_multiple_values_const_eval_0(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> (tensor<32x32xbf16>) attributes {tt.function_type = "const_eval"} {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }

  // CHECK: func.func @return_multiple_values(
  func.func @return_multiple_values(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = ttcore.load_cached(@return_multiple_values_const_eval_0, [%arg1, %arg2, %arg3])
    %0 = ttcore.load_cached(@return_multiple_values_const_eval_0, [%arg1, %arg2]) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> (tensor<32x32xbf16>)
    // CHECK: = "ttir.add"(%arg0, %arg1)
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg2, %arg3)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.subtract"(%0, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: = "ttir.multiply"(%{{.*}}, %{{.*}})
    %4 = "ttir.multiply"(%1, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: = "ttir.multiply"(%{{.*}}, %{{.*}})
    %11 = "ttir.multiply"(%4, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %11 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func private @join_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  func.func private @join_const_eval_0(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
  // CHECK-NOT: func.func private @join_const_eval_1
  func.func private @join_const_eval_1(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
  // CHECK: func.func @join
  func.func @join(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@join_const_eval_0, [%arg0, %arg1, %arg2])
    %0 = ttcore.load_cached(@join_const_eval_0, [%arg0, %arg1]) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = ttcore.load_cached(@join_const_eval_1, [%arg1, %arg2]) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }


  // CHECK-LABEL: func.func private @shared_use_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  // CHECK: "ttir.neg"(%{{.*}})
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
  func.func private @shared_use_const_eval_0(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }

  // Two subgraphs that share only the load_cached op should be hoisted into a
  // single const-eval function.

  // CHECK: func.func @shared_use
  func.func @shared_use(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@shared_use_const_eval_0, [%arg1, %arg2])
    // CHECK-NOT: "ttcore.load_cached"
    %0 = ttcore.load_cached(@shared_use_const_eval_0, [%arg1, %arg2]) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.neg"(%0) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: "ttir.add"(%{{.*}}, %arg0)
    %3 = "ttir.add"(%2, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %3 : tensor<32x32xbf16>
  }
}
