// RUN: ttmlir-opt --const-eval-hoist-transform %s -o %t1.mlir
// RUN: ttmlir-opt --const-eval-hoist-transform %t1.mlir -o %t2.mlir
// RUN: FileCheck --input-file=%t1.mlir %s
// RUN: diff %t1.mlir %t2.mlir

module {
  // CHECK-LABEL: func.func private @test_redo_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})

  // CHECK: func.func @test_redo(
  func.func @test_redo(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                    %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                    %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                    %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@test_redo_const_eval_0, [%arg2, %arg3])
    %1 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: "ttir.add"(%arg0, %arg1)
    %3 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
    %5 = "ttir.add"(%3, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %5 : tensor<32x32xbf16>
  }

  // Now test a more complex case with multiple const-eval functions
  // CHECK-LABEL: func.func private @test_multi_redo_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}})

  // CHECK-LABEL: func.func private @test_multi_redo_const_eval_1
  // CHECK: "ttir.multiply"(%{{.*}}, %{{.*}})

  // CHECK: func.func @test_multi_redo(
  func.func @test_multi_redo(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                          %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                          %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                          %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                          %arg4: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@test_multi_redo_const_eval_0, [%arg2, %arg3])
    %1 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: ttcore.load_cached(@test_multi_redo_const_eval_1, [%arg3, %arg4])
    %3 = "ttir.multiply"(%arg3, %arg4) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttir.add"(%arg0, %arg1)
    %5 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttir.add"(%{{.*}}, %{{.*}})
    %7 = "ttir.add"(%5, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttir.multiply"(%{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%7, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %9 : tensor<32x32xbf16>
  }
}
