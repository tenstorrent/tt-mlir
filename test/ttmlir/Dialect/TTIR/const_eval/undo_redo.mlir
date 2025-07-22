// RUN: ttmlir-opt --const-eval-hoist-transform %s -o %t1.mlir
// RUN: ttmlir-opt --undo-const-eval %t1.mlir -o %t2.mlir
// RUN: ttmlir-opt --const-eval-hoist-transform %t2.mlir -o %t3.mlir
// RUN: ttmlir-opt --const-eval-hoist-transform %t1.mlir -o %t4.mlir
// RUN: FileCheck %s < %t1.mlir
// RUN: FileCheck %s --check-prefix=UNDONE < %t2.mlir
// RUN: diff %t1.mlir %t3.mlir
// RUN: diff %t1.mlir %t4.mlir

module {
  // CHECK-LABEL: func.func @test_undo_redo_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}})

  // UNDONE-LABEL: func.func @test_undo_redo
  // UNDONE-NOT: func.func @test_undo_redo_const_eval_0
  // UNDONE-NOT: ttcore.load_cached
  // UNDONE: "ttir.add"(%arg2, %arg3, %{{.*}})
  // UNDONE: "ttir.add"(%arg0, %arg1, %{{.*}})
  // UNDONE: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}})

  // CHECK: func.func @test_undo_redo(
  func.func @test_undo_redo(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                    %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                    %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                    %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@test_undo_redo_const_eval_0, [%arg2, %arg3])
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg2, %arg3, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.add"(%arg0, %arg1, %{{.*}})
    %3 = "ttir.add"(%arg0, %arg1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}})
    %5 = "ttir.add"(%3, %1, %4) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %5 : tensor<32x32xbf16>
  }

  // Now test a more complex case with multiple const-eval functions
  // CHECK-LABEL: func.func @test_multi_undo_redo_const_eval_0
  // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}})

  // CHECK-LABEL: func.func @test_multi_undo_redo_const_eval_1
  // CHECK: "ttir.multiply"(%{{.*}}, %{{.*}}, %{{.*}})

  // UNDONE-LABEL: func.func @test_multi_undo_redo
  // UNDONE-NOT: func.func @test_multi_undo_redo_const_eval_0
  // UNDONE-NOT: func.func @test_multi_undo_redo_const_eval_1
  // UNDONE-NOT: ttcore.load_cached
  // UNDONE: "ttir.add"(%arg2, %arg3, %{{.*}})
  // UNDONE: "ttir.multiply"(%arg3, %arg4, %{{.*}})
  // UNDONE: "ttir.add"(%arg0, %arg1, %{{.*}})
  // UNDONE: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}})
  // UNDONE: "ttir.multiply"(%{{.*}}, %{{.*}}, %{{.*}})

  // CHECK: func.func @test_multi_undo_redo(
  func.func @test_multi_undo_redo(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                          %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                          %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                          %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                          %arg4: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached(@test_multi_undo_redo_const_eval_0, [%arg2, %arg3])
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg2, %arg3, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: ttcore.load_cached(@test_multi_undo_redo_const_eval_1, [%arg3, %arg4])
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.multiply"(%arg3, %arg4, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %4 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.add"(%arg0, %arg1, %{{.*}})
    %5 = "ttir.add"(%arg0, %arg1, %4) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %6 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.add"(%{{.*}}, %{{.*}}, %{{.*}})
    %7 = "ttir.add"(%5, %1, %6) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %8 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.multiply"(%{{.*}}, %{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%7, %3, %8) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %9 : tensor<32x32xbf16>
  }
}
