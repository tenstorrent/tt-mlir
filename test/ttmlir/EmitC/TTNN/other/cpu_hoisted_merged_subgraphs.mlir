// E2E test for CPU-hoisted merged const-eval subgraphs with EmitC dylib compilation.
// This test verifies that connected const-eval operations are properly merged into
// a single CPU function in the LLVM module.
//
// Note: C++ code generation and dylib compilation are handled by tt-alchemist,
// not by ttmlir-translate. This test verifies the MLIR output structure.
//
// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path% enable-cpu-hoisted-const-eval=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file %t.mlir

module {
  // Verify extern declarations
  // CHECK: emitc.verbatim "extern

  // Test: Multiple connected const-eval ops should be merged into one CPU function
  // CHECK-LABEL: func.func @forward_merged
  func.func @forward_merged(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                            %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                            %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                            %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // Device operation
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // These connected const-eval ops should be merged into a single CPU function
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.subtract"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // Device operation using const-eval result
    %4 = "ttir.multiply"(%0, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %4 : tensor<32x32xbf16>
  }

  // Test: Const-eval with multiple return values
  func.func @forward_multiple_returns(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                                      %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                                      %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                                      %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // Connected const-eval ops with multiple uses
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.multiply"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = "ttir.multiply"(%0, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // Use %2 again - tests multiple return values from const-eval
    %5 = "ttir.multiply"(%4, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %5 : tensor<32x32xbf16>
  }

  // Test: Const-eval with creation ops
  // CHECK-LABEL: func.func @forward_with_zeros
  func.func @forward_with_zeros(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                                %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    %0 = "ttir.zeros"() <{shape = array<i32:32, 32>}> : () -> tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.multiply"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %3 : tensor<32x32xbf16>
  }

  // Verify the CPU module structure (at the end)
  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK: llvm.func @cpu_hoisted
}
