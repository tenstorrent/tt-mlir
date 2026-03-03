// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
// Test that integer input to sort is automatically converted to ui16 by the workaround pass

module {
  func.func public @test_sort_integer_input_workaround(%arg0: tensor<1x10xsi32>) -> (tensor<1x10xsi32>, tensor<1x10xsi32>) {
    // CHECK-LABEL: func.func public @test_sort_integer_input_workaround
    // Verify si32→ui16 typecast is inserted
    // CHECK: ttnn.to_layout
    // CHECK: dtype = #ttcore.supportedDataTypes<u16>
    // CHECK: tensor<1x10xsi32
    // CHECK: tensor<1x10xui16
    // Verify sort operates on ui16
    // CHECK: ttnn.sort
    // CHECK: descending = true
    // CHECK: tensor<1x10xui16
    // CHECK: tensor<1x10xui16
    // CHECK: tensor<1x10xui16
    // Verify values converted back to si32
    // CHECK: ttnn.to_layout
    // CHECK: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK: tensor<1x10xui16
    // CHECK: tensor<1x10xsi32
    %values, %indices = "ttir.sort"(%arg0) <{descending = true, dim = 1 : si32, stable = false}> : (tensor<1x10xsi32>) -> (tensor<1x10xsi32>, tensor<1x10xsi32>)
    return %values, %indices : tensor<1x10xsi32>, tensor<1x10xsi32>
  }
}
