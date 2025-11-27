// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s

// CHECK: module {
module {
  // CHECK: func.func @test_arange_dtype() -> tensor<5xi32>
  func.func @test_arange_dtype() -> tensor<5xi32> {
    // CHECK: %[[VAR:.*]] = "ttir.arange"() <{
    // CHECK-SAME: arange_dimension = 0 : i64,
    // CHECK-SAME: dtype = i32,
    // CHECK-SAME: end = 5 : si64,
    // CHECK-SAME: start = 0 : si64,
    // CHECK-SAME: step = 1 : si64}>
    // CHECK-SAME: : () -> tensor<5xi32>
    %0 = "ttir.arange"() <{
      arange_dimension = 0 : i64,
      dtype = i32,
      end = 5 : si64,
      start = 0 : si64,
      step = 1 : si64
    }> : () -> tensor<5xi32>
    return %0 : tensor<5xi32>
  }
}
