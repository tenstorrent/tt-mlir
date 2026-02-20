// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module {
  // Test basic arange with default start=0, step=1
  func.func @arange_basic() -> tensor<10xf32> {
    // CHECK-LABEL: func.func @arange_basic
    // CHECK: tensor.empty
    // CHECK: linalg.generic
    // CHECK: linalg.index
    // CHECK: arith.sitofp
    // CHECK: arith.mulf
    // CHECK: arith.addf
    // CHECK: linalg.yield
    // CHECK-NOT: ttir.arange
    %0 = "ttir.arange"() <{arange_dimension = 0 : i64, dtype =f32, end = 10 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }

  // Test arange with non-zero start
  func.func @arange_with_start() -> tensor<5xf32> {
    // CHECK-LABEL: func.func @arange_with_start
    // CHECK: tensor.empty
    // CHECK: linalg.generic
    // CHECK: linalg.index
    // CHECK-NOT: ttir.arange
    %0 = "ttir.arange"() <{arange_dimension = 0 : i64, dtype = f32, end = 10 : si64, start = 5 : si64, step = 1 : si64}> : () -> tensor<5xf32>
    return %0 : tensor<5xf32>
  }

  // Test arange with step > 1
  func.func @arange_with_step() -> tensor<5xf32> {
    // CHECK-LABEL: func.func @arange_with_step
    // CHECK: tensor.empty
    // CHECK: linalg.generic
    // CHECK: linalg.index
    // CHECK-NOT: ttir.arange
    %0 = "ttir.arange"() <{arange_dimension = 0 : i64, dtype = f32, end = 10 : si64, start = 0 : si64, step = 2 : si64}> : () -> tensor<5xf32>
    return %0 : tensor<5xf32>
  }

  // Test arange with bf16 element type
  func.func @arange_bf16() -> tensor<8xbf16> {
    // CHECK-LABEL: func.func @arange_bf16
    // CHECK: tensor.empty
    // CHECK: linalg.generic
    // CHECK: linalg.index
    // CHECK: arith.sitofp
    // CHECK-NOT: ttir.arange
    %0 = "ttir.arange"() <{arange_dimension = 0 : i64, dtype = f32, end = 8 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<8xbf16>
    return %0 : tensor<8xbf16>
  }

  // Test arange with integer element type
  func.func @arange_i32() -> tensor<10xi32> {
    // CHECK-LABEL: func.func @arange_i32
    // CHECK: tensor.empty
    // CHECK: linalg.generic
    // CHECK: linalg.index
    // CHECK: arith.index_cast
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: linalg.yield
    // CHECK-NOT: ttir.arange
    %0 = "ttir.arange"() <{arange_dimension = 0 : i64, dtype = f32, end = 10 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<10xi32>
    return %0 : tensor<10xi32>
  }
}
