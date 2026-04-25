// RUN: ttmlir-opt --convert-ttir-to-smt -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @test_eq
func.func @test_eq(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi1> {
  // CHECK: %[[BOOL:.*]] = smt.eq {{.*}} : !smt.bv<32>
  // CHECK: smt.ite %[[BOOL]]
  %0 = "ttir.eq"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  return %0 : tensor<1xi1>
}

// CHECK-LABEL: func.func @test_ne
func.func @test_ne(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi1> {
  // CHECK: %[[BOOL:.*]] = smt.distinct {{.*}} : !smt.bv<32>
  // CHECK: smt.ite %[[BOOL]]
  %0 = "ttir.ne"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  return %0 : tensor<1xi1>
}

// Default signless integer -> signed comparison
// CHECK-LABEL: func.func @test_lt_signed
func.func @test_lt_signed(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi1> {
  // CHECK: smt.bv.cmp slt {{.*}} : !smt.bv<32>
  %0 = "ttir.lt"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  return %0 : tensor<1xi1>
}

// Explicitly unsigned -> unsigned comparison
// CHECK-LABEL: func.func @test_lt_unsigned
func.func @test_lt_unsigned(%a: tensor<1xui32>, %b: tensor<1xui32>) -> tensor<1xi1> {
  // CHECK: smt.bv.cmp ult {{.*}} : !smt.bv<32>
  %0 = "ttir.lt"(%a, %b) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xi1>
  return %0 : tensor<1xi1>
}

// CHECK-LABEL: func.func @test_le
func.func @test_le(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi1> {
  // CHECK: smt.bv.cmp sle {{.*}} : !smt.bv<32>
  %0 = "ttir.le"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  return %0 : tensor<1xi1>
}

// CHECK-LABEL: func.func @test_gt
func.func @test_gt(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi1> {
  // CHECK: smt.bv.cmp sgt {{.*}} : !smt.bv<32>
  %0 = "ttir.gt"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  return %0 : tensor<1xi1>
}

// CHECK-LABEL: func.func @test_ge
func.func @test_ge(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi1> {
  // CHECK: smt.bv.cmp sge {{.*}} : !smt.bv<32>
  %0 = "ttir.ge"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  return %0 : tensor<1xi1>
}

// Unsigned >= test
// CHECK-LABEL: func.func @test_ge_unsigned
func.func @test_ge_unsigned(%a: tensor<1xui32>, %b: tensor<1xui32>) -> tensor<1xi1> {
  // CHECK: smt.bv.cmp uge {{.*}} : !smt.bv<32>
  %0 = "ttir.ge"(%a, %b) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xi1>
  return %0 : tensor<1xi1>
}
