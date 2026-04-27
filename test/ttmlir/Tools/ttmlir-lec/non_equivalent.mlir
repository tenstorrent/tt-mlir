// RUN: ttmlir-lec %s -c1=add -c2=sub --emit-smtlib -o %t
// RUN: FileCheck %s --input-file=%t

// Addition is not subtraction — the miter's assertion is satisfiable.
// Use --emit-smtlib to verify the LEC pipeline produces a valid script
// without needing a solver in the lit environment.

// CHECK: (check-sat)
// CHECK-NOT: error

func.func @add(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "ttir.add"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

func.func @sub(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "ttir.subtract"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}
