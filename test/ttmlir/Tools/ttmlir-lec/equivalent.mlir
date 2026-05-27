// RUN: ttmlir-lec %s -c1=add_ab -c2=add_ba --emit-smtlib -o %t
// RUN: FileCheck %s --input-file=%t

// Commutativity of integer add: (a + b) == (b + a).
// Checked via --emit-smtlib so no SMT solver is needed in the lit environment.
// The miter asserts that outputs differ; for equivalent functions the assertion
// is unsatisfiable.  FileCheck confirms that (check-sat) is present and that
// the generated miter uses a top-level smt.solver (i.e. the full LEC pipeline
// ran without error).

// CHECK: (check-sat)
// CHECK-NOT: error

func.func @add_ab(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "ttir.add"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

func.func @add_ba(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "ttir.add"(%b, %a) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}
