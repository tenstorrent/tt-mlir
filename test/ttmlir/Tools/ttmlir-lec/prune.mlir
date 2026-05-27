// RUN: ttmlir-lec %s -c1=multi_out -c2=multi_out_impl \
// RUN:   --check-output="port_a" --emit-smtlib -o %t
// RUN: FileCheck %s --input-file=%t

// Two-output function where only the first output ("port_a") is checked.
// After pruning, only the cone feeding port_a should remain in the SMT script.
// The unused port_b add/sub pair must be dropped.

// The pruned function has one tensor<1xi32> result, so the SMT solver sees
// a 32-bit bitvector comparison.
// CHECK: BitVec 32
// CHECK-NOT: bvneg
// CHECK: (check-sat)
// CHECK-NOT: error

func.func @multi_out(%a: tensor<1xi32>, %b: tensor<1xi32>)
    -> (tensor<1xi32> {ttir.name = "port_a"},
        tensor<1xi32> {ttir.name = "port_b"}) {
  %0 = "ttir.add"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = "ttir.subtract"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0, %1 : tensor<1xi32>, tensor<1xi32>
}

func.func @multi_out_impl(%a: tensor<1xi32>, %b: tensor<1xi32>)
    -> (tensor<1xi32> {ttir.name = "port_a"},
        tensor<1xi32> {ttir.name = "port_b"}) {
  %0 = "ttir.add"(%b, %a) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = "ttir.add"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0, %1 : tensor<1xi32>, tensor<1xi32>
}
