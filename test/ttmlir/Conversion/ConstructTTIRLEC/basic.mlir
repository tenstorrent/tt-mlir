// RUN: ttmlir-opt --convert-ttir-to-smt "--construct-ttir-lec=c1=add_ab c2=add_ba" -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @add_ab(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "ttir.add"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

func.func @add_ba(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "ttir.add"(%b, %a) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// Original two functions should be erased and replaced by lec_main
// CHECK-LABEL: func.func @lec_main()
// CHECK: smt.solver
// CHECK: smt.declare_fun "arg0" : !smt.bv<32>
// CHECK: smt.declare_fun "arg1" : !smt.bv<32>
// CHECK: smt.assert
// CHECK: smt.check sat
