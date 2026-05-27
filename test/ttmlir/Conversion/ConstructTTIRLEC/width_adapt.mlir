// Verify width adapter inserts zero-extend on narrower side and truncates on wider side.
// RUN: ttmlir-opt --convert-ttir-to-smt "--construct-ttir-lec=c1=fn_i3 c2=fn_i8" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Narrow: i3 in/out
func.func @fn_i3(%a: tensor<1xi3>) -> tensor<1xi3> {
  return %a : tensor<1xi3>
}

// Promoted: i8 in/out (mimicking TTIRWorkarounds promotion)
func.func @fn_i8(%a: tensor<1xi8>) -> tensor<1xi8> {
  return %a : tensor<1xi8>
}

// CHECK-LABEL: func.func @lec_main()
// Common width is i3, declare_fun at bv<3>
// CHECK: smt.declare_fun "arg0" : !smt.bv<3>
// Zero-extend on the i8 side: concat 5 zero bits
// CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<5>
// CHECK: smt.bv.concat
// On output: truncate the i8 output to i3
// CHECK: smt.bv.extract {{.*}} from 0 : (!smt.bv<8>) -> !smt.bv<3>
// CHECK: smt.distinct
