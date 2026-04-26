// RUN: ttmlir-opt --convert-ttir-to-smt "--construct-ttir-lec=c1=fn_a c2=fn_b check-output=match" -o %t %s
// RUN: FileCheck --check-prefix=CHECK-MATCH %s --input-file=%t

// RUN: ttmlir-opt --convert-ttir-to-smt "--construct-ttir-lec=c1=fn_a c2=fn_b check-output=diff" -o %t2 %s
// RUN: FileCheck --check-prefix=CHECK-DIFF %s --input-file=%t2

// RUN: ttmlir-opt --convert-ttir-to-smt "--construct-ttir-lec=c1=fn_a c2=fn_b check-output-idx=0" -o %t3 %s
// RUN: FileCheck --check-prefix=CHECK-IDX0 %s --input-file=%t3

// fn_a: out0 = a+b (named "match"), out1 = a-b (named "diff")
func.func @fn_a(%a: tensor<1xi32>, %b: tensor<1xi32>)
    -> (tensor<1xi32> {hw.port_name = "match"},
        tensor<1xi32> {hw.port_name = "diff"}) {
  %0 = "ttir.add"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = "ttir.subtract"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0, %1 : tensor<1xi32>, tensor<1xi32>
}

// fn_b: out0 = b+a (matches fn_a out0), out1 = a*b (does NOT match fn_a out1)
func.func @fn_b(%a: tensor<1xi32>, %b: tensor<1xi32>)
    -> (tensor<1xi32> {hw.port_name = "match"},
        tensor<1xi32> {hw.port_name = "diff"}) {
  %0 = "ttir.add"(%b, %a) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = "ttir.multiply"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0, %1 : tensor<1xi32>, tensor<1xi32>
}

// When check-output=match, only one distinct (one OR with the false seed)
// CHECK-MATCH-LABEL: func.func @lec_main()
// CHECK-MATCH: smt.distinct
// CHECK-MATCH-NOT: smt.distinct

// When check-output=diff, only one distinct as well
// CHECK-DIFF-LABEL: func.func @lec_main()
// CHECK-DIFF: smt.distinct
// CHECK-DIFF-NOT: smt.distinct

// When check-output-idx=0, only one distinct
// CHECK-IDX0-LABEL: func.func @lec_main()
// CHECK-IDX0: smt.distinct
// CHECK-IDX0-NOT: smt.distinct
