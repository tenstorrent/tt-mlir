// RUN: ttmlir-opt --collapse-parallel-loops -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_collapse_last_two_dims() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c224 = arith.constant 224 : index
  %c3 = arith.constant 3 : index

  // CHECK: scf.parallel ([[ARG0:%.+]], [[ARG1:%.+]], [[ARG2:%.+]]) = (%c0_0, %c0, %c0) to (%c672, %c224, %c8) step (%c1_1, %c1, %c1)
  scf.parallel (%arg1, %arg2, %arg3, %arg4) = (%c0, %c0, %c0, %c0) to (%c8, %c224, %c224, %c3) step (%c1, %c1, %c1, %c1) {
    // CHECK: [[ARG3:%.+]] = arith.divui [[ARG0]]
    // CHECK: [[TMP:%.+]]= arith.muli [[ARG3]]
    // CHECK: [[ARG4:%.+]] = arith.subi [[ARG0]], [[TMP]]
    %sum = arith.addi %arg1, %arg2 : index
    %product = arith.muli %arg3, %arg4 : index

    // CHECK: arith.addi [[ARG2]], [[ARG1]]
    // CHECK: arith.muli [[ARG3]], [[ARG4]]
  }
  return
}

func.func @test_collapse_last_three_dims() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // CHECK: scf.parallel ([[ARG0:%.+]], [[ARG1:%.+]], [[ARG2:%.+]]) = (%c0_0, %c0, %c0) to (%c512, %c8, %c8) step (%c1_1, %c1, %c1)
  scf.parallel (%arg1, %arg2, %arg3, %arg4, %arg5) = (%c0, %c0, %c0, %c0, %c0) to (%c8, %c8, %c8, %c8, %c8) step (%c1, %c1, %c1, %c1, %c1) {

    %sum = arith.addi %arg1, %arg2 : index
    %product = arith.muli %arg3, %arg4 : index
    %result = arith.addi %arg5, %arg5 : index
    // CHECK: [[ARG3:%.+]] = arith.divui [[ARG0]]
    // CHECK: [[TMP1:%.+]]= arith.muli [[ARG3]]
    // CHECK: [[TMP2:%.+]] = arith.subi [[ARG0]], [[TMP1]]
    // CHECK: [[ARG4:%.+]] = arith.divui [[TMP2]]
    // CHECK: [[TMP3:%.+]]= arith.muli [[ARG4]]
    // CHECK: [[ARG5:%.+]] = arith.subi [[TMP2]], [[TMP3]]
    // CHECK: arith.addi [[ARG2]], [[ARG1]]
    // CHECK: arith.muli [[ARG3]], [[ARG4]]
    // CHECK: arith.addi [[ARG5]], [[ARG5]]

  }
  return
}
