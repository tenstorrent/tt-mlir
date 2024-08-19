// RUN: ttmlir-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK: arith
// CHECK: builtin
// CHECK: cf
// CHECK: emitc
// CHECK: func
// CHECK: linalg
// CHECK: ml_program
// CHECK: scf
// CHECK: tensor
// CHECK: tosa
// CHECK: tt
// CHECK: ttir
// CHECK: ttkernel
// CHECK: ttmetal
// CHECK: ttnn
// CHECK: vector
