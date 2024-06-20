// RUN: ttmlir-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-SAME: arith,builtin,cf,emitc,func,linalg,ml_program,scf,tensor,tosa,tt,ttir,ttkernel,ttmetal,vector
