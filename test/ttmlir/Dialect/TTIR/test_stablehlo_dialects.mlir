// REQUIRES: stablehlo
// RUN: ttmlir-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK: chlo
// CHECK: quant
// CHECK: sparse_tensor
// CHECK: stablehlo
// CHECK: vhlo
