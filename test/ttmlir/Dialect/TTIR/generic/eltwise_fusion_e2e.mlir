// RUN: ttmlir-opt %s --ttir-elementwise-fusion --linalg-fuse-elementwise-ops -canonicalize -split-input-file | FileCheck %s

!ttype = tensor<128x96xf32>

func.func @named_elementwise(%a: !ttype, %b: !ttype, %c: !ttype) -> (!ttype) {
  // named elementwise op, binary:
  // CHECK: ttir.generic{{.+}}iterator_types = [#parallel, #parallel]
  // CHECK: linalg.generic{{.+}}iterator_types = ["parallel", "parallel"]
  // CHECK: ttir.tile_add
  %0 = ttir.empty() : !ttype
  %1 = "ttir.multiply"(%a, %b, %0) : (!ttype, !ttype, !ttype) -> !ttype
  %2 = ttir.empty() : !ttype
  %3 = "ttir.add"(%1, %c, %2) : (!ttype, !ttype, !ttype) -> !ttype
  return %3 : !ttype
}
