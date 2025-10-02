// RUN: ttmlir-opt --wrap-single-affine-loops -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_simple(%arg0: memref<2xf32>) -> memref<2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
  affine.for %arg1 = 0 to 2 {
    %0 = affine.load %arg0[%arg1] : memref<2xf32>
    %1 = arith.maximumf %0, %cst : f32
    affine.store %1, %alloc[%arg1] : memref<2xf32>
  }
  //CHECK: affine.for %arg1 = 0 to 1
  //CHECK-NEXT: affine.for %arg2 = 0 to 2
  //CHECK-NOT: affine.for
  return %alloc : memref<2xf32>
}
