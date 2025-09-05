//RUN: ttmlir-opt --wrap-single-affine-loops %s | FileCheck %s

func.func @test_simple(%arg0: memref<3000xf32>) -> memref<3000xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<3000xf32>
  affine.for %arg1 = 0 to 3000 {
    %0 = affine.load %arg0[%arg1] : memref<3000xf32>
    %1 = arith.maximumf %0, %cst : f32
    affine.store %1, %alloc[%arg1] : memref<3000xf32>
  }
  //CHECK: affine.for %arg1 = 0 to 64
  //CHECK-NEXT: affine.for %arg2 = 0 to 47
  //CHECK-NOT: affine.for
  //CHECK: arith.cmpi
  //CHECK: scf.if
  return %alloc : memref<3000xf32>
}
