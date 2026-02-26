// RUN: ttmlir-opt --d2m-op-scheduler %s | FileCheck %s

module {
  func.func @test_op_scheduler() {
    // Create dummy memrefs and circular buffers for testing
    %alloc_32 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_4 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_8 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_12 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_16 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_20 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_24 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_28 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_29 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_30 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_34 = memref.alloc() : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>

    // CHECK-LABEL: d2m.generic
    // CHECK: ^unified0
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_32, %alloc_4, %alloc_8, %alloc_12, %alloc_16, %alloc_20, %alloc_24, %alloc_28, %alloc_29, %alloc_30 : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)
        outs(%alloc_34 : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)  {
    ^unified0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb3: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb4: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb5: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb6: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb7: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb8: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb9: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb10: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>):
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %7 = d2m.wait %cb8 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %8 = d2m.wait %cb9 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %9 = d2m.reserve %cb10 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %10 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %11 = d2m.wait %cb2 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %12 = d2m.wait %cb3 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %13 = d2m.wait %cb4 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %14 = d2m.wait %cb5 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %15 = d2m.wait %cb6 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %16 = d2m.wait %cb7 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %17 = d2m.wait %cb0 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %subview = memref.subview %17[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_37 = memref.subview %10[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_38 = memref.subview %11[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_39 = memref.subview %12[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_40 = memref.subview %13[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_41 = memref.subview %14[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_42 = memref.subview %15[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_43 = memref.subview %16[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_44 = memref.subview %7[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_45 = memref.subview %8[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_46 = memref.subview %9[%arg1, %arg2] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          // CHECK: affine.for
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: %[[V11:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V12:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V13:.*]] = "d2m.tile_div"(%[[V11]], %[[V12]])
          // CHECK-NEXT: %[[V14:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V15:.*]] = "d2m.tile_pow"(%[[V13]], %[[V14]])
          // CHECK-NEXT: %[[V16:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V17:.*]] = "d2m.tile_div"(%[[V15]], %[[V16]])
          // CHECK-NEXT: %[[V18:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V19:.*]] = "d2m.tile_pow"(%[[V17]], %[[V18]])
          // CHECK-NEXT: %[[V20:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V21:.*]] = "d2m.tile_div"(%[[V19]], %[[V20]])
          // CHECK-NEXT: %[[V22:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V23:.*]] = "d2m.tile_pow"(%[[V21]], %[[V22]])
          // CHECK-NEXT: %[[V24:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V25:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V26:.*]] = "d2m.tile_maximum"(%[[V24]], %[[V25]])
          // CHECK-NEXT: %[[V27:.*]] = "d2m.tile_div"(%[[V23]], %[[V26]])
          // CHECK-NEXT: %[[V28:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
          // CHECK-NEXT: %[[V29:.*]] = "d2m.tile_log"(%[[V28]])
          // CHECK-NEXT: %[[V30:.*]] = "d2m.tile_div"(%[[V29]], %[[V27]])
          // CHECK-NEXT: affine.store %[[V30]]
          // CHECK: {d2m.linalg_root, d2m.scheduled}
          affine.for %arg3 = 0 to 1 {
            affine.for %arg4 = 0 to 1 {
              %18 = affine.load %subview[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %19 = affine.load %subview_37[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %20 = affine.load %subview_38[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %21 = affine.load %subview_39[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %22 = affine.load %subview_40[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %23 = affine.load %subview_41[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %24 = affine.load %subview_42[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %25 = affine.load %subview_43[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %26 = affine.load %subview_44[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %27 = affine.load %subview_45[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
              %28 = "d2m.tile_maximum"(%26, %27) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %29 = "d2m.tile_div"(%19, %20) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %30 = "d2m.tile_pow"(%29, %21) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %31 = "d2m.tile_div"(%30, %22) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %32 = "d2m.tile_pow"(%31, %23) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %33 = "d2m.tile_div"(%32, %24) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %34 = "d2m.tile_pow"(%33, %25) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %35 = "d2m.tile_div"(%34, %28) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %36 = "d2m.tile_log"(%18) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %37 = "d2m.tile_div"(%36, %35) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              affine.store %37, %subview_46[%arg3, %arg4] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
            }
          } {d2m.linalg_root}
        }
      }
    }
    return
  }
}
