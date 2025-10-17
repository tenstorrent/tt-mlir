// RUN: ttmlir-opt --ttcore-register-device --lower-affine --d2m-generic-linearize-memref --lower-affine --convert-d2m-to-ttkernel --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // Test that D2MToTTKernel can see through dst_reinterpret_cast
  // when lowering copy_tile and pack_tile operations around typecast

  // CHECK-LABEL: func.func @test_typecast_f32_to_bf16
  func.func @test_typecast_f32_to_bf16(%arg0: memref<1x8x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x8x!ttcore.tile<32x32, bf16>, #l1_>) attributes {d2m.thread = #d2m.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %dst = "d2m.acquire_dst"() : () -> memref<1x1x8x!ttcore.tile<32x32, f32>, #dst>

    // Loop 1: Copy from L1 to dst
    scf.for %i = %c0 to %c1 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        %0 = memref.load %arg0[%i, %j] : memref<1x8x!ttcore.tile<32x32, f32>, #l1_>
        memref.store %0, %dst[%c0, %i, %j] : memref<1x1x8x!ttcore.tile<32x32, f32>, #dst>
      }
    }

    // Loop 2: Typecast in dst with reinterpret cast to store back
    scf.for %i = %c0 to %c1 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        %0 = memref.load %dst[%c0, %i, %j] : memref<1x1x8x!ttcore.tile<32x32, f32>, #dst>
        %1 = "d2m.tile_typecast"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, bf16>
        // This reinterpret cast allows storing bf16 result back to f32-typed dst
        %2 = "d2m.dst_reinterpret_cast"(%1) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, f32>
        memref.store %2, %dst[%c0, %i, %j] : memref<1x1x8x!ttcore.tile<32x32, f32>, #dst>
      }
    }

    // Loop 3: Copy from dst to L1 with reinterpret cast
    scf.for %i = %c0 to %c1 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        %0 = memref.load %dst[%c0, %i, %j] : memref<1x1x8x!ttcore.tile<32x32, f32>, #dst>
        // This reinterpret cast converts back to bf16 for L1 store
        %1 = "d2m.dst_reinterpret_cast"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, bf16>
        memref.store %1, %arg1[%i, %j] : memref<1x8x!ttcore.tile<32x32, bf16>, #l1_>
      }
    }

    // CHECK-NOT: d2m.dst_reinterpret_cast
    // CHECK: ttkernel.tile_regs_acquire
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.typecast_tile_init
    // CHECK: ttkernel.typecast_tile(%{{.*}}, <f32>, <bf16>)
    // CHECK: ttkernel.pack_tile
    return
  }
}
