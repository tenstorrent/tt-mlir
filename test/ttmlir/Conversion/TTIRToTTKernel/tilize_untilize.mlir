// RUN: ttmlir-opt --ttcore-register-device --convert-ttir-to-ttkernel -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_tile_tilize_block
  func.func @test_tile_tilize_block(%arg0: memref<128x192xf32, #l1_>, %arg1: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> () {
    ttir.await %arg0 : (memref<128x192xf32, #l1_>)
    // CHECK-NOT: ttir.tile_tilize_block
    // CHECK: ttkernel.tilize_init
    // CHECK: ttkernel.experimental::tilize_block
    "ttir.tile_tilize_block"(%arg0, %arg1) : (memref<128x192xf32, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    ttir.yield %arg1 : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_tile_untilize_block
  func.func @test_tile_untilize_block(%arg0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<128x192xf32, #l1_>) -> () {
    ttir.await %arg0 : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>)
    // CHECK-NOT: ttir.tile_untilize_block
    // CHECK: ttkernel.untilize_init
    // CHECK: ttkernel.experimental::untilize_block
    "ttir.tile_untilize_block"(%arg0, %arg1) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<128x192xf32, #l1_>) -> ()
    ttir.yield %arg1 : (memref<128x192xf32, #l1_>)
    return
  }
}
