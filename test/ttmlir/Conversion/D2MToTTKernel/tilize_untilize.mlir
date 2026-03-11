// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_tile_tilize_block
  func.func @test_tile_tilize_block() attributes {d2m.thread = #d2m.thread<compute>} {
    %arg0 = d2m.get_cb(0) : !d2m.cb<memref<128x192xf32, #l1_>>
    %arg1 = d2m.get_cb(1) : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>
    // CHECK-NOT: d2m.tile_tilize_block
    // CHECK: ttkernel.tilize_init
    // CHECK: ttkernel.experimental::tilize_block
    %in = d2m.wait %arg0 : !d2m.cb<memref<128x192xf32, #l1_>> -> memref<128x192xf32, #l1_>
    %out = d2m.reserve %arg1 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    %result = "d2m.tile_tilize_block"(%in, %out) : (memref<128x192xf32, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_tile_untilize_block
  func.func @test_tile_untilize_block() attributes {d2m.thread = #d2m.thread<compute>} {
    %arg0 = d2m.get_cb(0) : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>
    %arg1 = d2m.get_cb(1) : !d2m.cb<memref<128x192xf32, #l1_>>
    // CHECK-NOT: d2m.tile_untilize_block
    // CHECK: ttkernel.pack_untilize_init
    // CHECK: ttkernel.experimental::pack_untilize_block
    // CHECK: ttkernel.pack_untilize_uninit
    %in = d2m.wait %arg0 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %arg1 : !d2m.cb<memref<128x192xf32, #l1_>> -> memref<128x192xf32, #l1_>
    %result = "d2m.tile_untilize_block"(%in, %out) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<128x192xf32, #l1_>) -> memref<128x192xf32, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_tile_tilize_block_i32
  func.func @test_tile_tilize_block_i32() attributes {d2m.thread = #d2m.thread<compute>} {
    %arg0 = d2m.get_cb(0) : !d2m.cb<memref<128x192xi32, #l1_>>
    %arg1 = d2m.get_cb(1) : !d2m.cb<memref<4x6x!ttcore.tile<32x32, si32>, #l1_>>
    // CHECK-NOT: d2m.tile_tilize_block
    // CHECK: ttkernel.tilize_init
    // CHECK: ttkernel.experimental::tilize_block
    %in = d2m.wait %arg0 : !d2m.cb<memref<128x192xi32, #l1_>> -> memref<128x192xi32, #l1_>
    %out = d2m.reserve %arg1 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, si32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, si32>, #l1_>
    %result = "d2m.tile_tilize_block"(%in, %out) : (memref<128x192xi32, #l1_>, memref<4x6x!ttcore.tile<32x32, si32>, #l1_>) -> memref<4x6x!ttcore.tile<32x32, si32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_tile_untilize_block_i32
  func.func @test_tile_untilize_block_i32() attributes {d2m.thread = #d2m.thread<compute>} {
    %arg0 = d2m.get_cb(0) : !d2m.cb<memref<4x6x!ttcore.tile<32x32, si32>, #l1_>>
    %arg1 = d2m.get_cb(1) : !d2m.cb<memref<128x192xi32, #l1_>>
    // CHECK-NOT: d2m.tile_untilize_block
    // CHECK: ttkernel.pack_untilize_init
    // CHECK: ttkernel.experimental::pack_untilize_block
    // CHECK: ttkernel.pack_untilize_uninit
    %in = d2m.wait %arg0 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, si32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, si32>, #l1_>
    %out = d2m.reserve %arg1 : !d2m.cb<memref<128x192xi32, #l1_>> -> memref<128x192xi32, #l1_>
    %result = "d2m.tile_untilize_block"(%in, %out) : (memref<4x6x!ttcore.tile<32x32, si32>, #l1_>, memref<128x192xi32, #l1_>) -> memref<128x192xi32, #l1_>
    return
  }
}
