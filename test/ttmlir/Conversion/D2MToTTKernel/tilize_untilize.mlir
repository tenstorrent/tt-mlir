// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_tile_tilize_block
  func.func @test_tile_tilize_block(%arg0: !d2m.cb<memref<128x192xf32, #l1_>>, %arg1: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>) -> () attributes {d2m.thread = #d2m.thread<compute>} {
    // CHECK-NOT: d2m.tile_tilize_block
    // CHECK: ttkernel.tilize_init
    // CHECK: ttkernel.experimental::tilize_block
    %in = d2m.pop %arg0 : !d2m.cb<memref<128x192xf32, #l1_>> -> memref<128x192xf32, #l1_>
    %out = d2m.reserve %arg1 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_tilize_block"(%in, %out) : (memref<128x192xf32, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    return
  }

  // CHECK-LABEL: func.func @test_tile_untilize_block
  func.func @test_tile_untilize_block(%arg0: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>, %arg1: !d2m.cb<memref<128x192xf32, #l1_>>) -> () attributes {d2m.thread = #d2m.thread<compute>} {
    // CHECK-NOT: d2m.tile_untilize_block
    // CHECK: ttkernel.untilize_init
    // CHECK: ttkernel.experimental::untilize_block
    %in = d2m.pop %arg0 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %arg1 : !d2m.cb<memref<128x192xf32, #l1_>> -> memref<128x192xf32, #l1_>
    "d2m.tile_untilize_block"(%in, %out) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<128x192xf32, #l1_>) -> ()
    return
  }
}
