// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" %s | ttmlir-translate --ttmetal-to-flatbuffer > %t.ttm
// RUN: xxd %t.ttm | FileCheck %s --check-prefix=CHECK-HEX
// RUN: strings %t.ttm | FileCheck %s --check-prefix=CHECK-STR

// Verify flatbuffer generation handles memref with TileType elements.
// Function arguments are memref<...x!ttcore.tile<32x32, f32>> after bufferization.

#l1 = #ttcore.memory_space<l1>

// CHECK-HEX: TTM0
// CHECK-STR: func.func @test_tiletype(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>
module {
  func.func @test_tiletype(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1> {
    return %arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1>
  }

  func.func private @kernel() attributes {
    ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 0>]>,
    ttkernel.thread = #ttkernel.thread<compute>
  } {
    return
  }
}
