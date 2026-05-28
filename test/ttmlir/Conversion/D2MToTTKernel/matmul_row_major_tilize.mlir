// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Verifies the row-major-fed matmul lowering path:
//   * `d2m.tile_tilize_block` writes a row-major source CB into a tiled
//     scratch CB.
//   * `d2m.tile_matmul` consumes the tiled scratch CB.
// The lowering must emit `ttkernel.mm_init_short_with_dt` (not the plain
// `mm_init_short`) so the unpacker srcA data format is reconfigured back
// from the row-major source to the tiled scratch before `matmul_tiles`.

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @row_major_fed_matmul
  func.func @row_major_fed_matmul() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb_rm  = d2m.get_cb(0) : !d2m.cb<memref<128x96xf32, #l1_>>
    %cb_a   = d2m.get_cb(1) : !d2m.cb<memref<4x3x!ttcore.tile<32x32, f32>, #l1_>>
    %cb_b   = d2m.get_cb(2) : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>>
    %cb_out = d2m.get_cb(3) : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>

    %rm = d2m.wait %cb_rm : !d2m.cb<memref<128x96xf32, #l1_>> -> memref<128x96xf32, #l1_>
    %a  = d2m.reserve %cb_a : !d2m.cb<memref<4x3x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x3x!ttcore.tile<32x32, f32>, #l1_>
    %tilized = "d2m.tile_tilize_block"(%rm, %a) : (memref<128x96xf32, #l1_>, memref<4x3x!ttcore.tile<32x32, f32>, #l1_>) -> memref<4x3x!ttcore.tile<32x32, f32>, #l1_>

    %b   = d2m.wait %cb_b : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %cb_out : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f32>, #l1_>
    %dst = d2m.acquire_dst() : memref<8x!ttcore.tile<32x32, f32>, #dst_>

    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 2 {
        affine.for %k = 0 to 3 {
          %va = affine.load %a[%i, %k] : memref<4x3x!ttcore.tile<32x32, f32>, #l1_>
          %vb = affine.load %b[%k, %j] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
          %vc = affine.load %dst[%i * 2 + %j] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
          %r = "d2m.tile_matmul"(%va, %vb, %vc) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %r, %dst[%i * 2 + %j] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
        }
      }
    }

    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 2 {
        %vd = affine.load %dst[%i * 2 + %j] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
        affine.store %vd, %out[%i, %j] : memref<4x2x!ttcore.tile<32x32, f32>, #l1_>
      }
    }

    // CHECK: ttkernel.tilize_init
    // CHECK: ttkernel.experimental::tilize_block
    // CHECK: ttkernel.mm_init_short_with_dt
    // CHECK: ttkernel.matmul_tiles
    return
  }

  // CHECK-LABEL: func.func @plain_matmul
  // No upstream tilize → keep the cheap `mm_init_short`.
  func.func @plain_matmul() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb_a   = d2m.get_cb(0) : !d2m.cb<memref<4x3x!ttcore.tile<32x32, f32>, #l1_>>
    %cb_b   = d2m.get_cb(1) : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>>
    %cb_out = d2m.get_cb(2) : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>

    %a   = d2m.wait %cb_a : !d2m.cb<memref<4x3x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x3x!ttcore.tile<32x32, f32>, #l1_>
    %b   = d2m.wait %cb_b : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %cb_out : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f32>, #l1_>
    %dst = d2m.acquire_dst() : memref<8x!ttcore.tile<32x32, f32>, #dst_>

    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 2 {
        affine.for %k = 0 to 3 {
          %va = affine.load %a[%i, %k] : memref<4x3x!ttcore.tile<32x32, f32>, #l1_>
          %vb = affine.load %b[%k, %j] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
          %vc = affine.load %dst[%i * 2 + %j] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
          %r = "d2m.tile_matmul"(%va, %vb, %vc) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %r, %dst[%i * 2 + %j] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
        }
      }
    }

    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 2 {
        %vd = affine.load %dst[%i * 2 + %j] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
        affine.store %vd, %out[%i, %j] : memref<4x2x!ttcore.tile<32x32, f32>, #l1_>
      }
    }

    // CHECK-NOT: ttkernel.mm_init_short_with_dt
    // CHECK: ttkernel.mm_init_short(
    // CHECK: ttkernel.matmul_tiles
    return
  }
}
