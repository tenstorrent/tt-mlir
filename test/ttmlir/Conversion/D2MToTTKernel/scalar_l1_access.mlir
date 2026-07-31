// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

// A scalar (non-tile) read of an L1 circular buffer inside a datamovement thread
// is a real RISC-V read: it lowers to a reinterpret_cast of the CB base address
// plus an element subscript. This is what makes a dependent load -- an address
// computed from data resident in L1 -- expressible.

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

module {
  // CHECK-LABEL: func.func private @dependent_load
  func.func private @dependent_load() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %idx_cb = d2m.get_cb(0) : !d2m.cb<memref<32xi32, #l1>>
    %out_cb = d2m.get_cb(1) : !d2m.cb<memref<8x!ttcore.tile<32x32, f32>, #l1>>
    %src = d2m.get_arg(2) resolution_stage = compile : memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>

    %idx = d2m.wait %idx_cb : !d2m.cb<memref<32xi32, #l1>> -> memref<32xi32, #l1>
    %dst = d2m.reserve %out_cb : !d2m.cb<memref<8x!ttcore.tile<32x32, f32>, #l1>> -> memref<8x!ttcore.tile<32x32, f32>, #l1>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index

    // The loaded index feeds the DRAM address of the NoC read.
    // CHECK: %[[BASE:[0-9]+]] = ttkernel.get_read_ptr(%{{[0-9]+}}) : (!ttkernel.cb<32, i32>) -> i32
    // CHECK: %[[PTR:[0-9]+]] = ttkernel.reinterpret_cast(%[[BASE]]) : (i32) -> !ttkernel.l1_addr_ptr
    // CHECK: %[[OFF:[0-9]+]] = arith.index_cast %{{[a-z0-9]+}} : index to i32
    // CHECK: %[[VAL:[0-9]+]] = ttkernel.load_from_l1(%[[PTR]], %[[OFF]]) : (!ttkernel.l1_addr_ptr, i32) -> i32
    // CHECK: %[[ROW:[0-9]+]] = arith.index_cast %[[VAL]] : i32 to index
    // CHECK: ttkernel.noc_async_read
    scf.for %i = %c0 to %c8 step %c1 {
      %v = memref.load %idx[%i] : memref<32xi32, #l1>
      %row = arith.index_cast %v : i32 to index
      %tx = d2m.dma_read %src[%c0, %c0, %row], %dst[%i], <1> : (memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<8x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
    }
    return
  }

  // 16-bit elements select the narrow tt_l1_ptr flavor.
  // CHECK-LABEL: func.func private @scalar_load_i16
  func.func private @scalar_load_i16() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<64xi16, #l1>>
    %out_cb = d2m.get_cb(1) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %src = d2m.get_arg(2) resolution_stage = compile : memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %buf = d2m.wait %cb : !d2m.cb<memref<64xi16, #l1>> -> memref<64xi16, #l1>
    %dst = d2m.reserve %out_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index

    // CHECK: %[[PTR16:[0-9]+]] = ttkernel.reinterpret_cast(%{{[0-9]+}}) : (i32) -> !ttkernel.l1_addr_ptr<16>
    // CHECK: ttkernel.load_from_l1(%[[PTR16]], %{{[0-9]+}}) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
    %v = memref.load %buf[%c5] : memref<64xi16, #l1>
    %row = arith.index_cast %v : i16 to index
    %tx = d2m.dma_read %src[%c0, %c0, %row], %dst[%c0], <1> : (memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
    d2m.dma_wait %tx : !d2m.mem_tx<read>
    return
  }

  // 8-bit elements likewise.
  // CHECK-LABEL: func.func private @scalar_load_i8
  func.func private @scalar_load_i8() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<64xi8, #l1>>
    %out_cb = d2m.get_cb(1) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %src = d2m.get_arg(2) resolution_stage = compile : memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %buf = d2m.wait %cb : !d2m.cb<memref<64xi8, #l1>> -> memref<64xi8, #l1>
    %dst = d2m.reserve %out_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index

    // CHECK: %[[PTR8:[0-9]+]] = ttkernel.reinterpret_cast(%{{[0-9]+}}) : (i32) -> !ttkernel.l1_addr_ptr<8>
    // CHECK: ttkernel.load_from_l1(%[[PTR8]], %{{[0-9]+}}) : (!ttkernel.l1_addr_ptr<8>, i32) -> i8
    %v = memref.load %buf[%c3] : memref<64xi8, #l1>
    %row = arith.index_cast %v : i8 to index
    %tx = d2m.dma_read %src[%c0, %c0, %row], %dst[%c0], <1> : (memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
    d2m.dma_wait %tx : !d2m.mem_tx<read>
    return
  }

  // Multi-dimensional accesses linearize to a single element offset: 2*8+3 == 19.
  // CHECK-LABEL: func.func private @scalar_load_2d
  func.func private @scalar_load_2d() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<4x8xi32, #l1>>
    %out_cb = d2m.get_cb(1) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %src = d2m.get_arg(2) resolution_stage = compile : memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %buf = d2m.wait %cb : !d2m.cb<memref<4x8xi32, #l1>> -> memref<4x8xi32, #l1>
    %dst = d2m.reserve %out_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // CHECK: %[[LIN:[0-9]+]] = arith.index_cast %{{.*}} : index to i32
    // CHECK: ttkernel.load_from_l1(%{{[0-9]+}}, %[[LIN]])
    %v = memref.load %buf[%c2, %c3] : memref<4x8xi32, #l1>
    %row = arith.index_cast %v : i32 to index
    %tx = d2m.dma_read %src[%c0, %c0, %row], %dst[%c0], <1> : (memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
    d2m.dma_wait %tx : !d2m.mem_tx<read>
    return
  }
}

// The tile-granular meaning of memref.load in a *compute* region -- where the
// load materializes a CB slot index rather than reading a value -- is unchanged;
// see get_dst_idx.mlir and typecast_with_dst_reinterpret_cast.mlir.
// Unsupported shapes are covered by scalar_l1_access_invalid.mlir.
