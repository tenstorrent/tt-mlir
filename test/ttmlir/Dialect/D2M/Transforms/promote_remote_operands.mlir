// RUN: ttmlir-opt --ttcore-register-device --d2m-promote-remote-operands %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // Test 1: Triton-style placeholder outs, real I/O in additionalArgs.
  // %argA is loaded, %argB is loaded, %argC is stored. After promotion they
  // move into ins/outs and the now-duplicate refs in additionalArgs are dropped.
  // %scalar stays in additionalArgs.
  // CHECK-LABEL: func.func @test_promote
  func.func @test_promote(
      %argA: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>,
      %argB: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>,
      %argC: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>,
      %scalar: i32) {
    // CHECK: d2m.generic
    // CHECK: ins(%{{.*}}, %{{.*}} : memref{{.*}}#dram>, memref{{.*}}#dram>)
    // CHECK: outs(%{{.*}} : memref{{.*}}#dram>)
    // CHECK: additionalArgs(%{{.*}} : i32)
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins()
        outs(%argA : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>)
        additionalArgs(%argA, %argB, %argC, %scalar : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>, i32) {
      %c0 = arith.constant 0 : index
      %a = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #l1>
      %b = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #l1>
      %c = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #l1>
      d2m.remote_load %a %argA[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f16>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>
      d2m.remote_load %b %argB[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f16>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>
      d2m.remote_store %argC[%c0, %c0] %c : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>, memref<2x2x!ttcore.tile<32x32, f16>, #l1>
    }
    return
  }

  // Test 2: Already-canonical generic is left alone (no spurious rewrites).
  // CHECK-LABEL: func.func @test_canonical_noop
  func.func @test_canonical_noop(
      %src: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>,
      %dst: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>) {
    // CHECK: d2m.generic
    // CHECK: ins(%{{.*}} : memref{{.*}}#dram>)
    // CHECK: outs(%{{.*}} : memref{{.*}}#dram>)
    // CHECK-NOT: additionalArgs
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%src : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>)
        outs(%dst : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>) {
      %c0 = arith.constant 0 : index
      %buf = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #l1>
      d2m.remote_load %buf %src[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f16>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>
      d2m.remote_store %dst[%c0, %c0] %buf : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>, memref<2x2x!ttcore.tile<32x32, f16>, #l1>
    }
    return
  }

  // Test 3: Multiple distinct store targets — can't unambiguously normalize
  // (verifier needs exactly one outs operand), so the pass leaves the op alone.
  // CHECK-LABEL: func.func @test_multiple_stores
  func.func @test_multiple_stores(
      %argA: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>,
      %argB: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>,
      %argC: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>) {
    // CHECK: d2m.generic
    // CHECK: additionalArgs(%{{.*}}, %{{.*}}, %{{.*}} : memref{{.*}}, memref{{.*}}, memref{{.*}})
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins()
        outs(%argA : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>)
        additionalArgs(%argA, %argB, %argC : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>) {
      %c0 = arith.constant 0 : index
      %a = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #l1>
      d2m.remote_load %a %argA[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f16>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>
      d2m.remote_store %argB[%c0, %c0] %a : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>, memref<2x2x!ttcore.tile<32x32, f16>, #l1>
      d2m.remote_store %argC[%c0, %c0] %a : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #dram>, memref<2x2x!ttcore.tile<32x32, f16>, #l1>
    }
    return
  }
}
