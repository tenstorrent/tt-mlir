// Strict IR test for per-generic 1P→NC and NP→1C DFB lowering.
//
// Structurally inspired by the matmul-with-compute-thread-tiling output
// (a d2m.generic with two DM readers on noc=0 and noc=1 plus a compute
// region carrying `num_threads_per_cluster = 4`), but with shapes
// chosen so the addressing math is easy to read and pin down by hand:
//
//   risc mask math
//     DM noc=0   : 1 << 0          = 0x001 = 1
//     DM noc=1   : 1 << 1          = 0x002 = 2
//     Compute N  : ((1<<N)-1) << 8
//                    N=1 → 0x100 = 256
//                    N=2 → 0x300 = 768
//                    N=4 → 0xF00 = 3840
//
//   pattern inference (consumer side, multi-consumer DFB)
//     CB volume * num_consumers == full operand volume → strided
//     CB volume == full operand volume                 → all
//
// Per-hart NOC addressing inside kernels is produced by the existing
// d2m.my_thread_id SPMD path and is not the subject of these tests.

// RUN: ttmlir-opt --ttcore-register-device \
// RUN:   --convert-d2m-to-ttkernel="use-dfbs=true" \
// RUN:   --convert-d2m-to-ttmetal="use-dfbs=true" %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>

module {
  //===--------------------------------------------------------------===//
  // Case 1: matmul-shaped 1P→4C with one STRIDED input and one ALL input.
  //
  // - thread[0]: DM reader noc=0  → produces CB %cb_a (A stripe, STRIDED)
  // - thread[1]: DM reader noc=1  → produces CB %cb_b (B broadcast, ALL)
  // - thread[2]: compute N=4      → consumes both (single kernel symbol)
  //
  // A: full operand 4 tiles, CB 1 tile → ratio 4 == num_consumers → STRIDED
  // B: full operand 1 tile,  CB 1 tile → ratio 1                  → ALL
  //
  // Expected DFB A:
  //   num_producers = 1, num_consumers = 4
  //   producer_risc_mask = 1    (BRISC, noc=0)
  //   consumer_risc_mask = 3840 (TRISC0..3)
  //   producer_pattern = strided, consumer_pattern = strided
  //
  // Expected DFB B:
  //   num_producers = 1, num_consumers = 4
  //   producer_risc_mask = 2    (NCRISC, noc=1)
  //   consumer_risc_mask = 3840
  //   producer_pattern = strided, consumer_pattern = all
  //===--------------------------------------------------------------===//

  // CHECK-LABEL: func.func @matmul_shape_4ct
  // First DFB emitted (DM reader noc=0 → compute): STRIDED input.
  // CHECK: ttmetal.create_dataflow_buffer
  // CHECK-SAME: num_producers = 1
  // CHECK-SAME: num_consumers = 4
  // CHECK-SAME: producer_risc_mask = 1
  // CHECK-SAME: consumer_risc_mask = 3840
  // CHECK-SAME: producer_pattern = <strided>
  // CHECK-SAME: consumer_pattern = <strided>
  // CHECK: ttmetal.bind_dfb_to_kernels
  // CHECK-SAME: consumer_kernel = @compute_4ct
  // CHECK-SAME: producer_kernel = @dm_reader_a
  // Second DFB (DM reader noc=1 → compute): ALL (broadcast).
  // CHECK: ttmetal.create_dataflow_buffer
  // CHECK-SAME: num_producers = 1
  // CHECK-SAME: num_consumers = 4
  // CHECK-SAME: producer_risc_mask = 2
  // CHECK-SAME: consumer_risc_mask = 3840
  // CHECK-SAME: producer_pattern = <strided>
  // CHECK-SAME: consumer_pattern = <all>
  // CHECK: ttmetal.bind_dfb_to_kernels
  // CHECK-SAME: consumer_kernel = @compute_4ct
  // CHECK-SAME: producer_kernel = @dm_reader_b
  // ComputeConfigAttr carries the per-generic N=4. The default 1 is
  // elided by the custom printer so the presence of the field is the
  // positive signal.
  // CHECK: ttmetal.enqueue_program
  // dfb_ids positional 0, 1 in emission order.
  // CHECK-SAME: dfb_ids = array<i64: 0, 1>
  // CHECK-SAME: #ttmetal.compute_config<@compute_4ct
  // CHECK-SAME: num_threads_per_cluster = 4
  func.func @matmul_shape_4ct(%arg_a: memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>, %arg_b: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    %alloc_out = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>
    // A: full operand 4 tiles, CB 1 tile → 4× ratio → STRIDED.
    %cb_a = memref.alloc() {address = 0x2000 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    // B: full operand 1 tile, CB 1 tile → 1× ratio → ALL.
    %cb_b = memref.alloc() {address = 0x3000 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_reader_a, noc = 0>, #d2m.thread<datamovement, @dm_reader_b, noc = 1>, #d2m.thread<compute, @compute_4ct, num_threads_per_cluster = 4>]}
        ins(%arg_a, %arg_b : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%alloc_out : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
    additionalArgs(%cb_a, %cb_b : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>)
    return
  }
  // Each reader produces exactly one CB; classifyCBRole follows the
  // get_cb operand_idx to mark the producer side. operand_idx 3 = cb_a,
  // operand_idx 4 = cb_b (after ins=2 + outs=1).
  func.func private @dm_reader_a() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(3) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.reserve %cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.push %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }
  func.func private @dm_reader_b() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(4) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.reserve %cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.push %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }
  // The compute kernel consumes both CBs; it is the sole consumer kernel
  // for both DFBs (matmul-style fan-in). After D2MMaterializeComputeThreadForall
  // SPMDs the forall, this kernel body is a single binary executed by
  // all 4 compute harts.
  func.func private @compute_4ct() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb_a = d2m.get_cb(3) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %cb_b = d2m.get_cb(4) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %ma = d2m.wait %cb_a : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    %mb = d2m.wait %cb_b : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.pop %cb_a : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    d2m.pop %cb_b : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }

  //===--------------------------------------------------------------===//
  // Case 2: Matmul-output-shaped 4P→1C STRIDED.
  //
  // Each compute hart writes its own M-slice of the output; the DM
  // writer pops linearly.
  //
  // num_producers      = 4    (compute N=4)
  // num_consumers      = 1    (DM writer)
  // producer_risc_mask = 3840 (TRISC0..3)
  // consumer_risc_mask = 1    (BRISC, noc=0 auto-assigned)
  // producer_pattern   = strided (matmul output interleave)
  // consumer_pattern   = strided (degenerate single consumer)
  //===--------------------------------------------------------------===//

  // CHECK-LABEL: func.func @matmul_output_4ct
  // CHECK: ttmetal.create_dataflow_buffer
  // CHECK-SAME: num_producers = 4
  // CHECK-SAME: num_consumers = 1
  // CHECK-SAME: producer_risc_mask = 3840
  // CHECK-SAME: consumer_risc_mask = 1
  // CHECK-SAME: producer_pattern = <strided>
  // CHECK-SAME: consumer_pattern = <strided>
  // CHECK: ttmetal.bind_dfb_to_kernels
  // CHECK-SAME: consumer_kernel = @dm_writer
  // CHECK-SAME: producer_kernel = @compute_producer_4ct
  func.func @matmul_output_4ct(%arg_in: memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    %alloc_out = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>
    // Output CB stripe = 1 tile; full output = 4 tiles → 4× ratio → producer STRIDED.
    %cb_out = memref.alloc() {address = 0x2000 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute, @compute_producer_4ct, num_threads_per_cluster = 4>, #d2m.thread<datamovement, @dm_writer>]}
        ins(%arg_in : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%alloc_out : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
    additionalArgs(%cb_out : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>)
    return
  }
  func.func private @compute_producer_4ct() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.reserve %cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.push %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }
  func.func private @dm_writer() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.wait %cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.pop %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }

  //===--------------------------------------------------------------===//
  // Case 3: Per-generic N is read independently — two generics with
  // different N values in the same module, each emits its own
  // cardinality and risc_mask.
  //
  // First generic: N=4 → consumer_risc_mask = 3840
  // Second generic: N=2 → consumer_risc_mask = 768 ((1<<2-1)<<8)
  //===--------------------------------------------------------------===//

  // CHECK-LABEL: func.func @two_generics_different_N
  // CHECK: ttmetal.create_dataflow_buffer
  // CHECK-SAME: num_consumers = 4
  // CHECK-SAME: consumer_risc_mask = 3840
  // CHECK: ttmetal.enqueue_program
  // CHECK-SAME: num_threads_per_cluster = 4
  // CHECK: ttmetal.create_dataflow_buffer
  // CHECK-SAME: num_consumers = 2
  // CHECK-SAME: consumer_risc_mask = 768
  // CHECK: ttmetal.enqueue_program
  // CHECK-SAME: num_threads_per_cluster = 2
  func.func @two_generics_different_N(%arg0: memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    %alloc1 = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>
    %cb_n4 = memref.alloc() {address = 0x2000 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_reader_n4>, #d2m.thread<compute, @compute_n4, num_threads_per_cluster = 4>]}
        ins(%arg0 : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%alloc1 : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
    additionalArgs(%cb_n4 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>)

    %alloc2 = memref.alloc() {alignment = 64 : i64, address = 0x3000} : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>
    // N=2: CB stripe is 2 tiles, full operand 4 tiles → ratio 2 → STRIDED.
    %cb_n2 = memref.alloc() {address = 0x4000 : i64, alignment = 16 : i64} : memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_reader_n2>, #d2m.thread<compute, @compute_n2, num_threads_per_cluster = 2>]}
        ins(%arg0 : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%alloc2 : memref<1x1x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
    additionalArgs(%cb_n2 : memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>)
    return
  }
  func.func private @dm_reader_n4() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.reserve %cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.push %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }
  func.func private @compute_n4() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.wait %cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.pop %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }
  func.func private @dm_reader_n2() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(2) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.reserve %cb : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.push %cb : <memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }
  func.func private @compute_n2() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb = d2m.get_cb(2) : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.wait %cb : !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.pop %cb : <memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }
}
