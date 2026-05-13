// Verify that with --convert-d2m-to-ttmetal=use-dfbs=true the host-side
// emission produces a ttmetal.create_dataflow_buffer + a
// ttmetal.bind_dfb_to_kernels per CB additional arg, with cardinality
// inferred from which kernel funcs produce/consume each CB. Also verify
// that $dfb_ids on enqueue_program carries the positional ids and the
// existing CB path is unchanged when the flag is off.

// RUN: ttmlir-opt --ttcore-register-device \
// RUN:   --convert-d2m-to-ttkernel="use-dfbs=true" \
// RUN:   --convert-d2m-to-ttmetal="use-dfbs=true" %s | FileCheck %s --check-prefix=DFB

// RUN: ttmlir-opt --ttcore-register-device \
// RUN:   --convert-d2m-to-ttkernel --convert-d2m-to-ttmetal %s | FileCheck %s --check-prefix=CB

#l1_ = #ttcore.memory_space<l1>

module {
  // DFB-LABEL: func.func @eltwise
  // CB-LABEL: func.func @eltwise

  // Input buffer: DM reader (BRISC, mask 0x1) -> compute (TRISC0, mask 0x100).
  // DFB: ttmetal.create_dataflow_buffer
  // DFB-SAME: producer_risc_mask = 1
  // DFB-SAME: consumer_risc_mask = 256
  // DFB: ttmetal.bind_dfb_to_kernels
  // DFB-SAME: consumer_kernel = @compute, producer_kernel = @dm_reader

  // Output buffer: compute (TRISC0, mask 0x100) -> DM writer (NCRISC, mask 0x2).
  // DFB: ttmetal.create_dataflow_buffer
  // DFB-SAME: producer_risc_mask = 256
  // DFB-SAME: consumer_risc_mask = 2
  // DFB: ttmetal.bind_dfb_to_kernels
  // DFB-SAME: consumer_kernel = @dm_writer, producer_kernel = @compute

  // DFB: ttmetal.enqueue_program
  // DFB-SAME: dfb_ids = array<i64: 0, 1>

  // CB: ttmetal.enqueue_program
  // CB-SAME: cb_ports = array<i64: 0, 1>
  // CB-SAME: dfb_ids = array<i64>
  // CB-NOT: ttmetal.create_dataflow_buffer
  // CB-NOT: ttmetal.bind_dfb_to_kernels
  func.func @eltwise(%arg0: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>) -> memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_> {
    %alloc = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>
    %cb_in = memref.alloc() {address = 0x2000 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    %cb_out = memref.alloc() {address = 0x3000 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_reader>, #d2m.thread<compute, @compute>, #d2m.thread<datamovement, @dm_writer>]}
        ins(%arg0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
    additionalArgs(%cb_in, %cb_out : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>)
    return %alloc : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>
  }

  // Reader: produces cb_in.
  func.func private @dm_reader() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb_in = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.reserve %cb_in : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.push %cb_in : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }

  // Compute: consumes cb_in, produces cb_out.
  func.func private @compute() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb_in = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %cb_out = d2m.get_cb(3) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %in = d2m.wait %cb_in : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    %out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.push %cb_out : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    d2m.pop %cb_in : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }

  // Writer: consumes cb_out.
  func.func private @dm_writer() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb_out = d2m.get_cb(3) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    %m = d2m.wait %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>
    d2m.pop %cb_out : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1_>>
    return
  }
}
