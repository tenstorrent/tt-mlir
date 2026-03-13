// RUN: ttmlir-opt --ttkernel-control-dst-section %s | FileCheck %s

// CHECK-LABEL: func.func @single_dst_section
func.func @single_dst_section() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  // CHECK: ttkernel.tile_regs_acquire
  // CHECK: ttkernel.copy_tile
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  // CHECK: ttkernel.tile_regs_commit
  // CHECK: ttkernel.tile_regs_wait
  // CHECK: ttkernel.pack_tile
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  // CHECK: ttkernel.tile_regs_release
  return
}

// CHECK-LABEL: func.func @multiple_flat_dst_sections
func.func @multiple_flat_dst_sections() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: ttkernel.tile_regs_acquire
  // CHECK: ttkernel.copy_tile
  // CHECK: ttkernel.copy_tile
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.copy_tile(%cb, %c0, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  // CHECK: ttkernel.tile_regs_commit
  // CHECK: ttkernel.tile_regs_wait
  // CHECK: ttkernel.pack_tile
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  // CHECK: ttkernel.tile_regs_release

  // CHECK: ttkernel.tile_regs_acquire
  // CHECK: ttkernel.copy_tile
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  // CHECK: ttkernel.tile_regs_commit
  // CHECK: ttkernel.tile_regs_wait
  // CHECK: ttkernel.pack_tile
  ttkernel.pack_tile(%c0, %cb, %c1, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  // CHECK: ttkernel.tile_regs_release
  return
}

// CHECK-LABEL: func.func @nested_dst_section_in_loop
func.func @nested_dst_section_in_loop() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // CHECK: ttkernel.tile_regs_acquire
    // CHECK: ttkernel.copy_tile
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
    // CHECK: ttkernel.tile_regs_commit
    // CHECK: ttkernel.tile_regs_wait
    // CHECK: ttkernel.pack_tile
    ttkernel.pack_tile(%c0, %cb, %i, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
    // CHECK: ttkernel.tile_regs_release
  }
  return
}

// CHECK-LABEL: func.func @pack_between_compute_ops
// Tests the "one tile fused" case with packing between each compute op:
// a single tile_regs_acquire covers two compute+pack sequences (exp → pack,
// then recip → pack). Verifies that tile_regs_commit/wait are inserted before
// each pack_tile and tile_regs_release is inserted after each.
func.func @pack_between_compute_ops() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_in   = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb_out0 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb_out1 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  // CHECK: ttkernel.tile_regs_acquire
  // CHECK: ttkernel.copy_tile
  // CHECK: ttkernel.exp_tile
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb_in, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  // First pack: commit/wait inserted before, release after.
  // CHECK: ttkernel.tile_regs_commit
  // CHECK: ttkernel.tile_regs_wait
  // CHECK: ttkernel.pack_tile
  // CHECK: ttkernel.tile_regs_release
  ttkernel.pack_tile(%c0, %cb_out0, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  // Second compute op on the same DST slot.
  ttkernel.tile_regs_acquire() : () -> ()
  // CHECK: ttkernel.tile_regs_acquire
  // CHECK: ttkernel.recip_tile
  ttkernel.recip_tile(%c0) : (index) -> ()
  // Second pack: commit/wait inserted before, release after.
  // CHECK: ttkernel.tile_regs_commit
  // CHECK: ttkernel.tile_regs_wait
  // CHECK: ttkernel.pack_tile
  // CHECK: ttkernel.tile_regs_release
  ttkernel.pack_tile(%c0, %cb_out1, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  return
}

// CHECK-LABEL: func.func @mixed_flat_and_nested
func.func @mixed_flat_and_nested() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // CHECK: ttkernel.tile_regs_acquire
  // CHECK: ttkernel.copy_tile
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  // CHECK: ttkernel.tile_regs_commit
  // CHECK: ttkernel.tile_regs_wait
  // CHECK: ttkernel.pack_tile
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  // CHECK: ttkernel.tile_regs_release

  scf.for %i = %c0 to %c2 step %c1 {
    // CHECK: ttkernel.tile_regs_acquire
    // CHECK: ttkernel.copy_tile
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
    // CHECK: ttkernel.tile_regs_commit
    // CHECK: ttkernel.tile_regs_wait
    // CHECK: ttkernel.pack_tile
    ttkernel.pack_tile(%c0, %cb, %i, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
    // CHECK: ttkernel.tile_regs_release
  }

  // CHECK: ttkernel.tile_regs_acquire
  // CHECK: ttkernel.copy_tile
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  // CHECK: ttkernel.tile_regs_commit
  // CHECK: ttkernel.tile_regs_wait
  // CHECK: ttkernel.pack_tile
  ttkernel.pack_tile(%c0, %cb, %c1, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  // CHECK: ttkernel.tile_regs_release
  return
}
