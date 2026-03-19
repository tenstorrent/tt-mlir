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

// CHECK-LABEL: func.func @single_dst_section_pack_tile_block
func.func @single_dst_section_pack_tile_block() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  // CHECK: ttkernel.tile_regs_acquire
  // CHECK: ttkernel.copy_tile
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  // CHECK: ttkernel.tile_regs_commit
  // CHECK: ttkernel.tile_regs_wait
  // CHECK: ttkernel.pack_tile_block
  ttkernel.pack_tile_block(%c0, %cb, %c4) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  // CHECK: ttkernel.tile_regs_release
  return
}
