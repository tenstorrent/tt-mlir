// RUN: ttmlir-opt --canonicalize %s | FileCheck %s


// CHECK-LABEL: func.func @test_consecutive_read_barriers
func.func @test_consecutive_read_barriers() {
  // CHECK: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  // CHECK-NOT: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_consecutive_write_barriers
func.func @test_consecutive_write_barriers() {
  // CHECK: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  // CHECK-NOT: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_consecutive_read_barriers_same_noc
func.func @test_consecutive_read_barriers_same_noc(%noc: i8) {
  // CHECK: ttkernel.noc_async_read_barrier(%{{.*}})
  ttkernel.noc_async_read_barrier(%noc) : (i8) -> ()
  // CHECK-NOT: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier(%noc) : (i8) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_consecutive_read_barriers_different_noc
func.func @test_consecutive_read_barriers_different_noc(%noc0: i8, %noc1: i8) {
  // CHECK: ttkernel.noc_async_read_barrier(%{{.*}})
  ttkernel.noc_async_read_barrier(%noc0) : (i8) -> ()
  // CHECK: ttkernel.noc_async_read_barrier(%{{.*}})
  ttkernel.noc_async_read_barrier(%noc1) : (i8) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_consecutive_write_barriers_same_noc
func.func @test_consecutive_write_barriers_same_noc(%noc: i8) {
  // CHECK: ttkernel.noc_async_write_barrier(%{{.*}})
  ttkernel.noc_async_write_barrier(%noc) : (i8) -> ()
  // CHECK-NOT: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier(%noc) : (i8) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_consecutive_write_barriers_different_noc
func.func @test_consecutive_write_barriers_different_noc(%noc0: i8, %noc1: i8) {
  // CHECK: ttkernel.noc_async_write_barrier(%{{.*}})
  ttkernel.noc_async_write_barrier(%noc0) : (i8) -> ()
  // CHECK: ttkernel.noc_async_write_barrier(%{{.*}})
  ttkernel.noc_async_write_barrier(%noc1) : (i8) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_consecutive_unpack_stall_on_pack
func.func @test_consecutive_unpack_stall_on_pack() {
  // CHECK: ttkernel.experimental.unpack_stall_on_pack
  "ttkernel.experimental.unpack_stall_on_pack"() : () -> ()
  // CHECK-NOT: ttkernel.experimental.unpack_stall_on_pack
  "ttkernel.experimental.unpack_stall_on_pack"() : () -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_read_then_write_barrier
func.func @test_read_then_write_barrier() {
  // CHECK: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  // CHECK: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_read_barriers_with_push_between
func.func @test_read_barriers_with_push_between(%num_pages: i32) {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<16, !ttcore.tile<32x32, f32>>
  // CHECK: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  ttkernel.cb_push_back(%cb, %num_pages) : (!ttkernel.cb<16, !ttcore.tile<32x32, f32>>, i32) -> ()
  // CHECK-NOT: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  ttkernel.cb_push_back(%cb, %num_pages) : (!ttkernel.cb<16, !ttcore.tile<32x32, f32>>, i32) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_read_barrier_with_intervening_read
func.func @test_read_barrier_with_intervening_read(
    %x: index, %y: index, %remote_addr: i32, %l1_addr: i32, %size: i32) {
  // CHECK: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  ttkernel.noc_async_read core[%x, %y], %remote_addr, %l1_addr, %size : (index, index, i32, i32, i32) -> ()
  // CHECK: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_write_barrier_with_intervening_write
func.func @test_write_barrier_with_intervening_write(
    %l1_addr: i32, %x: index, %y: index, %remote_addr: i32, %size: i32) {
  // CHECK: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  ttkernel.noc_async_write %l1_addr, core[%x, %y], %remote_addr, %size : (i32, index, index, i32, i32) -> ()
  // CHECK: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_write_barrier_with_intervening_inline_write
func.func @test_write_barrier_with_intervening_inline_write(
    %x: index, %y: index, %dst_addr: i32, %val: i32, %byte_enable: i8, %noc_id: i8) {
  // CHECK: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  ttkernel.noc_inline_dw_write(core[%x, %y], %dst_addr, %val, %byte_enable, noc %noc_id) : (index, index, i32, i32, i8, i8) -> ()
  // CHECK: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_mixed_consecutive_barriers
func.func @test_mixed_consecutive_barriers() {
  // CHECK: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  // CHECK-NOT: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  // CHECK: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  // CHECK-NOT: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  // CHECK: return
  return
}
