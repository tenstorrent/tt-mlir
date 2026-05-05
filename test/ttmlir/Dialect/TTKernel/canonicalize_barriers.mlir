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

// CHECK-LABEL: func.func @test_consecutive_unpack_stall_on_pack
func.func @test_consecutive_unpack_stall_on_pack() {
  // CHECK: ttkernel.experimental::unpack_stall_on_pack
  "ttkernel.experimental::unpack_stall_on_pack"() : () -> ()
  // CHECK-NOT: ttkernel.experimental::unpack_stall_on_pack
  "ttkernel.experimental::unpack_stall_on_pack"() : () -> ()
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
    %noc_addr: !ttkernel.noc_addr, %l1_addr: i32, %size: i32) {
  // CHECK: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  ttkernel.noc_async_read(%noc_addr, %l1_addr, %size) : (!ttkernel.noc_addr, i32, i32) -> ()
  // CHECK: ttkernel.noc_async_read_barrier
  ttkernel.noc_async_read_barrier() : () -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @test_write_barrier_with_intervening_write
func.func @test_write_barrier_with_intervening_write(
    %l1_addr: i32, %noc_addr: !ttkernel.noc_addr, %size: i32) {
  // CHECK: ttkernel.noc_async_write_barrier
  ttkernel.noc_async_write_barrier() : () -> ()
  ttkernel.noc_async_write(%l1_addr, %noc_addr, %size) : (i32, !ttkernel.noc_addr, i32) -> ()
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
