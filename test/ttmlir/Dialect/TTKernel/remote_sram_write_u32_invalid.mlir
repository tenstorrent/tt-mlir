// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Test: $src_sram_addr must be i32, !ttkernel.l1_addr, or !ttkernel.local_semaphore.
func.func @test_remote_sram_write_u32_wrong_src_type(%dst: !ttkernel.noc_addr) {
  // expected-note @+1 {{prior use here}}
  %bad_src = arith.constant 0 : i64
  // expected-error @+1 {{use of value '%bad_src' expects different type than prior uses: '!ttkernel.l1_addr' vs 'i64'}}
  ttkernel.remote_sram_write_u32(%bad_src, %dst) : (!ttkernel.l1_addr, !ttkernel.noc_addr) -> ()
  return
}
