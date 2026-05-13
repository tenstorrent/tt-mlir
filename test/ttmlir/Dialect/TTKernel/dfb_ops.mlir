// RUN: ttmlir-opt %s | FileCheck %s
// Parser round-trip for TTKernel DFB sync ops:
//   ttkernel.dfb_reserve_back / dfb_push_back / dfb_wait_front /
//   dfb_pop_front / dfb_finish
// The verifier allows DFB ops in a func.func directly under a Module
// (same shape as the existing CB ops).

// CHECK-LABEL: func.func @test_ttkernel_dfb_ops_1p1c
func.func @test_ttkernel_dfb_ops_1p1c(%dfb: !ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>, %n: i32) {
  // CHECK: ttkernel.dfb_reserve_back(%{{.*}}, %{{.*}}) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>, i32) -> ()
  ttkernel.dfb_reserve_back(%dfb, %n) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>, i32) -> ()
  // CHECK: ttkernel.dfb_push_back(%{{.*}}, %{{.*}}) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>, i32) -> ()
  ttkernel.dfb_push_back(%dfb, %n) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>, i32) -> ()
  // CHECK: ttkernel.dfb_wait_front(%{{.*}}, %{{.*}}) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>, i32) -> ()
  ttkernel.dfb_wait_front(%dfb, %n) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>, i32) -> ()
  // CHECK: ttkernel.dfb_pop_front(%{{.*}}, %{{.*}}) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>, i32) -> ()
  ttkernel.dfb_pop_front(%dfb, %n) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>, i32) -> ()
  // CHECK: ttkernel.dfb_finish(%{{.*}}) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>) -> ()
  ttkernel.dfb_finish(%dfb) : (!ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>) -> ()
  return
}

// CHECK-LABEL: func.func @test_ttkernel_dfb_ops_1p4c
func.func @test_ttkernel_dfb_ops_1p4c(%dfb: !ttkernel.dfb<16, !ttcore.tile<32x32, bf16>, 1, 4>, %n: i32) {
  // CHECK: ttkernel.dfb_wait_front
  ttkernel.dfb_wait_front(%dfb, %n) : (!ttkernel.dfb<16, !ttcore.tile<32x32, bf16>, 1, 4>, i32) -> ()
  ttkernel.dfb_pop_front(%dfb, %n) : (!ttkernel.dfb<16, !ttcore.tile<32x32, bf16>, 1, 4>, i32) -> ()
  return
}

// CHECK-LABEL: func.func @test_ttkernel_dfb_arg_type
// CHECK-SAME: ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = dfb_id, operand_index = 0>]>
func.func @test_ttkernel_dfb_arg_type() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = dfb_id, operand_index = 0>]>} {
  return
}
