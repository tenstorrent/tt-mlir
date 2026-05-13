// RUN: ttmlir-opt %s | FileCheck %s
// Test parser round-trip for the !ttkernel.dfb type.

// CHECK-LABEL: func.func @test_ttkernel_dfb_type_1p1c
// CHECK-SAME: !ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>
func.func @test_ttkernel_dfb_type_1p1c(%arg0: !ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>) -> !ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1> {
  return %arg0 : !ttkernel.dfb<8, !ttcore.tile<32x32, bf16>, 1, 1>
}

// CHECK-LABEL: func.func @test_ttkernel_dfb_type_1p4c
// CHECK-SAME: !ttkernel.dfb<16, !ttcore.tile<32x32, bf16>, 1, 4>
func.func @test_ttkernel_dfb_type_1p4c(%arg0: !ttkernel.dfb<16, !ttcore.tile<32x32, bf16>, 1, 4>) -> !ttkernel.dfb<16, !ttcore.tile<32x32, bf16>, 1, 4> {
  return %arg0 : !ttkernel.dfb<16, !ttcore.tile<32x32, bf16>, 1, 4>
}
