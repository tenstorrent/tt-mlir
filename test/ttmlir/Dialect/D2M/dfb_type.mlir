// RUN: ttmlir-opt %s | FileCheck %s
// Test parser round-trip for the !d2m.dfb type.

#l1_ = #ttcore.memory_space<l1>

// CHECK: #l1 = #ttcore.memory_space<l1>

// CHECK-LABEL: @test_d2m_dfb_type_1p1c
// CHECK-SAME: !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
func.func @test_d2m_dfb_type_1p1c(%arg0: !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>) -> !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> {
  return %arg0 : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
}

// CHECK-LABEL: @test_d2m_dfb_type_1p4c
// CHECK-SAME: !d2m.dfb<memref<8x4x!ttcore.tile<32x32, bf16>, #ttcore.dfb_layout<8192x2048, 8, 1, 4, strided, blocked>, #l1>>
func.func @test_d2m_dfb_type_1p4c(%arg0: !d2m.dfb<memref<8x4x!ttcore.tile<32x32, bf16>,
                                                          #ttcore.dfb_layout<8192x2048, 8, 1, 4, strided, blocked>,
                                                          #l1_>>) -> !d2m.dfb<memref<8x4x!ttcore.tile<32x32, bf16>,
                                                                                       #ttcore.dfb_layout<8192x2048, 8, 1, 4, strided, blocked>,
                                                                                       #l1_>> {
  return %arg0 : !d2m.dfb<memref<8x4x!ttcore.tile<32x32, bf16>,
                                  #ttcore.dfb_layout<8192x2048, 8, 1, 4, strided, blocked>,
                                  #l1_>>
}
