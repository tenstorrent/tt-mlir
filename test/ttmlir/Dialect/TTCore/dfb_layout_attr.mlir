// RUN: ttmlir-opt %s | FileCheck %s
// Test parser round-trip for #ttcore.dfb_layout attribute. The
// producer/consumer access patterns are encoded as bare DFBAccessPattern
// enum keywords (not as #ttcore.dfb_access_pattern attribute syntax).

#l1_ = #ttcore.memory_space<l1>

// CHECK: #l1 = #ttcore.memory_space<l1>

// CHECK-LABEL: @test_dfb_layout_1p1c_strided
// CHECK-SAME: memref<8x4x!ttcore.tile<32x32, bf16>, #ttcore.dfb_layout<8192x2048, 2, 1, 1, strided, strided>, #l1>
func.func @test_dfb_layout_1p1c_strided(%arg0: memref<8x4x!ttcore.tile<32x32, bf16>,
                                                       #ttcore.dfb_layout<8192x2048, 2, 1, 1, strided, strided>,
                                                       #l1_>) {
  return
}

// CHECK-LABEL: @test_dfb_layout_1p4c_blocked
// CHECK-SAME: memref<8x4x!ttcore.tile<32x32, bf16>, #ttcore.dfb_layout<8192x2048, 8, 1, 4, strided, blocked>, #l1>
func.func @test_dfb_layout_1p4c_blocked(%arg0: memref<8x4x!ttcore.tile<32x32, bf16>,
                                                      #ttcore.dfb_layout<8192x2048, 8, 1, 4, strided, blocked>,
                                                      #l1_>) {
  return
}
