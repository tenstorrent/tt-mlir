// RUN: ttmlir-opt --convert-ttir-to-ttmetal %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #tt.memory_space<l1>

func.func @alloc_no_addr(%arg0: memref<1x1x8x24x!tt.tile<32x32, f32>, #tt.shard<98304x4096>, #l1_>, %arg1: memref<1x1x24x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>) ->memref<8x8x1x4x!tt.tile<32x32, f32>, #l1_> {
    // CHECK: %{{[0-9]+}} = "ttmetal.create_buffer"()
    // CHECK-SAME: {{.*}}address = 1000 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8x1x4x!tt.tile<32x32, f32>, #l1_>
    return %alloc : memref<8x8x1x4x!tt.tile<32x32, f32>, #l1_>
}

func.func @alloc_with_addr(%arg0: memref<1x1x8x24x!tt.tile<32x32, f32>, #tt.shard<98304x4096>, #l1_>, %arg1: memref<1x1x24x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>) -> memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.shard<12288x4096>, #l1_> {
    // CHECK: %{{[0-9]+}} = "ttmetal.create_buffer"()
    // CHECK-SAME: {{.*}}address = 86016 : i64
    %alloc = memref.alloc() {alignment = 64 : i64, address = 0x15000 : i64} : memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.shard<12288x4096>, #l1_>
    return %alloc : memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.shard<12288x4096>, #l1_>
}
