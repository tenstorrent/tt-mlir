// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>

func.func @alloc_with_addr(%arg0: memref<1x1x8x24x!ttcore.tile<32x32, f32>, #ttcore.shard<98304x4096>, #l1_>, %arg1: memref<1x1x24x32x!ttcore.tile<32x32, f32>, #ttcore.shard<131072x4096>, #l1_>) -> memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #l1_> {
    // CHECK: %{{[0-9]+}} = "ttmetal.create_buffer"()
    // CHECK-SAME: {{.*}}address = 86016 : i64
    %alloc = memref.alloc() {alignment = 64 : i64, address = 0x15000 : i64} : memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #l1_>
    return %alloc : memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #l1_>
}
