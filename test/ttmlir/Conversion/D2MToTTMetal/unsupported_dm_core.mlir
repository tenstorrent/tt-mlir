// RUN: not ttmlir-opt --ttcore-register-device --convert-d2m-to-ttmetal %s 2>&1 | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

module {
  func.func @unsupported_dm_core(%arg0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>

    // CHECK: error: DM core indices greater than 1 are not supported by D2MToTTMetal lowering yet
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, dm_core = 2>, #d2m.thread<compute>]}
        ins(%arg0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^datamovement0:
    }, {
    ^compute0:
    }
    return
  }
}
