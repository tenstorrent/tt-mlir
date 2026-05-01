// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=test-assume-l1-capacity=8388608" -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: %python -c "import sys; lines=open(sys.argv[1]).read().split('\n'); depth=0; awaiting=False; inside=False; entry_depth=0; bad=[]; exec('for l in lines:\n if \"d2m.generic\" in l: awaiting=True; entry_depth=depth\n depth+=l.count(\"{\")-l.count(\"}\")\n if awaiting and depth>entry_depth and \"d2m.generic\" not in l: inside=True; awaiting=False\n if inside and depth<=entry_depth: inside=False\n if inside and \"memref.alloc()\" in l and (\"cb_layout\" in l)!=(\"address\" in l): bad.append(l.strip())'); assert not bad, 'In-generic alloc with cb_layout xor address: '+str(bad)" %t

// This test verifies that all in-generic allocs either have both cb_layout and address (backed by real CB),
// or neither (aliased) after allocator pass. Having only one is invalid. The Python check tracks
// brace depth to identify when we're inside a d2m.generic body, then validates
// that all inner allocs have consistent cb_layout and address attributes.
//
// Since blockGenerics creates views on operands which affects the aliasing decision,
// we need to ensure aliasing logic is consistent with reblocking.

#l1 = #ttcore.memory_space<l1>
#eltwise = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {
  // CHECK-LABEL: func.func @reblock_cb_layout_consistency
  // After reblocking the eltwise add generic, the downstream untilize generic
  // should have properly allocated CB buffers (with both cb_layout AND address).
  // Eltwise add generic - this gets reblocked by the allocator from [1,1] to [2,1]
  // CHECK: d2m.view_layout
  // CHECK: d2m.generic {block_factors = [2, 1]
  // Untilize generic - consumes the reblocked output via a view
  // CHECK: d2m.view_layout
  // CHECK: d2m.generic {block_factors = [1, 1]
  // The reblocked generic should produce cb_layout allocs with addresses
  func.func @reblock_cb_layout_consistency() -> memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #l1> {
    %lhs = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %add_out = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%add_out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^bb0:
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %i = 0 to %bf0 {
        affine.for %j = 0 to %bf1 {
          %off0 = d2m.block_offset(0) : index
          %idx0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%i)[%off0]
          %off1 = d2m.block_offset(1) : index
          %idx1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%j)[%off1]

          %in0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>
          %ld0 = d2m.remote_load %in0 %lhs[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %in1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>
          %ld1 = d2m.remote_load %in1 %rhs[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %out = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>

          linalg.generic {indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = ["parallel", "parallel"]}
              ins(%in0, %in1 : memref<2x4x!ttcore.tile<32x32, f32>>, memref<2x4x!ttcore.tile<32x32, f32>>)
              outs(%out : memref<2x4x!ttcore.tile<32x32, f32>>) {
          ^bb0(%t0: !ttcore.tile<32x32, f32>, %t1: !ttcore.tile<32x32, f32>, %t2: !ttcore.tile<32x32, f32>):
            %sum = "d2m.tile_add"(%t0, %t1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %sum : !ttcore.tile<32x32, f32>
          }

          d2m.remote_store %add_out[%idx0, %idx1] %out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }

    %untilize_out = memref.alloc() : memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%add_out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%untilize_out : memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #l1>) {
    ^bb0:
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %i = 0 to %bf0 {
        affine.for %j = 0 to %bf1 {
          %off0 = d2m.block_offset(0) : index
          %idx0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%i)[%off0]
          %off1 = d2m.block_offset(1) : index
          %idx1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%j)[%off1]

          %tile_in = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>
          %ld = d2m.remote_load %tile_in %add_out[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>

          %scalar_out = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32>
          %untilized = "d2m.tile_untilize_block"(%tile_in, %scalar_out) : (memref<2x4x!ttcore.tile<32x32, f32>>, memref<64x128xf32>) -> memref<64x128xf32>

          d2m.remote_store %untilize_out[%idx0, %idx1] %scalar_out : memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #l1>, memref<64x128xf32> -> memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #l1>
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }

    return %untilize_out : memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #l1>
  }
}
