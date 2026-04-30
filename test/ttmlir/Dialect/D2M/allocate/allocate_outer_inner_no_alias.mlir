// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: %python -c "import re,sys;s=open(sys.argv[1]).read();a=dict(re.findall(r'(%\\S+)\\s*=\\s*memref\\.alloc\\(\\)\\s*\\{[^}]*address\\s*=\\s*([0-9]+)\\s*:\\s*i64',s));l=re.findall(r'd2m\\.remote_load\\s+(%\\S+)\\s+(%\\S+)\\[',s);c=re.findall(r'd2m\\.remote_store\\s+(%\\S+)\\[',s);n=[l[0][1],l[1][1],c[0],l[0][0],l[1][0]];assert len({a[x] for x in n})==5" %t

// This test covers lifetime and address planning across d2m.spatial.
// Memrefs allocated outside the spatial region are used inside the region,
// so they must stay live until the spatial op completes.
// Memrefs allocated inside the region must not overlap addresses with the
// already-live outer allocations.
// In this IR, all relevant memrefs must get distinct addresses. FileCheck is
// not practical for pairwise distinctness, so a short inline Python check is used.

#l1 = #ttcore.memory_space<l1>
#s = #ttcore.shard<4096x4096, 1>
#m0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#m1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#m2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#par = #ttcore.iterator_type<parallel>
#red = #ttcore.iterator_type<reduction>

module {
  // CHECK-LABEL: func.func @outer_inner_no_alias
  // CHECK-COUNT-3: memref.alloc(){{.*}}: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // CHECK: d2m.spatial
  func.func @outer_inner_no_alias() {
    %a = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    %b = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    %c = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>]}
        ins(%a, %b : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>)
        outs(%c : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#m0, #m1, #m2], iterator_types = [#par, #par, #red], threads = [#d2m.thread<unified>]}
          ins(%a, %b : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>)
          outs(%c : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      ^bb0:
        %u = d2m.get_block_factor(0) : index
        %v = d2m.get_block_factor(1) : index
        %w = d2m.get_block_factor(2) : index
        affine.for %i = 0 to %u {
          affine.for %j = 0 to %v {
            affine.for %k = 0 to %w {
              %l = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
              %x = d2m.remote_load %l %a[%i, %k] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
              %r = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
              %y = d2m.remote_load %r %b[%k, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
              %o = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
              "d2m.tile_matmul_block"(%x, %y, %o) : (memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> ()
              d2m.remote_store %c[%i, %j] %o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
            } {d2m.blocking_loop = 2}
          } {d2m.blocking_loop = 1}
        } {d2m.blocking_loop = 0}
      }
    }
  // CHECK-COUNT-3: memref.dealloc %{{[A-Za-z0-9_.$-]+}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // CHECK: return
    return
  }
}
