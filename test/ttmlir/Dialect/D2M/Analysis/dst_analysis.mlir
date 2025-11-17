// RUN: ttmlir-opt --ttcore-register-device --d2m-dst-analysis="strategy=basic emit-diagnostics=true" %s 2>&1 | FileCheck %s --check-prefix=BASIC
// RUN: ttmlir-opt --ttcore-register-device --d2m-dst-analysis="strategy=graph-coloring emit-diagnostics=true" %s 2>&1 | FileCheck %s --check-prefix=GC
// RUN: ttmlir-opt --ttcore-register-device --d2m-dst-analysis="strategy=greedy emit-diagnostics=true" %s 2>&1 | FileCheck %s --check-prefix=GREEDY

// Test DST analysis with different strategies on a simple matmul operation.

#l1_ = #ttcore.memory_space<l1>

module {
  // BASIC: remark: DST analysis (basic): 3 slices required
  // GC: remark: DST analysis (graph-coloring): {{[0-9]+}} slices required
  // GREEDY: remark: DST analysis (greedy): {{[0-9]+}} slices required

  func.func @simple_matmul(%in0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192>, #l1_>,
                            %in1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192>, #l1_>,
                            %out0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192>, #l1_>,
                       memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192>, #l1_>)
      outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          affine.for %k = 0 to 2 {
            %a = affine.load %mem0[%i, %k] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %b = affine.load %mem1[%k, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %c = affine.load %mem2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %result = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            affine.store %result, %mem2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          }
        }
      }
    }
    return
  }
}
