// RUN: not ttmlir-opt --d2m-distribute-compute-threads='split-dims=-1' %s 2>&1 | FileCheck %s --check-prefix=NEGATIVE-DIM
// RUN: not ttmlir-opt --d2m-distribute-compute-threads='split-dims=2 matmul-interchange=1,0' %s 2>&1 | FileCheck %s --check-prefix=BAD-INTERCHANGE
// RUN: not ttmlir-opt --d2m-distribute-compute-threads='split-dims=3' %s 2>&1 | FileCheck %s --check-prefix=CURRENT-DIM
// RUN: not ttmlir-opt --d2m-distribute-compute-threads='split-dims=0,0' %s 2>&1 | FileCheck %s --check-prefix=DUPLICATE-DIM
// RUN: not ttmlir-opt --d2m-distribute-compute-threads='split-dims=0' %s 2>&1 | FileCheck %s --check-prefix=NON-PARALLEL
// RUN: not ttmlir-opt --d2m-distribute-compute-threads='split-dims=2' %s 2>&1 | FileCheck %s --check-prefix=NOT-OUTPUT-DIM
// RUN: not ttmlir-opt --d2m-distribute-compute-threads='split-dims=1' %s 2>&1 | FileCheck %s --check-prefix=TOO-SMALL

#l1 = #ttcore.memory_space<l1>
#map3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_drop_d2 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map2d = affine_map<(d0, d1) -> (d0, d1)>

// NEGATIVE-DIM: error: 'linalg.generic' op split-dims original dim index must be non-negative, got -1
// BAD-INTERCHANGE: error: 'linalg.generic' op split-dims original dim 2 cannot be mapped through matmul-interchange
// CURRENT-DIM: error: 'linalg.generic' op split-dims current dim 3 is outside linalg loop rank 3
// DUPLICATE-DIM: error: 'linalg.generic' op split-dims maps multiple requests to current dim 0
module {
  func.func @valid_until_requested_dim_validation(%arg0: memref<4x4x4x!ttcore.tile<32x32, bf16>, #l1>, %arg1: memref<4x4x4x!ttcore.tile<32x32, bf16>, #l1>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<4x4x4x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%arg1 : memref<4x4x4x!ttcore.tile<32x32, bf16>, #l1>)
     {
      linalg.generic {indexing_maps = [#map3d, #map3d], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x4x4x!ttcore.tile<32x32, bf16>, #l1>) outs(%arg1 : memref<4x4x4x!ttcore.tile<32x32, bf16>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
        linalg.yield %in : !ttcore.tile<32x32, bf16>
      }
    }
    return
  }
}

// NON-PARALLEL: error: 'linalg.generic' op requested split dim 0 is not a parallel iterator
// NOT-OUTPUT-DIM: error: 'linalg.generic' op requested split dim 2 is not a plain output affine dim
module {
  func.func @invalid_candidate_dim(%arg0: memref<4x4x4x!ttcore.tile<32x32, bf16>, #l1>, %arg1: memref<4x4x!ttcore.tile<32x32, bf16>, #l1>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<4x4x4x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%arg1 : memref<4x4x!ttcore.tile<32x32, bf16>, #l1>)
     {
      linalg.generic {indexing_maps = [#map3d, #map_drop_d2], iterator_types = ["reduction", "parallel", "parallel"]} ins(%arg0 : memref<4x4x4x!ttcore.tile<32x32, bf16>, #l1>) outs(%arg1 : memref<4x4x!ttcore.tile<32x32, bf16>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
        linalg.yield %in : !ttcore.tile<32x32, bf16>
      }
    }
    return
  }
}

// TOO-SMALL: error: 'linalg.generic' op requested split dim 1 has static output extent 1, smaller than any supported split factor
module {
  func.func @too_small(%arg0: memref<4x1x!ttcore.tile<32x32, bf16>, #l1>, %arg1: memref<4x1x!ttcore.tile<32x32, bf16>, #l1>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<4x1x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%arg1 : memref<4x1x!ttcore.tile<32x32, bf16>, #l1>)
     {
      linalg.generic {indexing_maps = [#map2d, #map2d], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<4x1x!ttcore.tile<32x32, bf16>, #l1>) outs(%arg1 : memref<4x1x!ttcore.tile<32x32, bf16>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
        linalg.yield %in : !ttcore.tile<32x32, bf16>
      }
    }
    return
  }
}
