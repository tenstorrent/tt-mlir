// RUN: not ttmlir-opt --ttcore-register-device --d2m-allocate %s 2>&1 | FileCheck %s

// CHECK: error: 'func.func' op required L1 memory usage 68719476736 exceeds memory capacity

!memreftype = memref<1x1x4096x4096x!ttcore.tile<32x32, f32>, #ttcore.shard<16777216x4096>, #ttcore.memory_space<l1>>

func.func @oom() -> !memreftype {
  %alloc = memref.alloc() : !memreftype
  return %alloc : !memreftype
}
