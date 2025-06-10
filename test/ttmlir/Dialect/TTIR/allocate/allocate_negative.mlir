// RUN: not ttmlir-opt --tt-register-device --ttir-allocate %s 2>&1 | FileCheck %s

// CHECK: error: 'func.func' op required memory usage 68719476736 exceeds memory size

!memreftype = memref<1x1x4096x4096x!tt.tile<32x32, f32>, #tt.shard<16777216x4096>, #tt.memory_space<l1>>

func.func @oom() -> !memreftype {
  %alloc = memref.alloc() : !memreftype
  return %alloc : !memreftype
}
