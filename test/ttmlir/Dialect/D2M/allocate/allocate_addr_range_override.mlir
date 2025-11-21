// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=available-l1-addr-range=622592,1048576" 2>&1 -o %t %s
// RUN: FileCheck %s --input-file=%t

!memreftype_l1   = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<1024x1024, 1>, #ttcore.memory_space<l1>>

// CHECK-LABEL: func @allocate_addr_range_override
func.func @allocate_addr_range_override() -> !memreftype_l1 {
  // A single alloc will be given the min available address.
  // CHECK: %{{.+}} = memref.alloc() {address = 622592
  %alloc = memref.alloc() : !memreftype_l1

  return %alloc : !memreftype_l1
}
