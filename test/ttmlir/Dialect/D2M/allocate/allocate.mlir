// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate 2>&1 -o %t %s
// RUN: FileCheck %s --input-file=%t

!memreftype_l1   = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<1024x1024>, #ttcore.memory_space<l1>>
!memreftype_dram = memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<1024x1024>, #ttcore.memory_space<dram>>

// CHECK-LABEL: func @allocate_l1
func.func @allocate_l1() -> !memreftype_l1 {
  // CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
  %alloc = memref.alloc() : !memreftype_l1

  return %alloc : !memreftype_l1
}

// CHECK-LABEL: func @allocate_dram
func.func @allocate_dram() -> !memreftype_dram {
  // CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #dram>
  %alloc = memref.alloc() : !memreftype_dram

  return %alloc : !memreftype_dram
}

// CHECK-LABEL: func @allocate_mixed
func.func @allocate_mixed() -> (!memreftype_l1, !memreftype_dram, !memreftype_l1, !memreftype_dram, !memreftype_l1) {
  // CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
  %alloc_0 = memref.alloc() : !memreftype_l1
  // CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #dram>
  %alloc_1 = memref.alloc() : !memreftype_dram
  // CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
  %alloc_2 = memref.alloc() : !memreftype_l1
  // CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #dram>
  %alloc_3 = memref.alloc() : !memreftype_dram
  // CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
  %alloc_4 = memref.alloc() : !memreftype_l1

  return %alloc_0, %alloc_1, %alloc_2, %alloc_3, %alloc_4 : !memreftype_l1, !memreftype_dram, !memreftype_l1, !memreftype_dram, !memreftype_l1
}
