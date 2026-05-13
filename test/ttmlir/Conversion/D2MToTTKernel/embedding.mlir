// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>

module {
  func.func private @embedding_bf16_table_ui32_indices() attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
    %indices = d2m.get_arg(0) resolution_stage = compile : memref<1x1x32x32xui32, #ttcore.shard<128x4, 1>, #l1>
    %weight = d2m.get_arg(1) resolution_stage = compile : memref<1x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    %output = d2m.get_arg(2) resolution_stage = compile : memref<2x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    %index_scratch = d2m.get_arg(3) resolution_stage = compile : memref<1x1024xui32, #ttcore.cb_layout<4096x4, 1>, #l1>
    %row_scratch = d2m.get_arg(4) resolution_stage = compile : memref<1x1024xbf16, #ttcore.cb_layout<2048x2, 1>, #l1>
    // CHECK-LABEL: func.func private @embedding_bf16_table_ui32_indices
    // CHECK: !ttkernel.cb<1024, ui32>
    // CHECK: !ttkernel.cb<1024, bf16>
    // CHECK: %[[INDEX_BYTES:.*]] = arith.constant 16 : i32
    // CHECK: %[[BF16_BYTES:.*]] = arith.constant 2 : i32
    // CHECK: ttkernel.noc_async_read({{.*}}, {{.*}}, %[[INDEX_BYTES]])
    // CHECK: ttkernel.load_from_l1({{.*}}, {{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
    // CHECK: %[[ROW_ELEMS:.*]] = arith.select {{.*}} : index
    // CHECK: %[[ROW_ELEMS_I32:.*]] = arith.index_cast %[[ROW_ELEMS]] : index to i32
    // CHECK: %[[ROW_BYTES:.*]] = arith.muli %[[ROW_ELEMS_I32]], %[[BF16_BYTES]] : i32
    // CHECK: ttkernel.noc_async_read({{.*}}, {{.*}}, %[[ROW_BYTES]])
    // CHECK: ttkernel.noc_async_write({{.*}}, {{.*}}, %[[ROW_BYTES]])
    d2m.indexed_row_copy %indices, %weight, %output scratch %index_scratch, %row_scratch<6, 5> {indicesShape = array<i64: 2, 3>} : memref<1x1x32x32xui32, #ttcore.shard<128x4, 1>, #l1>, memref<1x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>, memref<2x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>, memref<1x1024xui32, #ttcore.cb_layout<4096x4, 1>, #l1>, memref<1x1024xbf16, #ttcore.cb_layout<2048x2, 1>, #l1>
    return
  }

  func.func private @embedding_i32_table_ui32_indices() attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
    %indices = d2m.get_arg(0) resolution_stage = compile : memref<1x1x32x32xui32, #ttcore.shard<128x4, 1>, #l1>
    %weight = d2m.get_arg(1) resolution_stage = compile : memref<1x1x32x32xi32, #ttcore.shard<128x4, 1>, #l1>
    %output = d2m.get_arg(2) resolution_stage = compile : memref<3x1x32x32xi32, #ttcore.shard<128x4, 1>, #l1>
    %index_scratch = d2m.get_arg(3) resolution_stage = compile : memref<1x1024xui32, #ttcore.cb_layout<4096x4, 1>, #l1>
    %row_scratch = d2m.get_arg(4) resolution_stage = compile : memref<1x1024xi32, #ttcore.cb_layout<4096x4, 1>, #l1>
    // CHECK-LABEL: func.func private @embedding_i32_table_ui32_indices
    // CHECK: !ttkernel.cb<1024, ui32>
    // CHECK: !ttkernel.cb<1024, i32>
    // CHECK: %[[INDEX_BYTES:.*]] = arith.constant 16 : i32
    // CHECK: %[[I32_BYTES:.*]] = arith.constant 4 : i32
    // CHECK: ttkernel.noc_async_read({{.*}}, {{.*}}, %[[INDEX_BYTES]])
    // CHECK: ttkernel.load_from_l1({{.*}}, {{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
    // CHECK: %[[ROW_ELEMS:.*]] = arith.select {{.*}} : index
    // CHECK: %[[ROW_ELEMS_I32:.*]] = arith.index_cast %[[ROW_ELEMS]] : index to i32
    // CHECK: %[[ROW_BYTES:.*]] = arith.muli %[[ROW_ELEMS_I32]], %[[I32_BYTES]] : i32
    // CHECK: ttkernel.noc_async_read({{.*}}, {{.*}}, %[[ROW_BYTES]])
    // CHECK: ttkernel.noc_async_write({{.*}}, {{.*}}, %[[ROW_BYTES]])
    d2m.indexed_row_copy %indices, %weight, %output scratch %index_scratch, %row_scratch<3, 1> {indicesShape = array<i64: 3, 1>} : memref<1x1x32x32xui32, #ttcore.shard<128x4, 1>, #l1>, memref<1x1x32x32xi32, #ttcore.shard<128x4, 1>, #l1>, memref<3x1x32x32xi32, #ttcore.shard<128x4, 1>, #l1>, memref<1x1024xui32, #ttcore.cb_layout<4096x4, 1>, #l1>, memref<1x1024xi32, #ttcore.cb_layout<4096x4, 1>, #l1>
    return
  }
}
