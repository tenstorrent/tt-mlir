// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// d2m.fill_pad_cb (emitted by ExpandDMAReadCompositeView) lowers to
// ttkernel.experimental::fill_pad_cb. packed_value and num_bytes pass through;
// dtype attr maps to the kernel's DataFormat template arg.

#l1 = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_fill_pad_cb_bf16
  func.func @test_fill_pad_cb_bf16() attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<32x32xbf16, #l1>>
    %0 = d2m.reserve %cb : <memref<32x32xbf16, #l1>> -> memref<32x32xbf16, #l1>
    // CHECK-NOT: d2m.fill_pad_cb
    // CHECK: ttkernel.experimental::fill_pad_cb
    // bf16(1.0) packed twice = 0x3F803F80 = 1065369472
    d2m.fill_pad_cb(%cb) {dtype = bf16, num_bytes = 2048 : i32, packed_value = 1065369472 : i32} : !d2m.cb<memref<32x32xbf16, #l1>>
    d2m.push %cb : <memref<32x32xbf16, #l1>>
    return
  }
}
