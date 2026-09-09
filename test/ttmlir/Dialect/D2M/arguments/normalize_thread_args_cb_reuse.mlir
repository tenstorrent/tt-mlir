// RUN: ttmlir-opt --split-input-file --d2m-normalize-thread-args %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-normalize-thread-args --d2m-generic-regions-to-funcs --convert-d2m-to-ttkernel %s | FileCheck %s --check-prefix=KERNEL

#l1 = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @reuse_get_cb_ports_counts_fixed_io
  // KERNEL-LABEL: func.func private @compute_kernel0
  // KERNEL-SAME: operand_index = 31>]
  // CHECK: d2m.physical_cb_ports = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 2>
  // CHECK: d2m.get_cb(31)
  // CHECK: d2m.get_cb(32)
  func.func @reuse_get_cb_ports_counts_fixed_io(
      %in: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %out: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a0: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a1: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a2: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a3: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a4: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a5: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a6: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a7: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a8: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a9: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a10: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a11: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a12: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a13: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a14: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a15: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a16: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a17: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a18: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a19: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a20: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a21: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a22: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a23: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a24: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a25: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a26: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a27: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a28: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a29: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a30: memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
        ins(%in : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
        outs(%out : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
        additionalArgs(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15, %a16, %a17, %a18, %a19, %a20, %a21, %a22, %a23, %a24, %a25, %a26, %a27, %a28, %a29, %a30 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
    ^compute0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb2 = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb2 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb3 = d2m.get_cb(3) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb3 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb4 = d2m.get_cb(4) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb4 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb5 = d2m.get_cb(5) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb5 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb6 = d2m.get_cb(6) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb6 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb7 = d2m.get_cb(7) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb7 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb8 = d2m.get_cb(8) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb8 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb9 = d2m.get_cb(9) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb9 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb10 = d2m.get_cb(10) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb10 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb11 = d2m.get_cb(11) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb11 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb12 = d2m.get_cb(12) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb12 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb13 = d2m.get_cb(13) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb13 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb14 = d2m.get_cb(14) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb14 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb15 = d2m.get_cb(15) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb15 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb16 = d2m.get_cb(16) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb16 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb17 = d2m.get_cb(17) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb17 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb18 = d2m.get_cb(18) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb18 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb19 = d2m.get_cb(19) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb19 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb20 = d2m.get_cb(20) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb20 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb21 = d2m.get_cb(21) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb21 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb22 = d2m.get_cb(22) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb22 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb23 = d2m.get_cb(23) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb23 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb24 = d2m.get_cb(24) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb24 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb25 = d2m.get_cb(25) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb25 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb26 = d2m.get_cb(26) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb26 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb27 = d2m.get_cb(27) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb27 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb28 = d2m.get_cb(28) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb28 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb29 = d2m.get_cb(29) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb29 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb30 = d2m.get_cb(30) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb30 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb31 = d2m.get_cb(31) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb31 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb32 = d2m.get_cb(32) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb32 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    }
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @reuse_cb_backed_get_arg_ports
  // KERNEL-LABEL: func.func private @compute_kernel0
  // KERNEL-SAME: operand_index = 2>, <arg_type = cb_port, operand_index = 4>
  // CHECK: d2m.physical_cb_ports = array<i64: 0, 1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32>
  // CHECK: d2m.get_arg(32)
  // CHECK: d2m.get_arg(31)
  func.func @reuse_cb_backed_get_arg_ports(
      %in: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %out: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a0: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a1: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a2: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a3: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a4: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a5: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a6: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a7: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a8: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a9: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a10: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a11: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a12: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a13: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a14: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a15: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a16: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a17: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a18: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a19: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a20: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a21: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a22: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a23: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a24: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a25: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a26: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a27: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a28: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a29: memref<1x1x!ttcore.tile<32x32, f32>, #l1>,
      %a30: memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
        ins(%in : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
        outs(%out : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
        additionalArgs(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15, %a16, %a17, %a18, %a19, %a20, %a21, %a22, %a23, %a24, %a25, %a26, %a27, %a28, %a29, %a30 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
    ^compute0:
      %c0 = arith.constant 0 : index
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      d2m.push %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %v0 = memref.load %a0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v1 = memref.load %a1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v2 = memref.load %a2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v3 = memref.load %a3[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v4 = memref.load %a4[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v5 = memref.load %a5[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v6 = memref.load %a6[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v7 = memref.load %a7[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v8 = memref.load %a8[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v9 = memref.load %a9[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v10 = memref.load %a10[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v11 = memref.load %a11[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v12 = memref.load %a12[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v13 = memref.load %a13[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v14 = memref.load %a14[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v15 = memref.load %a15[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v16 = memref.load %a16[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v17 = memref.load %a17[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v18 = memref.load %a18[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v19 = memref.load %a19[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v20 = memref.load %a20[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v21 = memref.load %a21[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v22 = memref.load %a22[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v23 = memref.load %a23[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v24 = memref.load %a24[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v25 = memref.load %a25[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v26 = memref.load %a26[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v27 = memref.load %a27[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v28 = memref.load %a28[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v29 = memref.load %a29[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %v30 = memref.load %a30[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }
}
