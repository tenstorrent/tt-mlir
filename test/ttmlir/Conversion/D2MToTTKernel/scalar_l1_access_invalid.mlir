// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel --split-input-file --verify-diagnostics %s

// Shapes that look like a scalar L1 access -- integer or index elements in L1 --
// but cannot be lowered. Each would otherwise fail deep in the conversion with an
// unresolved materialization, an invalid arith cast, or a ttkernel op verifier
// message, none of which name the actual problem.

#l1 = #ttcore.memory_space<l1>

module {
  // Scalar stores are not supported: the datamovement thread would become a
  // scalar CB producer, which needs the read/write pointer question settled.
  func.func private @scalar_store() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<32xi32, #l1>>
    %buf = d2m.reserve %cb : !d2m.cb<memref<32xi32, #l1>> -> memref<32xi32, #l1>
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : i32
    // expected-error @+1 {{scalar stores to L1 are not supported}}
    memref.store %c7, %buf[%c0] : memref<32xi32, #l1>
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>

module {
  // In a compute region memref.load is tile-granular and reads nothing, so a
  // scalar read there has no meaning.
  func.func private @scalar_load_in_compute() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<32xi32, #l1>>
    %buf = d2m.wait %cb : !d2m.cb<memref<32xi32, #l1>> -> memref<32xi32, #l1>
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{scalar L1 access is only supported on a datamovement thread}}
    %v = memref.load %buf[%c0] : memref<32xi32, #l1>
    %use = arith.addi %v, %v : i32
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>

module {
  // Unsigned elements: the arith ops a loaded index feeds are signless, so an
  // unsigned value cannot be used for addressing.
  func.func private @scalar_load_ui32() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<32xui32, #l1>>
    %ocb = d2m.get_cb(1) : !d2m.cb<memref<32xui32, #l1>>
    %buf = d2m.wait %cb : !d2m.cb<memref<32xui32, #l1>> -> memref<32xui32, #l1>
    %o = d2m.reserve %ocb : !d2m.cb<memref<32xui32, #l1>> -> memref<32xui32, #l1>
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{unsupported element type 'ui32' for a scalar L1 access}}
    %v = memref.load %buf[%c0] : memref<32xui32, #l1>
    memref.store %v, %o[%c0] : memref<32xui32, #l1>
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>

module {
  // Only 8/16/32-bit tt_l1_ptr flavors exist.
  func.func private @scalar_load_i64() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<32xi64, #l1>>
    %buf = d2m.wait %cb : !d2m.cb<memref<32xi64, #l1>> -> memref<32xi64, #l1>
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{unsupported element type 'i64' for a scalar L1 access}}
    %v = memref.load %buf[%c0] : memref<32xi64, #l1>
    %use = arith.addi %v, %v : i64
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>

module {
  // An index-element CB is not representable at all downstream; reject it here
  // rather than crash computing its page size.
  func.func private @scalar_load_index() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<32xindex, #l1>>
    %buf = d2m.wait %cb : !d2m.cb<memref<32xindex, #l1>> -> memref<32xindex, #l1>
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{unsupported element type 'index' for a scalar L1 access}}
    %v = memref.load %buf[%c0] : memref<32xindex, #l1>
    %use = arith.addi %v, %v : index
    return
  }
}
