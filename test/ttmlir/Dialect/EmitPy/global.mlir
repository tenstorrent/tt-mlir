// RUN: ttmlir-opt --split-input-file %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK: emitpy.global @valid_global_1 = #emitpy.opaque<"None">
  emitpy.global @valid_global_1 = #emitpy.opaque<"None">
}

// -----

module {
  // CHECK: emitpy.global @valid_global_2 = #emitpy.opaque<"[]">
  emitpy.global @valid_global_2 = #emitpy.opaque<"[]">
}

// -----

module {
  // CHECK: emitpy.global @valid_global_3 = #emitpy.opaque<"[ttnn.Tensor]">
  emitpy.global @valid_global_3 = #emitpy.opaque<"[ttnn.Tensor]">
}

// -----

module {
  // CHECK: emitpy.global @valid_global_4 = 1 : i64
  emitpy.global @valid_global_4 = 1 : i64
}

// -----

module {
  // CHECK: emitpy.global @tensor_global = #emitpy.opaque<"[ttnn.Tensor]">
  emitpy.global @tensor_global = #emitpy.opaque<"[ttnn.Tensor]">

  func.func @get_tensor() -> !emitpy.opaque<"[ttnn.Tensor]"> {
    // CHECK: %{{.*}} = emitpy.global_statement @tensor_global : !emitpy.opaque<"[ttnn.Tensor]">
    %0 = emitpy.global_statement @tensor_global : !emitpy.opaque<"[ttnn.Tensor]">
    return %0 : !emitpy.opaque<"[ttnn.Tensor]">
  }
}

// -----

module {
  // CHECK: emitpy.global @counter = 0 : i64
  emitpy.global @counter = 0 : i64

  func.func @get_counter() -> i64 {
    // CHECK: %{{.*}} = emitpy.global_statement @counter : i64
    %0 = emitpy.global_statement @counter : i64
    return %0 : i64
  }
}

// -----

module {
  // CHECK: emitpy.global @value = #emitpy.opaque<"None">
  emitpy.global @value = #emitpy.opaque<"None">

  func.func @assign_value(%arg0: !emitpy.opaque<"[ttnn.Tensor]">) -> () {
    // CHECK: emitpy.global_statement @value : !emitpy.opaque<"[ttnn.Tensor]">
    %0 = emitpy.global_statement @value : !emitpy.opaque<"[ttnn.Tensor]">
    // CHECK: emitpy.assign_global @value = %{{.*}} : !emitpy.opaque<"[ttnn.Tensor]">
    emitpy.assign_global @value = %arg0 : !emitpy.opaque<"[ttnn.Tensor]">
    return
  }
}

// -----

module {
  // CHECK: emitpy.global @counter = 0
  emitpy.global @counter = 0

  func.func @assign_counter(%arg0: i64) -> () {
    // CHECK: emitpy.global_statement @counter : i64
    %0 = emitpy.global_statement @counter : i64
    // CHECK: emitpy.assign_global @counter = %{{.*}} : i64
    emitpy.assign_global @counter = %arg0 : i64
    return
  }
}

// -----

module {
  // CHECK: emitpy.global @_global_var = #emitpy.opaque<"None">
  emitpy.global @_global_var = #emitpy.opaque<"None">

  func.func @use_global() -> !emitpy.opaque<"[ttnn.Tensor]"> {
    // CHECK: %{{.*}} = emitpy.global_statement @_global_var : !emitpy.opaque<"[ttnn.Tensor]">
    %0 = emitpy.global_statement @_global_var : !emitpy.opaque<"[ttnn.Tensor]">
    return %0 : !emitpy.opaque<"[ttnn.Tensor]">
  }
}

// -----

module {
  // CHECK: emitpy.global @counter = 0 : i64
  emitpy.global @counter = 0 : i64

  func.func @declare_global() -> i64 {
    // CHECK: %{{.*}} = emitpy.global_statement @counter : i64
    %0 = emitpy.global_statement @counter : i64
    return %0 : i64
  }
}

// -----

module {
  // CHECK: emitpy.global @global_var = #emitpy.opaque<"None">
  emitpy.global @global_var = #emitpy.opaque<"None">

  func.func @workflow(%arg0: !emitpy.opaque<"[ttnn.Tensor]">) -> !emitpy.opaque<"[ttnn.Tensor]"> {
    // CHECK: %{{.*}} = emitpy.global_statement @global_var : !emitpy.opaque<"[ttnn.Tensor]">
    %0 = emitpy.global_statement @global_var : !emitpy.opaque<"[ttnn.Tensor]">
    // CHECK: emitpy.assign_global @global_var = %{{.*}} : !emitpy.opaque<"[ttnn.Tensor]">
    emitpy.assign_global @global_var = %arg0 : !emitpy.opaque<"[ttnn.Tensor]">
    // CHECK: %{{.*}} = emitpy.global_statement @global_var : !emitpy.opaque<"[ttnn.Tensor]">
    %1 = emitpy.global_statement @global_var : !emitpy.opaque<"[ttnn.Tensor]">
    return %1 : !emitpy.opaque<"[ttnn.Tensor]">
  }
}

// -----

module {
  // CHECK: emitpy.global @global_a = #emitpy.opaque<"None">
  emitpy.global @global_a = #emitpy.opaque<"None">
  // CHECK: emitpy.global @global_b = #emitpy.opaque<"[ttnn.Tensor]">
  emitpy.global @global_b = #emitpy.opaque<"[ttnn.Tensor]">
  // CHECK: emitpy.global @global_c = 0 : i64
  emitpy.global @global_c = 0 : i64

  func.func @use_multiple() -> i64 {
    // CHECK: %{{.*}} = emitpy.global_statement @global_a : !emitpy.opaque<"None">
    %0 = emitpy.global_statement @global_a : !emitpy.opaque<"None">
    // CHECK: %{{.*}} = emitpy.global_statement @global_b : !emitpy.opaque<"[ttnn.Tensor]">
    %1 = emitpy.global_statement @global_b : !emitpy.opaque<"[ttnn.Tensor]">
    // CHECK: %{{.*}} = emitpy.global_statement @global_c : i64
    %2 = emitpy.global_statement @global_c : i64
    return %2 : i64
  }
}
