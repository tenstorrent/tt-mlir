// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false" -o %t.no_cpu_hoist %s
// RUN: FileCheck %s --input-file=%t.no_cpu_hoist

// An index that is a constant or parameter argument, or computed from one, has
// its relayout to host hoisted into a const-eval function. The index must end up
// si32, the type the argument already has after element type normalization, so
// the relayout must not add a typecast to a signless i32.

// CHECK-LABEL: func.func @constant_index
// CHECK: %[[INDEX:[0-9]+]] = ttcore.load_cached
// CHECK-SAME: -> tensor<si32
// CHECK: ttnn.case index(%[[INDEX]] : tensor<si32
func.func @constant_index(%arg0: tensor<32x32xf32>, %index: tensor<i32> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xf32> {
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0 : tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>):
    ttir.yield %a : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @index_from_parameter
// CHECK: %[[INDEX:[0-9]+]] = ttcore.load_cached
// CHECK-SAME: -> tensor<si32
// CHECK: ttnn.case index(%[[INDEX]] : tensor<si32
func.func @index_from_parameter(%arg0: tensor<32x32xf32>, %p: tensor<f32> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xf32> {
  %index = "ttir.typecast"(%p) : (tensor<f32>) -> tensor<i32>
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0 : tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>):
    ttir.yield %a : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}
