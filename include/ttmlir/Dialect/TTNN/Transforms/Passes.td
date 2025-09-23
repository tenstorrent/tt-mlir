// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD

include "mlir/Pass/PassBase.td"

def TTNNDeallocate: Pass<"ttnn-deallocate", "::mlir::ModuleOp"> {
  let summary = "Insert deallocate ops for tensors.";
  let description = [{
    This pass inserts deallocate ops after a tensor value's last use.
  }];
}

def TTNNDecomposeLayouts: Pass<"ttnn-decompose-layouts", "::mlir::ModuleOp"> {
  let summary = "Decompose ToLayoutOps to more granular memory ops.";
  let description = [{
    This pass decomposes ToLayoutOps to memory ops (e.g. toDevice, toMemoryConfig etc.).
  }];
}

def TTNNLayout : Pass<"ttnn-layout", "::mlir::ModuleOp"> {
  let summary = "Add layout information to tensors.";
  let description = [{
    This pass adds layout information to tensors.
  }];
}

def TTNNWorkarounds : Pass<"ttnn-workaround", "::mlir::ModuleOp"> {
  let summary = "Apply TTNN workarounds to the IR.";
  let description = [{
    This pass applies necessary TTNN workarounds to the IR in order to create
    a valid and functional IR that can be executed on the hardware.
  }];

  let options = [
      Option<"layoutWorkaroundsEnabled",
             "ttnn-enable-layout-workaround-pass",
             "bool", /*default=*/"true",
             "TTNN Layout Workarounds Pass">,
      Option<"decompositionWorkaroundsEnabled",
             "ttnn-enable-decomposition-workaround-pass",
             "bool", /*default=*/"true",
             "TTNN Decompsition Workarounds Pass">,
      Option<"optimizerEnabled",
             "ttnn-is-optimizer-enabled",
             "bool", /*default=*/"false",
             "flag specifying if optimizer is enabled">,
  ];
}

def TTNNCreateInputGenerators: Pass<"ttnn-create-input-gens", "::mlir::ModuleOp"> {
  let summary = "Create input generators for the forward functions.";
  let description = [{
    This pass creates input generators for the "forward" functions. It
    additionally creates a main function to run the forward function with the
    generated inputs.

    The pass is useful for EmitC path. By creating input generators before
    converting to Emitc Dialect, followed by transformation to C++ code, the
    resulting code won't require any edits to run.

    Given a forward function like this:

    ```mlir
    func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
      %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %0 : tensor<32x32xbf16>
    }
    ```

    The pass will prepend `_` to the existing function name to avoid name collision
    and create two function like this:

    ```mlir
    func.func @create_inputs_for_add() -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
      %0 = "ttnn.empty"() <{shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xbf16>
      %1 = "ttnn.empty"() <{shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xbf16>
      return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
    }

    func.func @main() -> i32 {
      %0:2 = call @create_inputs_for_add() : () -> (tensor<32x32xbf16>, tensor<32x32xbf16>)
      %1 = call @_add(%0#0, %0#1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %c0_i32 = arith.constant 0 : i32
      return %c0_i32 : i32
    }
    ```
  }];
}

def TTNNTuplifyTensors: Pass<"ttnn-tuplify-tensors", "::mlir::ModuleOp"> {
  let summary = "Tuplify tensors in all funcs within a module.";
  let description = [{
    This pass identifies all public functions in the module that exclusively use tensors as inputs and outputs. It rewrites
    each such function by packing all input arguments into a single tuple and wrapping the return value(s) into a tuple
    type as well.

    Given a forward function like this:

    ```mlir
    func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
      %0 = "ttnn.get_device"() : () -> !ttnn.device
      %1 = "ttnn.to_device"(%arg0, %0) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %2 = "ttnn.to_device"(%arg1, %0) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %3 = "ttnn.add"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %3 : tensor<32x32xbf16>
    }
    ```

    The pass will return:

    ```mlir
    func.func @add(%arg0: tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>, %arg1: !ttnn.device) -> tuple<tensor<32x32xbf16>> {
      %0 = ttcore.get_tuple_element %arg0[0] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %1 = ttcore.get_tuple_element %arg0[1] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %2 = "ttnn.to_device"(%0, %arg1) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %3 = "ttnn.to_device"(%1, %arg1) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %4 = "ttnn.add"(%2, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %5 = ttcore.tuple %4 : tuple<tensor<32x32xbf16>>
      return %5 : tuple<tensor<32x32xbf16>>
    }
    ```
  }];

  let options = [
    Option<"tuplifyInputIfEmpty",
            "tuplify-input-if-empty",
            "bool", /*default=*/"false",
            "If input arg is originally empty, force create an empty tuple arg. This is useful for scenarios where "
            "signatures are required have tuples on the input, like the EmitC dylib path.">
  ];
}

def TTNNPrepareConv2dWeightsAndBias : Pass<"ttnn-prepare-conv2d-weights-and-bias", "::mlir::ModuleOp"> {
  let summary = "Optimize Conv2d ops by inserting PrepareConv2dWeights and PrepareConv2dBias ops before them.";
  let description = [{
    This pass inserts a PrepareConv2dWeights and a PrepareConv2dBias operation before each Conv2d op, which preprocess the weights and bias used by the Conv2d operations.
    These operations can then be const-evaluated, leading to improved performance.
    To use this pass, the project must be built with the op model library by setting -DTTMLIR_ENABLE_OPMODEL=ON during the build process.
  }];
}

def TTNNFusing: Pass<"ttnn-fusing", "::mlir::ModuleOp">
{
  let summary = "TTNN fusing pass.";
  let description = "This pass tries to fuse operations together with goal to reduce the number of operations in the graph.";
}

def TTNNTraceHoistTransform : Pass<"ttnn-trace-hoist-transform", "::mlir::ModuleOp"> {
  let summary = "TTNN Trace hoist transform";
  let description = [{
    Transform pass which runs an analysis pass to find traceable subgraphs and hoist them into separate trace functions.
    The ops within the trace functions are then captured and replayed during runtime using the metal trace infra.
    This is a performance optimization feature that enables batches of ops to be replayed with a single command instead
    of dispatching commands for each op individually. One hard requirement is that all ops within a trace must be device
    operations and cannot have any host involvement. This will get verified within the verifier of the `ttnn.capture_or_execute_trace`
    operation.

    Example:
    ```mlir
    func.func @single_add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
      %0 = ttir.empty() : tensor<32x32xbf16>
      %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %1 : tensor<32x32xbf16>
    }
    ```

    the pass will rewrite the IR, hoist the operations and generate the following functions:

    ```mlir
    // Trace function. Contains the ops that were hoisted to trace.
    func.func @trace_0_single_add(%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout> attributes {ttnn.trace} {
      %0 = "ttnn.add"(%arg0, %arg1) <{output_dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
      return %0 : tensor<32x32xbf16, #ttnn_layout>
    }

    // Capture trace function. Copies inputs into pre-allocated empty tensors on device.
    // Notice that constants/params will not be copied into pre-allocated tensors - they
    // will be used directly instead.
    // Then, runs the trace function once to gather the outputs and warm up the device.
    // Finally, runs the trace function again to capture the trace.
    // Returns the trace id, function outputs, and the trace input output tensors.
    func.func @run_and_capture_trace_0_single_add(%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> (tensor<ui32, #ttnn_layout1>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) attributes {ttnn.trace} {
      %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
      %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
      %2 = "ttnn.from_device"(%arg0) : (tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout2>
      "ttnn.write_tensor"(%2, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #ttnn_layout2>, tensor<32x32xbf16, #ttnn_layout>) -> ()
      "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x32xbf16, #ttnn_layout2>) -> ()
      %3 = call @trace_0_single_add(%1, %arg1) : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
      %4 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn_layout1>
      %5 = call @trace_0_single_add(%1, %arg1) : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
      "ttnn.end_trace_capture"(%0, %4) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn_layout1>) -> ()
      return %4, %3, %1, %arg1, %5 : tensor<ui32, #ttnn_layout1>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>
    }
    ```

    These functions will get called conditionally within `ttnn.capture_or_execute_trace`:

    ```mlir
    func.func @single_add(%arg0: tensor<32x32xbf16, #ttnn_layout>, %arg1: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout> {
      %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
      %1 = "ttnn.capture_or_execute_trace"(%0, %arg0, %arg1) <{capture_callee = @run_and_capture_trace_0_single_add, execute_callee = @execute_trace_0_single_add}> : (!ttnn.device, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
      return %1 : tensor<32x32xbf16, #ttnn_layout>
    }
    ```

  }];
}

def TTNNUniqueLocations: Pass<"ttnn-unique-locations", "::mlir::ModuleOp">
{
  let summary = "Check if operations have unique locations.";
  let description = "This pass checks if all operations relevant to the optimizer have unique locations. If not, it will emit an error. This is necessary for the overrides to be applied correctly.";
}

def TTNNOperationValidationAndFallback: Pass<"ttnn-operation-validation-and-fallback", "::mlir::ModuleOp"> {
  let summary = "Validate operations and apply fallback strategies when needed.";
  let description = [{
    This pass validates TTNN operations by checking if their configurations
    will work at runtime. For operations that fail validation, it applies
    fallback strategies by transforming input operand layouts and data types
    to find working configurations.
  }];
}

def TTNNEmitPyWorkarounds: Pass<"ttnn-emitpy-workarounds", "::mlir::ModuleOp"> {
  let summary = "Apply TTNN workarounds specific to EmitPy pipeline.";
  let description = [{
    This pass applies TTNN workarounds that are specific to the EmitPy pipeline.
  }];
}

#endif
