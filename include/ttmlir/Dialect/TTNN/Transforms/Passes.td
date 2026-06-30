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
      Option<"allReduceWorkaroundEnabled",
             "ttnn-enable-all-reduce-workaround",
             "bool", /*default=*/"true",
             "Enable the all_reduce decomposition workaround which breaks "
             "all_reduce down into reduce_scatter + all_gather (or "
             "all_gather + local reduce).">,
      Option<"optimizationLevel",
             "ttnn-optimization-level",
             "int", /*default=*/"0",
             "Optimization level forwarded from the pipeline; controls which "
             "workarounds are active.">,
  ];
}

def TTNNDecomposition : Pass<"ttnn-decomposition", "::mlir::ModuleOp"> {
  let summary = "Decompose TTNN ops into primitive equivalents.";
  let description = [{
    This pass decomposes TTNN ops that cannot be handled by any available
    fused kernel into sequences of primitive TTNN ops. Unlike workarounds,
    these decompositions are permanent and functional - they apply when no
    suitable kernel exists for the given op configuration (e.g. shape).
  }];

  let options = [
    Option<"enableOpConstraints",
            "enable-op-constraints",
            "bool", /*default=*/"false",
            "Enable op model constraint validation for decomposition patterns. "
            "Requires TTMLIR_ENABLE_OPMODEL. When enabled, ops are validated "
            "before deciding whether to decompose.">,
    Option<"maxFallbackAttempts",
            "max-fallback-attempts",
            "uint32_t", /*default=*/"10000",
            "Maximum number of fallback configurations to try before "
            "giving up (0 = unlimited). Only used when "
            "enable-op-constraints is true.">
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

def TTNNLoadInputTensors: Pass<"ttnn-load-input-tensors", "::mlir::ModuleOp"> {
  let summary = "Load input tensors from disk for the forward functions.";
  let description = [{
    This pass loads input tensors from disk using ttnn.load_tensor operations
    for the "forward" functions. It additionally creates a main function to run
    the forward function with the loaded inputs.

    This is an alternative to TTNNCreateInputGenerators that loads real tensor
    data from disk instead of generating synthetic inputs with ttnn.ones.

    Given a forward function like this:

    ```mlir
    func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
      %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %0 : tensor<32x32xbf16>
    }
    ```

    The pass will prepend `_` to the existing function name to avoid name collision
    and create functions that load inputs from arg0.tensorbin, arg1.tensorbin, etc:

    ```mlir
    func.func @load_inputs_for_add() -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
      %0 = "ttnn.load_tensor"() <{file_path = "arg0.tensorbin"}> : () -> tensor<32x32xbf16>
      %1 = "ttnn.load_tensor"() <{file_path = "arg1.tensorbin"}> : () -> tensor<32x32xbf16>
      return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
    }

    func.func @main() -> i32 {
      %0:2 = call @load_inputs_for_add() : () -> (tensor<32x32xbf16>, tensor<32x32xbf16>)
      %1 = call @_add(%0#0, %0#1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %c0_i32 = arith.constant 0 : i32
      return %c0_i32 : i32
    }
    ```
  }];
  let options = [
    Option<"tensorLoadDirectory", "tensor-load-directory", "std::string", /*default=*/"\"\"",
           "Directory path where input tensors are stored">,
    Option<"tensorLoadFilePrefix", "tensor-load-file-prefix", "std::string", /*default=*/"\"arg\"",
           "Prefix for input tensor files">,
  ];
}

def TTNNSplitForwardFuncArgsByType : Pass<"ttnn-split-forward-func-args-by-type", "::mlir::ModuleOp"> {
  let summary = "Split forward function arguments into activations tuple and weights dictionary.";
  let description = [{
    This pass rewrites each forward function of device module by separating its tensor arguments into activations and weights.
    Activation arguments are packed into a tuple, while weight arguments are packed into a dictionary.

    Given a forward function like this:

    ```mlir
    func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                       %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>})
        -> tensor<32x32xbf16> {
      ...
    }
    ```

    The pass will produce:

    ```mlir
    func.func @forward(%arg0: tuple<tensor<32x32xbf16>>,
                       %arg1: !ttcore.dict)
        -> tensor<32x32xbf16>
        attributes {ttcore.original_argument_types = [...]} {
      %input0 = ttcore.get_tuple_element %arg0[0] : ...
      %weight0 = ttcore.get_key_value %arg1["weight_0"] : ...
      ...
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
            "If forward function input arg is originally empty, force create an empty tuple arg. This is useful for scenarios where "
            "signatures are required to have tuples on the input, like the EmitC dylib or EmitPy targetModule path.">
  ];
}

def TTNNPrepareModuleForExport : Pass<"ttnn-prepare-module-for-export", "::mlir::ModuleOp"> {
  let summary = "Prepare module for Python module export.";
  let description = [{
    Reshapes the first forward device function into the canonical entry-point
    signature consumed by the frontends by:
    1. Renaming the first forward device function to "forward"
    2. Adding a device argument to the function signature
    3. Replacing all ttnn.get_device ops with the device function argument
    4. Preparing the forward function input tensors to be in the right form (host/device, layout, dtype)

    Given a forward function like this:

    ```mlir
    func.func @add(%arg0: tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tuple<tensor<32x32xbf16>> {
      %0 = ttcore.get_tuple_element %arg0[0] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %1 = ttcore.get_tuple_element %arg0[1] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %2 = "ttnn.get_device"() : () -> !ttnn.device
      %3 = "ttnn.to_device"(%0, %2) : (tensor<32x32xbf16>, !ttnn.device) -> tensor<32x32xbf16>
      %4 = "ttnn.to_device"(%1, %2) : (tensor<32x32xbf16>, !ttnn.device) -> tensor<32x32xbf16>
      %5 = "ttnn.add"(%3, %4) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %6 = ttcore.tuple %5 : tuple<tensor<32x32xbf16>>
      return %6 : tuple<tensor<32x32xbf16>>
    }
    ```

    The pass will return:

    ```mlir
    func.func @forward(%arg0: tuple<tensor<32x32xbf16>, tensor<32x32xbf16>> {emitpy.name = "input"},
                       %arg1: !ttnn.device {emitpy.name = "device"}) -> tuple<tensor<32x32xbf16>> {
      %0 = ttcore.get_tuple_element %arg0[0] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %1 = ttcore.get_tuple_element %arg0[1] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %2 = "ttnn.to_device"(%0, %arg1) : (tensor<32x32xbf16>, !ttnn.device) -> tensor<32x32xbf16>
      %3 = "ttnn.to_device"(%1, %arg1) : (tensor<32x32xbf16>, !ttnn.device) -> tensor<32x32xbf16>
      %4 = "ttnn.add"(%2, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %5 = ttcore.tuple %4 : tuple<tensor<32x32xbf16>>
      return %5 : tuple<tensor<32x32xbf16>>
    }
    ```

    The example above shows the trivial case where tuple elements have no
    device-buffer layout and step 4 is a no-op. When a tuple element does
    carry a device-buffer layout, step 4 additionally retypes it to a host
    layout in the signature and stages each `ttcore.get_tuple_element`
    host -> device via a `ttnn.to_layout` followed by a `ttnn.to_device`.
    The target memory config (buffer type, memory layout, shard spec) is
    conveyed via the result tensor type's `TTNNLayoutAttr` encoding
    (`#device` below; `#host` is `SystemMemory + RowMajor`, `#host_staged`
    is `SystemMemory` carrying the target tile/dtype):

    ```mlir
    // Before:
    %0 = ttcore.get_tuple_element %arg0[0]
        : (tuple<tensor<32x32xbf16, #device>, ...>) -> tensor<32x32xbf16, #device>

    // After:
    %0 = ttcore.get_tuple_element %arg0[0]
        : (tuple<tensor<32x32xbf16, #host>, ...>)   -> tensor<32x32xbf16, #host>
    %1 = "ttnn.to_layout"(%0)
        : (tensor<32x32xbf16, #host>) -> tensor<32x32xbf16, #host_staged>
    %2 = "ttnn.to_device"(%1, %arg1)
        : (tensor<32x32xbf16, #host_staged>, !ttnn.device)
          -> tensor<32x32xbf16, #device>
    ```
  }];
}

def TTNNPrepareConv2dWeightsAndBias : Pass<"ttnn-prepare-conv2d-weights-and-bias", "::mlir::ModuleOp"> {
  let summary = "Optimize Conv2d ops by inserting PrepareConv2dWeights and PrepareConv2dBias ops before them.";
  let description = [{
    This pass inserts a PrepareConv2dWeights and a PrepareConv2dBias operation before each Conv2d op, which preprocess the weights and bias used by the Conv2d operations.
    These operations can then be const-evaluated, leading to improved performance.
    To use this pass, the project must be built with the op model library by setting -DTTMLIR_ENABLE_OPMODEL=ON during the build process.
  }];
}

def TTNNDeduceMoEComputeLayouts : Pass<"ttnn-deduce-moe-compute-layouts", "::mlir::ModuleOp"> {
  let summary = "Refine result layouts of the moe_compute weight-prep and compute ops using OpModel.";
  let description = [{
    The prepare_moe_compute_*_weights ops produce a 6-D packed tensor with a
    bank-permuted DRAM HEIGHT_SHARDED memory config, and ttnn.moe_compute
    produces height-sharded outputs on the device compute grid. The exact
    shapes and layouts depend on the target device and the hidden/intermediate
    sizes, so the TTIR caller cannot know them up front. This pass walks those
    ops emerging from TTIR->TTNN conversion and replaces their (placeholder)
    result layouts with the OpModel-derived specs obtained via graph-captured
    query of the corresponding tt-metal invocation.
    Requires -DTTMLIR_ENABLE_OPMODEL=ON.
  }];
}

def TTNNPrepareConv3dWeights : Pass<"ttnn-prepare-conv3d-weights", "::mlir::ModuleOp"> {
  let summary = "Insert PrepareConv3dWeightsOp before every Conv3dOp.";
  let description = [{
    TTIRToTTNN lowers ttir.conv3d to ttnn.conv3d with the raw 5D weight still attached.
    This pass walks every Conv3dOp, derives c_in_block from the optimizer's
    Conv3dConfigAttr (or falls back to TILE_WIDTH = 32), and inserts a
    PrepareConv3dWeightsOp that reshapes the raw 5D weight into the 2D layout the
    tt-metal runtime expects. The Conv3dOp's weight operand is then rewired to the
    prepared result. Mirrors TTNNPrepareConv2dWeightsAndBias.
  }];
}

def TTNNFusing: Pass<"ttnn-fusing", "::mlir::ModuleOp">
{
  let summary = "TTNN fusing pass.";
  let description = "This pass tries to fuse operations together with goal to reduce the number of operations in the graph.";

  let options = [
    Option<"enableOpConstraints",
            "enable-op-constraints",
            "bool", /*default=*/"false",
            "Enable op model constraint validation for fusion patterns. "
            "Requires TTMLIR_ENABLE_OPMODEL. When enabled, fused ops are "
            "validated in an isolated module before being applied.">,
    Option<"maxFallbackAttempts",
            "max-fallback-attempts",
            "uint32_t", /*default=*/"10000",
            "Maximum number of fallback configurations to try before "
            "giving up (0 = unlimited). Only used when "
            "enable-op-constraints is true.">,
    Option<"enableRoPEFusion",
            "enable-rope-fusion",
            "bool", /*default=*/"false",
            "Enable RoPE fusion patterns at the TTNN level. Disabled by "
            "default since RoPE fusion is handled at the TTIR level.">
  ];
}

def TTNNCreateD2MSubgraphs: Pass<"ttnn-create-d2m-subgraphs", "::mlir::ModuleOp"> {
  let summary = "TTNN create D2M subgraphs pass.";
  let description = [{
    This pass identifies potential subgraphs that can be fused through D2M compilation,
    outlines each subgraph into a private function, and replaces the original ops
    with a `ttnn.d2m_subgraph` op. Fusion through the D2M pipeline is done in D2MGenericFusion pass.

    Right now, this is limited to eltwise chains.
  }];
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
      %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
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
      %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
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
  let options = [
    Option<"maxFallbackAttempts",
           "max-fallback-attempts",
           "uint32_t", /*default=*/"10000",
           "Maximum number of fallback configurations to try before giving up (0 = unlimited)">
  ];
}

def TTNNRowMajorLayoutPropagation : Pass<"ttnn-rm-layout-propagation", "::mlir::ModuleOp"> {
  let summary = "Propagate Row-Major layout from inputs through ops";
  let description = [{
    This pass tracks tensors from input arguments through the dataflow graph
    and attempts to maintain Row-Major layout until an operation explicitly
    requires Tile layout. This optimization reduces unnecessary layout conversions.

    The pass starts from function input arguments and follows the tensor dataflow
    through operations. It marks operations to use Row-Major layout when possible,
    stopping propagation at operations that require Tile layout (e.g., matmul, conv).

    This pass should run before the optimizer to provide better initial layout hints.
  }];
}

def TTNNEmitPyWorkarounds: Pass<"ttnn-emitpy-workarounds", "::mlir::ModuleOp"> {
  let summary = "Apply TTNN workarounds specific to EmitPy pipeline.";
  let description = [{
    This pass applies TTNN workarounds that are specific to the EmitPy pipeline.
  }];
}

def TTNNAdjustDeallocs: Pass<"ttnn-adjust-deallocs", "::mlir::ModuleOp"> {
  let summary = "Adjust previously inserted tensor deallocation ops.";
  let description = [{
    This pass will adjust previously inserted tensor deallocation ops. It will
    remove deallocations of parameter and constant tensors.
  }];
}

def TTNNWeightDtypeConversion: Pass<"ttnn-weight-dtype-conversion", "::mlir::ModuleOp"> {
  let summary = "Convert weight tensors to a specified dtype in matmul, linear, and sparse_matmul operations.";
  let description = [{
    This pass converts weight tensors in matrix multiplication, linear, and
    sparse matrix multiplication operations to a specified data type while
    keeping activations and outputs in high precision (bf16/f32).

    For matmul/linear/sparse_matmul operations:
      - Inserts ttnn.typecast operations to convert weights to the target dtype
      - Only converts weights that trace back to constant/parameter arguments
      - Keeps activation inputs and outputs in their original precision
  }];

  list<Option> options = [
    Option<"targetDtype", "target-dtype", "::mlir::tt::ttnn::BFPDtype",
           /*default=*/"::mlir::tt::ttnn::BFPDtype::None",
           "Target data type for weight conversion (bfp_bf8, bfp_bf4).">,
  ];
}

def TTNNCCLActivationDtypeLowering: Pass<"ttnn-ccl-activation-dtype-lowering", "::mlir::ModuleOp"> {
  let summary = "Lower activation precision to bfp_bf8 around CCL ops.";
  let description = [{
    This pass is part of the Mixed Precision effort in compiler.
    This pass reduces ethernet traffic for collective communication (CCL) ops
    in large transformer models by lowering activation precision from bf16 to
    bfp_bf8 across the CCL boundary. Producer matmul ops have their result
    tensor's layout encoding rewritten to bfp_bf8 so no explicit ttnn.typecast is
    required at the producer side. On the consumer side, some patterns allow for
    implicit casting back to bf16 (e.g. residual add). Currently we added only patterns
    that don't require explicit casts on both sides.

    The pass implements two strict per-pattern matchers, each modeled after a
    Llama-style sub-graph:
      - Projection matmul (attention output / MLP down) -> [view / CCL]* ->
        residual add. The residual add's result encoding stays bf16 (it was
        constructed from the bf16 residual operand), so TTNN restores bf16
        at the block output by combining the bfp_bf8 CCL output with the
        bf16 residual — no exit-side cast needed.
      - MLP up / gate matmuls -> [view / CCL]* -> silu / multiply. The
        multiply's result encoding is rewritten to bfp_bf8 so the downstream
        FF2 matmul receives bfp_bf8 input; the FF2 residual add then
        restores bf16 via the projection-residual matcher when FF2 itself is
        lowered.

    A matmul only matches a pattern when the entire downstream chain shape
    matches the template, so the pass under-triggers by design. The idea is
    to add new patterns as we get more multichip models running and fuse patterns
    to more general ones as we get more cases.
  }];
}

def TTNNKVCacheDtypeConversion: Pass<"ttnn-kv-cache-dtype-conversion", "::mlir::ModuleOp"> {
  let summary = "Convert KV cache tensors to a lower precision dtype.";
  let description = [{
    This pass changes KV cache function argument types to a target dtype and inserts
    ttnn.typecast ops on the input operands of ttnn.fill_cache and
    ttnn.update_cache so written data matches the new cache format.
  }];

  list<Option> options = [
    Option<"targetDtype", "target-dtype", "::mlir::tt::ttnn::BFPDtype",
           /*default=*/"::mlir::tt::ttnn::BFPDtype::None",
           "Target data type for KV cache conversion (bfp_bf8, bfp_bf4).">
  ];
}

def TTNNConstEvalInputsToSystemMemory : Pass<"ttnn-force-const-eval-to-system-memory", "::mlir::ModuleOp"> {
  let summary = "Force const-eval function inputs to system memory.";
  let description = [{
    This pass forces all inputs of const-eval functions to be in system memory.

    The pass rewrites const-eval function signatures to change the memory configuration
    of all input tensor arguments to system memory, inserting necessary layout conversion
    ops to transfer the inputs to device memory if needed.

    This pass also modifies the forward functions' arguments that correspond to const-eval
    function inputs, forcing them to system memory as well.

    Note: only the arguments whose sole user is the ttcore.load_cached op are forced to system memory,
    in order to avoid unnecessary to_device transfers for other users.
  }];
}

def TTNNCollectPerfMetrics: Pass<"ttnn-collect-perf-metrics", "::mlir::ModuleOp"> {
  let summary = "Collect and export TTNN operation metrics.";
  let description = [{
    This pass collects metrics about TTNN operations and their memory placement,
    including information about sharded layouts, DRAM spills, and system memory usage.
    The metrics are serialized to JSON format for debugging, profiling, and external analysis.

    Usage:
    Enable performance metrics collection in the TTIR to TTNN runtime pipeline using:
    --ttir-to-ttnn-runtime-pipeline="ttnn-perf-metrics-enabled=true"

    Output file naming:
    - If --ttnn-perf-metrics-output-file is not specified or is the default, files are automatically
      named based on the module/function name and stored in perf_metrics/ directory
    - Example: module @MyModel produces perf_metrics/MyModel_perf_metrics.json
    - If custom output file is specified, that path is used directly

    Metrics collected:
    - total_ops: Total TTNN operations (excluding const-eval ops)
    - total_ops_with_output_tensor: Total TTNN operations that produce output tensors
    - total_shardable_ops: Total TTNN operations that can use sharded layouts
    - sharded_ops: Operations using sharded layouts (height/width/block sharded)
    - effectively_sharded_ops: Sharded ops that are NOT spilled to DRAM
    - sharded_and_spilled_ops: Sharded ops whose output is spilled to DRAM
    - dram_spilled_ops: Total operations spilled to DRAM (interleaved)
    - system_memory_ops: Operations with system_memory buffer type
    - Percentages: All counts as percentage of total_shardable_ops

    Verbose output (--ttnn-perf-metrics-verbose-output-enabled) includes:
    - Detailed per-operation analysis with sharding and memory layout information for shardable ops
    - Complete operation type breakdown with counts for all operation types (excluding const-eval functions)
  }];

  let options = [
    Option<"ttnnPerfMetricsOutputFile", "ttnn-perf-metrics-output-file", "std::string", /*default=*/"\"ttnn_perf_metrics.json\"",
           "Output file path for the performance metrics JSON">,
    Option<"ttnnPerfMetricsVerboseOutputEnabled", "ttnn-perf-metrics-verbose-output-enabled", "bool", /*default=*/"false",
           "Enable verbose output with per-operation details and operation type breakdown">,
    Option<"ttnnEnableTrace", "ttnn-enable-trace", "bool", /*default=*/"false",
           "Collect metrics only for the main traced function">
  ];
}

def TTNNSetComputeKernelConfig: Pass<"ttnn-set-compute-kernel-config", "::mlir::ModuleOp"> {
  let summary = "Set default ComputeKernelConfig attributes for operations.";
  let description = [{
    This pass sets ComputeKernelConfig attributes for TTNN operations that implement
    ComputeKernelConfigOpInterface.

    Pipeline behaviour:
    - If an operation has no compute_config attribute, a new config is created and pipeline
      option values are applied.
    - If a parameter is already set on the operation's compute_config, pipeline options do
      not override it.
    - If a parameter is unset (std::nullopt for math_fidelity, false for bool options),
      the corresponding pipeline option is applied.

    Large-inner-dimension bf16 matmul/linear correctness fix (tenstorrent/tt-metal#46101):
    For `ttnn.matmul` and `ttnn.linear` with inner dimension > 50000 and bf16 output,
    always sets `fp32_dest_acc_en = true` and `packer_l1_acc = true`, even when those
    parameters are already set to false. This rule runs after pipeline option defaults
    and takes precedence over explicit values on the op.

    Example usage:
    ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=hifi4" input.mlir
    ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=lofi fp32-dest-acc-en=true" input.mlir
  }];

  let options = [
    Option<"mathFidelity", "math-fidelity", "::mlir::tt::ttnn::OptionalMathFidelity", /*default=*/"::mlir::tt::ttnn::OptionalMathFidelity::HiFi4",
           "Set default math_fidelity (lofi, hifi2, hifi3, hifi4)">,
    Option<"mathApproxMode", "math-approx-mode", "bool", /*default=*/"false",
           "Set default math_approx_mode">,
    Option<"fp32DestAccEn", "fp32-dest-acc-en", "bool", /*default=*/"true",
           "Set default fp32_dest_acc_en">,
    Option<"packerL1Acc", "packer-l1-acc", "bool", /*default=*/"false",
           "Set default packer_l1_acc">,
    Option<"dstFullSyncEn", "dst-full-sync-en", "bool", /*default=*/"false",
           "Set default dst_full_sync_en">
  ];
}

def TTNNRecoverStructure: Pass<"ttnn-recover-structure", "::mlir::ModuleOp"> {
  let summary = "Recover program structure by splitting IR into multiple functions based on source locations.";
  let description = [{
    This pass analyzes TTNN operations and extracts Python source location information
    from each operation. It then groups operations by their originating Python functions
    and creates new MLIR functions based on these groups, effectively recovering the
    original program structure from the flattened MLIR representation.
  }];
}

def TTNNSimplifyLocsForCodegen: Pass<"ttnn-simplify-locs-for-codegen", "::mlir::ModuleOp"> {
  let summary = "Simplify location information for code generation.";
  let description = [{
    This pass simplifies location information by removing nested locations from NameLocs.
    It only processes TTNN dialect operations and skips UnknownLocs.
  }];
}

def TTNNPrepareConstEvalCaching : Pass<"ttnn-prepare-const-eval-caching", "::mlir::ModuleOp"> {
  let summary = "Prepare const-eval results for caching.";
  let description = [{
    Packs the results of all LoadCachedOps into a single global caching
    dictionary per forward function.  For each forward function containing
    LoadCachedOps it creates a global dictionary (e.g. `ce_cache_forward`),
    retrieves it at the top of the function body, and stores each
    LoadCachedOp result under its callee name.
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect"];
}

def TTNNFileSplit : Pass<"ttnn-file-split", "::mlir::ModuleOp"> {
  let summary = "Organize TTNN IR into FileOp containers before lowering.";
  let description = [{
    Organizes TTNN IR into FileOp containers (emitpy::FileOp or emitc::FileOp,
    selected via the `target` pass option) before lowering to a target dialect.

    Example usage:
    ```bash
    ttmlir-opt --ttnn-file-split="target=emitpy" input.mlir
    ```
  }];

  let dependentDialects = [
    "mlir::tt::emitpy::EmitPyDialect",
    "mlir::emitc::EmitCDialect",
    "mlir::tt::ttcore::TTCoreDialect"
  ];

  let options = [
    Option<"target", "target", "FileSplitTarget",
           /*default=*/"",
           "Target dialect for file splitting",
           [{::llvm::cl::values(
              clEnumValN(FileSplitTarget::EmitPy, "emitpy", "EmitPy target"),
              clEnumValN(FileSplitTarget::EmitC, "emitc", "EmitC target")
           )}]>
  ];
}

def TTNNRemoveDeallocs: Pass<"ttnn-remove-deallocs", "::mlir::ModuleOp"> {
  let summary = "Remove all ttnn.deallocate operations for debugging.";
  let description = [{
    This pass removes all ttnn.deallocate operations from the IR.
    Useful for debugging purposes to isolate issues related to deallocation.
  }];
}

def TTNNResolveComposites : Pass<"ttnn-resolve-composites", "::mlir::ModuleOp"> {
  let summary = "Resolve composite ops into typed ops or inline decompositions.";
  let description = [{
    This pass resolves `ttcore.composite` ops by attempting to replace each one
    with a dedicated typed TTNN op (e.g. `ttnn.rotary_embedding`).

    Resolution mode is controlled by the `composite-resolution` option:

    - `inline` (default): All composites are unconditionally inlined (decomposed
      into primitives). This is the safe default — downstream TTNN fusing
      patterns can still create typed ops from the inlined primitives.

    - `validate`: The typed op is validated via OpModel before promotion. If
      validation fails, the decomposition function body is inlined as a
      fallback. Requires TTMLIR_ENABLE_OPMODEL.

    - `force-promote`: Unconditionally promote registered composites to typed
      ops without validation. Intended for testing only.

    After resolution, decomposition functions that are no longer referenced are
    erased from the module.
  }];

  let options = [
    Option<"compositeResolution",
            "composite-resolution",
            "CompositeResolution",
            /*default=*/"CompositeResolution::Auto",
            "How to resolve composites.",
            [{::llvm::cl::values(
              clEnumValN(CompositeResolution::Auto, "auto",
                "Pipeline decides: validate when optimizer+OpModel available, "
                "else inline (default)."),
              clEnumValN(CompositeResolution::Inline, "inline",
                "Always inline decomposition; never upgraded by pipeline."),
              clEnumValN(CompositeResolution::Validate, "validate",
                "Promote if OpModel validates, else inline "
                "(requires TTMLIR_ENABLE_OPMODEL)."),
              clEnumValN(CompositeResolution::ForcePromote, "force-promote",
                "Unconditionally promote (testing only).")
            )}]>,
  ];
}

def TTNNCollaspeD2M : Pass<"ttnn-collaspe-d2m", "::mlir::ModuleOp"> {
  let summary = "Collapse D2M subgraph modules(s) back into original module.";
  let description = [{
    This pass collapses D2M subgraph modules(s) back into original module.
  }];
}

def TTNNConfigureCCLOps : Pass<"ttnn-configure-ccl-ops", "::mlir::ModuleOp"> {
  let summary = "Configure device-specific parameters on CCL operations.";
  let description = [{
    This pass configures all device-specific parameters on CCL operations
    (AllGather, AllReduce, ReduceScatter) based on the DeviceAttr. This
    includes setting the topology attribute from the per-axis mesh topology.
  }];
}

def TTNNAllocateDistributedOpBuffers
    : Pass<"ttnn-allocate-distributed-op-buffers", "::mlir::ModuleOp"> {
  let summary = "Allocate explicit scratch buffers for distributed ops.";
  let description = [{
    Walks every op implementing `DistributedOpInterface` and invokes
    `allocateBuffers` on it. See `DistributedOpInterface` for the rationale
    behind splitting buffer and semaphore allocation into separate passes
    and for the prelude placement contract.
  }];
}

def TTNNAllocateDistributedOpSemaphores
    : Pass<"ttnn-allocate-distributed-op-semaphores", "::mlir::ModuleOp"> {
  let summary = "Allocate explicit global semaphores for distributed ops.";
  let description = [{
    Walks every op implementing `DistributedOpInterface` and invokes
    `allocateSemaphores` on it. See `DistributedOpInterface` for the
    rationale behind splitting buffer and semaphore allocation into
    separate passes and for the prelude placement contract.
  }];
}

def TTNNMemoryManagement : Pass<"ttnn-memory-management", "::mlir::ModuleOp"> {
  let summary = "Memory usage optimization pass.";
  let description = [{
    This pass moves slice ops and reorganizes repeat ops to reduce memory usage.
  }];
  let options = [
    Option<"aggressiveMemorySaving",
           "aggressive-memory-saving",
           "bool", /*default=*/"true",
           "Opt-out: disable aggressive memory saving patterns to reduce DRAM usage.">,
  ];
}

def TTNNGreedyL1SpillManagement : Pass<"ttnn-greedy-l1-spill-management", "::mlir::ModuleOp"> {
  let summary = "Farthest-last-use eviction for L1 budget enforcement.";
  let description = [{
    This pass enforces L1 memory budgets by spilling tensors to DRAM using a
    farthest-last-use eviction strategy.
  }];
  let options = [
    Option<"enableDecisionTrace", "enable-decision-trace", "bool", "false",
           "Enable decision trace JSON output.">,
    Option<"decisionTraceDir", "decision-trace-dir", "std::string",
           "\"ttrt-artifacts/decision_trace\"",
           "Directory for decision trace JSON output.">,
  ];
}

def TTNNGreedyMemoryLayoutPropagation : Pass<"ttnn-greedy-memory-layout-propagation", "::mlir::ModuleOp"> {
  let summary = "Greedy edge-based memory layout propagation for all operands.";
  let description = [{
    This pass performs edge-based layout propagation using a greedy algorithm
    or beam search to find optimal tensor layouts across the module.
  }];
  let options = [
    Option<"maxLegalLayouts", "max-legal-layouts", "int64_t", "64",
           "Override maximum number of sharded layouts for legal layout analysis.">,
    Option<"rowMajorEnabled", "row-major-enabled", "bool", "false",
           "Enable row major layout generation in legal layout analysis.">,
    Option<"beamWidth", "beam-width", "int64_t", "8",
           "Beam width for layout propagation (1=greedy, >1=beam search).">,
    Option<"enableL1ShardingLayouts", "enable-l1-sharding-layouts",
           "bool", "true",
           "Enable L1 sharding layouts. Disabled at opt-level 1.">,
    Option<"maxInputCandidatesPerOperand", "max-input-candidates-per-operand",
           "int64_t", "64",
           "Maximum number of input candidates per operand before truncation.">,
    Option<"maxReshardCandidatesPerType", "max-reshard-candidates-per-type",
           "int64_t", "4",
           "Maximum reshard candidates per sharding type (block, height, width).">,
    Option<"enableDecisionTrace", "enable-decision-trace", "bool", "false",
           "Enable decision trace JSON output.">,
    Option<"decisionTraceDir", "decision-trace-dir", "std::string",
           "\"ttrt-artifacts/decision_trace\"",
           "Directory for decision trace JSON output.">,
    Option<"enableCompileTimeStats", "enable-compile-time-stats", "bool", "false",
           "Print per-op compile-time statistics at DEBUG level.">,
  ];
}

def TTNNResolveTTLangKernels : Pass<"ttnn-resolve-tt-lang-kernels", "::mlir::ModuleOp"> {
  let summary = "Resolve deferred ttnn.tt_lang_op kernels via the tt-lang Python compiler.";
  let description = [{
    This pass walks the module for `ttnn.tt_lang_op` operations with an
    empty `kernel_artifact` attribute and invokes
    `tt_torch.tt_lang.resolve_operation` through the host process's Python
    interpreter, passing the (now-final, post-TTIR->TTNN) shard-local
    operand types and `arg_roles` / `shard_spec`. The
    `tt_torch.tt_lang.resolve_operation` resolver lives in tt-xla's
    `tt_torch` Python package and returns a `CompiledTTNNKernel` JSON
    artifact, which the pass attaches back onto the op as the
    `kernel_artifact` `StringAttr`. This attribute is later consumed by
    the downstream TTNN-to-flatbuffer emitter to materialise a
    `ttnn.generic_op` descriptor.

    Important Notes:
    This pass must be run after all TTNN layout-decomposition and
    workaround passes (so each operand carries its final
    `#ttnn.ttnn_layout` encoding) and before flatbuffer emission.

    The Python interpreter's GIL is acquired via
    `pybind11::gil_scoped_acquire`; this pass does NOT call
    `Py_Initialize`. The host process is expected to already be running
    CPython (typical when this pass runs inside `pjrt_plugin_tt.so`,
    loaded from a `jax` / `torch_xla` Python program). When
    `ttmlir-translate` or `ttmlir-opt` is invoked from a non-Python
    parent the `tt_torch.tt_lang` import fails and the pass emits a
    diagnostic to that effect. A module with no `ttnn.tt_lang_op` is
    unaffected: the pass returns before touching Python at all.
  }];

  let options = [
    Option<"meshShape", "mesh-shape", "std::string",
           /*default=*/"\"\"",
           "Comma-separated mesh shape (e.g. 1,1 or 2,4) "
           "forwarded to `resolve_operation` so the Python compiler sees "
           "the active mesh. Empty defaults to [1].">,
    Option<"pythonModule", "python-module", "std::string",
           /*default=*/"\"tt_torch.tt_lang\"",
           "Python module exposing the resolver entry point.">,
    Option<"pythonFunction", "python-function", "std::string",
           /*default=*/"\"resolve_operation\"",
           "Function in `python-module` to invoke per op.">,
  ];
}

def TTNNLowerTTLangToGeneric : Pass<"ttnn-lower-tt-lang-to-generic", "::mlir::ModuleOp"> {
  let summary = "Lower resolved ttnn.tt_lang_op kernels into ttnn.generic ops.";
  let description = [{
    This pass rewrites each `ttnn.tt_lang_op` whose `kernel_artifact`
    attribute has been populated (by `--ttnn-resolve-tt-lang-kernels`)
    into an equivalent `ttnn.generic` op carrying a `#ttnn.program`
    descriptor.

    The `kernel_artifact` is the UTF-8 JSON payload produced by tt-xla's
    `tt_torch.tt_lang._serialize_compiled_operation`:

    {
      "format_version": 1,
      "kernels": [
        {"thread_type": "compute"|"noc",
         "cpp_source":  "<embedded C++ source>",
         "tensor_indices": [<int>, ...],
         "kernel_config": {...}},
        ...
      ],
      "core_range": {"start": [x, y], "end": [x, y]},
      "cb_configs": [{buffer_index, data_format, page_size, total_size}, ...],
      ...
    }

    Each kernel becomes a `#ttnn.compute_kernel` / `#ttnn.read_kernel` /
    `#ttnn.write_kernel` attribute carrying the kernel's C++ source inline
    (the `source` parameter), its CB-index / tensor-accessor compile-time
    args, and its buffer-address common runtime args. `cb_configs` become
    `#ttnn.kernel_cb` descriptors.

    This pass must run after `--ttnn-resolve-tt-lang-kernels` and before
    flatbuffer emission. A `ttnn.tt_lang_op` with an empty/unset
    `kernel_artifact` is not allowed and will result in an error.
  }];
}

#endif
