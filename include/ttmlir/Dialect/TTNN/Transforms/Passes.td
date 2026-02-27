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

def TTNNPrepareModuleForExport : Pass<"ttnn-prepare-module-for-export", "::mlir::ModuleOp"> {
  let summary = "Prepare module for Python module export.";
  let description = [{
    This pass prepares the module for export as a Python module by:
    1. Renaming the first public (non-private, non-const-eval) function to "forward"
    2. Adding a device argument to the function signature
    3. Replacing all ttnn.get_device ops with the device function argument

    Given a forward function like this:

    ```mlir
    func.func @add(%arg0: tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tuple<tensor<32x32xbf16>> {
      %0 = ttcore.get_tuple_element %arg0[0] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %1 = ttcore.get_tuple_element %arg0[1] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %2 = "ttnn.get_device"() : () -> !ttnn.device
      %3 = "ttnn.to_device"(%0, %2) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %4 = "ttnn.to_device"(%1, %2) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %5 = "ttnn.add"(%3, %4) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %6 = ttcore.tuple %5 : tuple<tensor<32x32xbf16>>
      return %6 : tuple<tensor<32x32xbf16>>
    }
    ```

    The pass will return:

    ```mlir
    func.func @forward(%arg0: tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>, %arg1: !ttnn.device) -> tuple<tensor<32x32xbf16>> {
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

  let options = [
    Option<"enableOpConstraints",
            "enable-op-constraints",
            "bool", /*default=*/"false",
            "When enabled fusing pass includes patterns which check for op constraints.">
  ];
}

def TTNND2MFusing: Pass<"ttnn-d2m-fusing", "::mlir::ModuleOp"> {
  let summary = "TTNN D2M fusing pass.";
  let description = [{
    This pass identifies potential subgraphs that can be fused through D2M compilation.

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
  let options = [
    Option<"tensorL1UsageCap",
           "tensor-l1-usage-cap",
           "float", /*default=*/"1.0",
           "Limit L1 memory usage to this fraction of available space (0.0-1.0)">,
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

  let options = [
    Option<"tensorL1UsageCap",
           "tensor-l1-usage-cap",
           "float", /*default=*/"1.0",
           "Limit L1 memory usage to this fraction of available space (0.0-1.0)">
  ];
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
  let summary = "Convert weight tensors to a specified dtype in matmul and linear operations.";
  let description = [{
    This pass converts weight tensors in matrix multiplication and linear
    operations to a specified data type while keeping activations and outputs
    in high precision (bf16/f32).

    For matmul/linear operations:
      - Inserts ttnn.typecast operations to convert weights to the target dtype
      - Only converts weights that trace back to constant/parameter arguments
      - Keeps activation inputs and outputs in their original precision
  }];

  list<Option> options = [
    Option<"targetDtype", "target-dtype", "std::string", /*default=*/"\"\"",
           "Target data type for weight conversion (e.g. bfp_bf8, bfp_bf4).">,
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
    Enable performance metrics collection in the TTIR to TTNN backend pipeline using:
    --ttir-to-ttnn-backend-pipeline="ttnn-perf-metrics-enabled=true"

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
    This pass sets default ComputeKernelConfig attributes for TTNN operations that support
    the ComputeKernelConfigOpInterface. It only sets parameter values that are not already
    defined, preserving any existing configuration decisions made by earlier compiler passes.

    Behaviour:
    - If an operation has no compute_config attribute, a new compute config is created with override values
    - If an operation has a compute_config but a specific parameter (e.g., math_fidelity) is
      unset (std::nullopt), the override value is applied for that parameter only
    - If a parameter is already set in the compute_config, it is not overridden, preserving
      the existing compiler decision

    This allows setting default values for ComputeKernelConfig parameters while respecting
    any explicit configuration choices made by prior passes.

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

def TTNNRemoveDeallocs: Pass<"ttnn-remove-deallocs", "::mlir::ModuleOp"> {
  let summary = "Remove all ttnn.deallocate operations for debugging.";
  let description = [{
    This pass removes all ttnn.deallocate operations from the IR.
    Useful for debugging purposes to isolate issues related to deallocation.
  }];
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

#endif
