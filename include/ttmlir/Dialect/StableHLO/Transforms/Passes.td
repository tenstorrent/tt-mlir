// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_StableHLO_DIALECT_PASSES_TD
#define TTMLIR_StableHLO_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"
include "shardy/dialect/sdy/ir/dialect.td"
include "shardy/dialect/sdy/ir/op_interface.td"

def StableHLOComplexDataTypeConversionPass : Pass<"stablehlo-complex-data-type-conversion", "::mlir::ModuleOp"> {
  let summary = "Convert complex data types to float pair representations in StableHLO.";
  let description = [{
    This pass converts StableHLO operations with complex element types to
    equivalent operations using float pair representations. Complex tensors of
    type tensor<...xcomplex<fN>> are converted to tensor<...x2xfN>, where the
    last dimension stores [real, imag] components.

    Complex specific ops, such as stablehlo.complex, stablehlo.real, and stablehlo.imag transpose
    the trailing dimension to the front for tilize/ untilize performance.

    Concretely, the pass handles:
      - stablehlo.complex -> stablehlo.reshape + stablehlo.concatenate + stablehlo.transpose
      - stablehlo.real / stablehlo.imag -> stablehlo.transpose + stablehlo.slice + stablehlo.reshape
      - stablehlo.constant with complex element type -> float-pair constant
      - stablehlo.reshape with complex result type -> float-pair reshape
      - stablehlo.broadcast_in_dim with complex result type -> float-pair broadcast_in_dim
      - func signatures and return ops with complex types

    This pass is intended to run after `stablehlo-complex-math-expander` (real-domain elementwise ops).
    To add support for elementwise ops with complex types, please register them in `stablehlo-complex-math-expander` pass.
  }];
  let dependentDialects = ["mlir::func::FuncDialect",
                           "::mlir::stablehlo::StablehloDialect",
                           "::mlir::sdy::SdyDialect"];
}

def RegisterCustomShardingRulePass : Pass<"register-custom-sharding-rule", "::mlir::ModuleOp">
{
  let summary = "Register custom sharding rules for operations.";

  let description = [{
    This pass registers custom sharding rules in the current MLIRContext so that
    Shardy can correctly propagate shardings through operations that do not
    have built-in sharding rules in Shardy.

    Currently, this pass attaches the `ShardingRuleOpInterface` external model
    implementation for `stablehlo.CustomCallOp`, allowing Shardy to query
    `ShardingRuleOpInterface::getShardingRule` for custom calls that are not
    covered by Shardy's built-in StableHLO sharding registry.

    Over time, additional custom operations that require project-specific
    sharding rules can also be registered in this pass by attaching their own
    `ShardingRuleOpInterface` implementations.

    This pass is intended to run before any Shardy sharding propagation passes.

    For more details on the interface and how sharding rules are expressed,
    please refer to:
      https://github.com/openxla/shardy/blob/80ade881ade0c7c0e4d788cca5b7b497f4dc249b/docs/sdy_op_interfaces.md#shardingruleopinterface-sdy_shardingruleopinterface
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}
def ApplyArgumentShardStatusPass : Pass<"apply-argument-shard-status", "::mlir::ModuleOp">
{
  let summary = "Annotate arguments with their shard status.";
  let description = [{
    This pass will analyze the module arguments and annotate arguments with whether they are pre-sharded or not.
    For argument, if a sdy.sharding or mhlo.sharding exists, it is presharded.
    For results, the user must pass that as a compiler option.

    Example, this pass will convert the following code:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "batch"=8]>
      func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch"}, {}, {}, {}\]>}) -> (tensor<2048x1024xf32>) {
        %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
        return %0 : tensor<2048x1024xf32>
      }
    }
    ```

    Into:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "batch"=8]>
      func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch"}, {}, {}, {}\]>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<2048x1024xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
        return %0 : tensor<2048x1024xf32>
      }
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect",
    "::mlir::tt::ttir::TTIRDialect"
  ];

  let options = [
    ListOption<"resultPresharded", "result-presharded", "int64_t", "Set whether each result is presharded or not">,
  ];
}

def ConvertXlaSdyToSdyPass : Pass<"convert-xla-sdy-to-sdy", "::mlir::ModuleOp">
{
  let summary = "Convert xla sdy annotations into sdy annotations.";
  let description = [{
    This pass will convert any xla.sdy annotations into sdy annotations. This is required due to some lingering code from torch-xla that does not do this for us (whereas xla does).

    Example, this pass will convert the following code:
      ```mlir
      module @SyncTensorsGraph.5 attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
          func.func @main(%arg0: tensor<32x128xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, \[{}, {_axis_0}\]>"}, mhlo.sharding = {devices=\[1,2\]<=\[2\]}"}, %arg1: tensor<32x32xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, \[{}, {}\]>"}, mhlo.sharding = {replicated}"}) -> tensor<32x128xf32> {
            %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = \[1\] x \[0\] : (tensor<32x32xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
          return %0 : tensor<32x128xf32>
        }
      }
      ```

      Into:
      ```mlir
      module @SyncTensorsGraph.5 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
        sdy.mesh @mesh = <["_axis_0_updated"=1, _axis_0=2]>
        func.func @main(%arg0: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{?}, {_axis_0, ?}\]>}, %arg1: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{?}, {?}\]>}) -> tensor<32x128xf32> {
          %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
          return %0 : tensor<32x128xf32>
        }
      }
      ```
    }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect",
    "::mlir::tt::ttir::TTIRDialect"
  ];
}

def StableHLOFusingPass : Pass<"stablehlo-fusing", "::mlir::ModuleOp">
{
  let summary = "StableHLO fusing pass for shardy passes optimization.";
  let description = [{
    This pass tries to fuse stablehlo operations together with the goal of optimizing shardy passes.
    If proposed fusion is not optimizing shardy passes, it should be implemented in lower level passes. i.e. in TTIR/ TTNN fusion passes.

    For example, it implements `ConcatenateToBroadcastInDimFusionPattern` pattern which fuses stablehlo.concatenate
    and stablehlo.reshape into stablehlo.broadcast_in_dim. This is due to the fact that
    stablehlo.broadcast_in_dim propagates shardings better during UserPriorityPropagationPass and InsertExplicitReshardsPass.
    See https://github.com/openxla/shardy/issues/945 for more details.

    Example, this pass will convert the following code:
    ```mlir
      %0 = stablehlo.concatenate %arg0, %arg0, %arg0, dim = 0 :
            (tensor<2x4xbf16>, tensor<2x4xbf16>, tensor<2x4xbf16>) ->
            tensor<6x4xbf16>
      %1 = stablehlo.reshape %0 : (tensor<6x4xbf16>) -> tensor<3x2x4xbf16>
    ```

    Into:
    ```mlir
      %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] :
            (tensor<2x4xbf16>) -> tensor<3x2x4xbf16>
    ```
  }];

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect"
  ];
}

def PartiallyConvertSdyToStableHLOPass : Pass<"partially-convert-sdy-to-stablehlo", "::mlir::ModuleOp">
{
  let summary = "Partially convert Shardy operations to StableHLO operations.";
  let description = [{
    This pass partially converts Shardy operations to their StableHLO equivalents.
    This is needed because tt-xla may receive graphs from XLA containing Shardy operations,
    but tt-mlir does not need or handle them beyond sharding annotations.

    Currently, this pass converts sdy.constant operations to stablehlo.constant operations.

    Example, this pass will convert the following code:
    ```mlir
    func.func @main() -> tensor<2x3xf32> {
      %0 = sdy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
      return %0 : tensor<2x3xf32>
    }
    ```

    Into:
    ```mlir
    func.func @main() -> tensor<2x3xf32> {
      %0 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
      return %0 : tensor<2x3xf32>
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
    "::mlir::sdy::SdyDialect"
  ];
}

def AnalyzeMeshPass : Pass<"analyze-mesh", "::mlir::ModuleOp">
{
  let summary = "Analyze the mesh compilation target.";
  let description = [{
    This pass will analyze the mesh of the graph and update shardings or annotations to match the target device. It will attempt to automatically parallelize if user provided option and mesh shape.
    For GSPMD graphs, it will analyze the mhlo annotations and attempt to deduce the mesh shape from them.
    For shardy based graphs, it will check for the existence of a shardy mesh, from which it will query the mesh target.
    If no mesh is present, it will assume a single chip graph and insert a 1x1 mesh.

    Example, this pass will convert the following code:
    ```mlir
    func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
        %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
        %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
        %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
        return %2 : tensor<1x128xf32>
    }
    ```

    Into:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "y"=1]>
      func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
        %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
        %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
        %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
        return %2 : tensor<1x128xf32>
      }
    }
    ```
  }];

  let options = [
    ListOption<"meshShape", "mesh-shape", "int64_t", "Set the mesh shape">,
    Option<"automaticArgAnalysis", "automatic-arg-analysis", "bool", /*default=*/"false", "Automatically determine argument shardings">,
  ];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect",
    "::mlir::tt::ttir::TTIRDialect"
  ];
}

def AutoShardingPass : Pass<"auto-sharding", "::mlir::ModuleOp"> {
  let summary = "Automatically search for optimal sharding configuration by evaluating CCL costs";
  let description = [{
    This pass enumerates possible sharding configurations for function arguments,
    runs the remaining StableHLO passes (propagation, reshard-to-collectives,
    canonicalization) on each variant, evaluates a cost function based on the
    resulting Shardy CCL operations, and selects the configuration with the
    lowest cost.

    The pass runs after AnalyzeMesh and before shardy propagation passes.
    For each candidate sharding, it clones the module, applies sharding hints,
    runs the remaining StableHLO passes, and counts/costs the Shardy CCL ops
    (sdy.all_gather, sdy.reduce_scatter, sdy.all_reduce, etc.) in the result.

    The winning sharding configuration is applied to the original module, which
    then continues through the normal pipeline.
  }];

  let options = [
    Option<"dumpVariants", "dump-variants", "bool",
           /*default=*/"false",
           "Dump each sharding variant IR to disk for inspection">,
    Option<"dumpDir", "dump-dir", "std::string",
           /*default=*/"\"/tmp/\"",
           "Directory to dump variants into (default: /tmp/)">,
  ];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def DecoupleConstFanoutPass : Pass<
    "decouple-const-fanout", "::mlir::ModuleOp"> {
  let summary = "Duplicate const-fed, value-preserving ops per-use to decouple propagation";
  let description = [{
    Duplicate const-fed, value-preserving ops per-use to avoid cross-path
    coupling in sharding/shape propagation. Runs before sharding passes.

    Initial target: `stablehlo.broadcast_in_dim` whose operand is a constant
    (or rank-0 scalar) and whose result has multiple uses.

    This keeps each consumer path independent, preventing one path's sharding
    needs (e.g., via `maximum`) from affecting another path (e.g., `add` with
    replicated bias).
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::stablehlo::StablehloDialect"
  ];
}

def DecomposeCustomCallTuplesPass : Pass<"decompose-custom-call-tuples", "::mlir::ModuleOp"> {
  let summary = "Eliminate stablehlo.tuple ops by forwarding operands to get_tuple_element users.";
  let description = [{
    This pass finds `stablehlo.tuple` operations and replaces all corresponding
    `stablehlo.get_tuple_element` users with direct references to the tuple's
    operands. The tuple op is erased if all uses are replaced.

    This is required because Shardy's sharding propagation does not support
    tuple types and they must be eliminated before sharding propagation can run.

    Example, this pass will convert the following code:
    ```mlir
    %0 = stablehlo.tuple %arg0, %arg1 : tuple<tensor<3x3xf32>, tensor<3xf32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3x3xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3xf32>
    ```

    Into:
    ```mlir
    // %1 is replaced by %arg0, %2 is replaced by %arg1
    ```
  }];

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect"
  ];
}

def RewriteCompositeDecompFunctionsPass : Pass<"rewrite-composite-decomp-functions", "::mlir::ModuleOp"> {
  let summary = "Rewrite bodies of decomposition functions referenced by targeted stablehlo.composite ops.";
  let description = [{
    For each `stablehlo.composite` op whose name is present in an internal
    registry (composite name -> rewriter function), resolves the composite's
    `decomposition` SymbolRefAttr to a `func::FuncOp` and invokes the
    registered rewriter on that function. The rewriter replaces the
    decomposition function's body while preserving its name, input types, and output
    types.

    Must run after `decompose-custom-call-tuples` and before
    `flatten-or-convert-composites`, so the new body is what gets inlined.
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::stablehlo::StablehloDialect"
  ];
}

def FlattenOrConvertCompositesPass : Pass<"flatten-or-convert-composites", "::mlir::ModuleOp"> {
  let summary = "Flatten or convert stablehlo.composite ops depending on whether they have custom sharding rules.";
  let description = [{
    This pass processes each `stablehlo.composite` op in one of two ways:

    1. Composites with custom sharding rules (listed in kCompositesWithCustomSharding)
       are converted to `stablehlo.custom_call` ops. The composite name becomes the
       `call_target_name`, composite attributes are carried as a
       `tt.composite_attributes` discardable attribute, and a `tt.has_custom_sharding`
       UnitAttr is used to tag the op.

    2. All other composites are inlined (flattened) by cloning the callee body at the
       call site. Every cloned op is annotated with `reoutline.group = "composite_<callee>"`.
       The first cloned op additionally receives a `reoutline.seed` (UnitAttr).
       After rewiring the cloned return values to the original users, the composite op
       itself is erased.

    This pass is intended to run before sharding propagation so that propagation
    occurs on a fully flattened graph for generic composites, while composites with
    custom sharding rules become custom_calls that Shardy can propagate through
    directly. A later re-outlining pass may reconstruct flattened composite ops by
    grouping operations with the same `reoutline.group` attribute, provided that the
    subgraph is closed and movable.

    This pass also checks whether the arguments and results of the composite are sharded,
    and performs inlining only if sharding (not replicated) is detected.

    Example, this pass will convert the following code:
    ```mlir
      module @Composite attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
        sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
        func.func @main(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{}, {"_axis_0"}\]>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
          %0 = stablehlo.composite "tt.some_composite" %arg0 {composite_attributes = {approximate = "tanh"}, decomposition = @tt.some_composite.impl} : (tensor<32x32xf32>) -> tensor<32x32xf32>
          return %0 : tensor<32x32xf32>
        }
        func.func private @tt.some_composite.impl(%arg0: tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
          %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
          %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
          %1 = stablehlo.multiply %arg0, %0 : tensor<32x32xf32>
          return %1 : tensor<32x32xf32>
        }
      }
    ```

    Into:
    ```mlir
    module @Composite attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
      sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
      func.func @main(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{}, {"_axis_0"}\]>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %cst = stablehlo.constant {reoutline.comp_attrs = {approximate = "tanh"}, reoutline.group = "composite_tt.some_composite.impl", reoutline.orig_name = "tt.some_composite", reoutline.seed} dense<5.000000e-01> : tensor<f32>
        %0 = stablehlo.broadcast_in_dim %cst, dims = [] {reoutline.group = "composite_tt.some_composite.impl"} : (tensor<f32>) -> tensor<32x32xf32>
        %1 = stablehlo.multiply %arg0, %0 {reoutline.group = "composite_tt.some_composite.impl"} : tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
      }
    }
    ```
  }];


  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::stablehlo::StablehloDialect"
  ];
}

def WrapUnderManualComputationPass : Pass<"wrap-under-manual-computation", "::mlir::ModuleOp">
{
  let summary = "Wrap all operations within a sdy manual computation op.";
  let description = [{
    This pass will wrap all the operations within a module under a manual computation op that defines per device tensor shapes.

    Example, this pass will convert the following code:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "y"=1]>
      func.func @add(%arg0: tensor<32x48x24x32xf32>, %arg1: tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<32x48x24x32xf32>
        return %0 : tensor<32x48x24x32xf32>
      }
    }
    ```

    Into:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "y"=1]>
      func.func @add(%arg0: tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{?}, {?}, {?}, {?}\]>}, %arg1: tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{?}, {?}, {?}, {?}\]>}) -> tensor<32x48x24x32xf32> {
        %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=\[<@mesh, \[{?}, {?}, {?}, {?}\]>, <@mesh, \[{?}, {?}, {?}, {?}\]>\] out_shardings=\[<@mesh, \[{?}, {?}, {?}, {?}\]>\] manual_axes={} (%arg2: tensor<32x48x24x32xf32>, %arg3: tensor<32x48x24x32xf32>) {
          %1 = stablehlo.add %arg2, %arg3 : tensor<32x48x24x32xf32>
          sdy.return %1 : tensor<32x48x24x32xf32>
        } : (tensor<32x48x24x32xf32>, tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32>
        return %0 : tensor<32x48x24x32xf32>
      }
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def AnnotateLocalShapesPass : Pass<"annotate-local-shapes", "::mlir::ModuleOp">
{
  let summary = "Annotate all arguments and results with their local shape.";
  let description = [{
    This pass will iterate through all arguments and results and annotate the local shape of that argument/result.
    For single device graphs, the local shape is the original shape.
    For multi-device graphs, the local shape can be sharded or replicated depending on the sharding config set in the graph.
    This information is required by any tool calling ttmlir runtime to tell it how the inputs should be preprocessed.

    Example, this pass will convert the following code:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "batch"=8]>
      func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch"}, {}, {}, {}\]>}) -> (tensor<2048x1024xf32>) {
        %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
        return %0 : tensor<2048x1024xf32>
      }
    }
    ```

    Into:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "batch"=8]>
      func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch"}, {}, {}, {}\]>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x2x32x32xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<2048x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch", ?}, {?}\]>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x1024xf32>>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = sdy.manual_computation(%arg0) in_shardings=\[<@mesh, \[{"batch"}, {}, {}, {}\]>\] out_shardings=\[<@mesh, \[{"batch", ?}, {?}\]>\] manual_axes={} (%arg1: tensor<1024x2x32x32xf32>) {
          %1 = stablehlo.reshape %arg1 {sdy.sharding = #sdy.sharding_per_value<\[<@mesh, \[{"batch", ?}, {?}\]>\]>} : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
          sdy.return %1 : tensor<2048x1024xf32>
        } : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
        return %0 : tensor<2048x1024xf32>
      }
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def UpdateGlobalToLocalShapesPass : Pass<"update-global-to-local-shapes", "::mlir::ModuleOp">
{
  let summary = "Update all the tensor dimensions based on their tensor sharding annotation from global shapes to local per device shapes.";
  let description = [{
    This pass will update all tensor dimensions from global to local shapes.

    Example, this pass will convert the following code:
    ```mlir
    module {
      sdy.mesh @mesh = <["model"=1, "batch"=2]>
      func.func @full_arg_annotation(%arg0: tensor<64x128xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch"}, {"model"}\]>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<64x128xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch", ?}, {"model", ?}\]>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = sdy.manual_computation(%arg0) in_shardings=\[<@mesh, \[{"batch"}, {"model"}\]>\] out_shardings=\[<@mesh, \[{"batch", ?}, {"model", ?}\]>\] manual_axes={} (%arg1: tensor<64x128xf32>) {
          %1 = stablehlo.cbrt %arg1 {sdy.sharding = #sdy.sharding_per_value<\[<@mesh, \[{"batch", ?}, {"model", ?}\]>\]>} : tensor<64x128xf32>
          sdy.return %1 : tensor<64x128xf32>
        } : (tensor<64x128xf32>) -> tensor<64x128xf32>
        return %0 : tensor<64x128xf32>
      }
    }
    ```

    Into:
    ```mlir
    module {
      sdy.mesh @mesh = <["model"=1, "batch"=2]>
      func.func @full_arg_annotation(%arg0: tensor<64x128xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<64x128xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = sdy.manual_computation(%arg0) in_shardings=\[<@mesh, \[{"batch"}, {"model"}\]>\] out_shardings=\[<@mesh, \[{"batch", ?}, {"model", ?}\]>\] manual_axes={"model", "batch"} (%arg1: tensor<32x128xf32>) {
          %1 = stablehlo.cbrt %arg1 : tensor<32x128xf32>
          sdy.return %1 : tensor<32x128xf32>
        } : (tensor<64x128xf32>) -> tensor<64x128xf32>
        return %0 : tensor<64x128xf32>
      }
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def ReplicateNonSplittableConstantsPass : Pass<"replicate-non-splittable-constants", "::mlir::ModuleOp">
{
  let summary = "Change sharding of non-splittable constants to replicated.";
  let description = [{
    This pass identifies stablehlo.constant operations whose values are
    non-splat and non-periodic across the sharding dimension, meaning they
    cannot be meaningfully sharded in SPMD (all devices run the same program
    and would need to split constant data that has no repeating pattern).

    For such constants, the pass changes their sharding annotation to fully
    replicated (all dimensions closed, no sharding axes). This must run
    before InsertExplicitReshards so that Shardy detects the mismatch between
    the replicated constant and its sharded consumers, and inserts the
    appropriate sdy.reshard operations. Those reshards are later converted to
    concrete collective operations by ReshardToCollectives.

    Without this pass, UpdateGlobalToLocalShapes would try to shard the
    constant value, which fails for non-splat, non-periodic data.

    Example constants that are non-splittable:
      - dense<[0, 1, 2, ..., 63]> : tensor<64xi64>  (iota / position indices)
      - dense<[[[0.0, 1.0, ..., 63.0]]]> : tensor<1x1x64xf32>  (RoPE freqs)
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def InsertExplicitReshardsPass : Pass<"insert-explicit-reshards", "::mlir::ModuleOp">
{
  let summary = "Insert explicit reshards only on non-solved graphs.";
  let description = [{
    This pass checks if the graph is already solved before running sdy.insert_explicit_reshards.
    A graph is considered solved if it contains sdy.manual_computation operations with non-empty
    manual_axes, indicating that the sharding propagation has already been completed.
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def ShardyCCLCanonicalizationPass : Pass<"shardy-ccl-canonicalization", "::mlir::ModuleOp">
{
  let summary = "Canonicalize Shardy CCL operations.";
  let description = [{
    This pass canonicalizes Shardy collective communication operations by fusing / rewriting
    patterns that can be represented more efficiently.
  }];
  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def ReoutlineCompositePass : Pass<"reoutline-composite", "::mlir::ModuleOp"> {
  let summary = "Reconstruct composite ops from flattened groups marked by reoutline.* attributes.";
  let description = [{
    This pass scans for ops annotated with `reoutline.group` and attempts to
    re-outline them into a single composite/call if the group forms a closed,
    movable subgraph within the function. Captured inputs become function arguments
    and escaping values become function results. If a group is not closed or cannot
    be moved safely due to side effects, it is skipped and the flattened form remains.

    Use this after sharding propagation and reshards insertion if you want to
    restore composite structure for subsequent conversions.

    Example, this pass will convert the following code:
    ```mlir
    module @Reoutline attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
      sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
      func.func @main(%arg0: tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %cst = stablehlo.constant {reoutline.comp_attrs = {approximate = "tanh"}, reoutline.group = "composite_tt.some_composite.impl", reoutline.orig_name = "tt.some_composite", reoutline.seed} dense<5.000000e-01> : tensor<f32>
        %0 = stablehlo.broadcast_in_dim %cst, dims = [] {reoutline.group = "composite_tt.some_composite.impl"} : (tensor<f32>) -> tensor<32x32xf32>
        %1 = stablehlo.multiply %arg0, %0 {reoutline.group = "composite_tt.some_composite.impl"} : tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
      }
    }
    ```

    Into:
    ```mlir
    module @Reoutline attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
      sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
      func.func @main(%arg0: tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = stablehlo.composite "tt.some_composite" %arg0 {composite_attributes = {approximate = "tanh"}, decomposition = @outlined_composite_tt.some_composite.impl} : (tensor<32x32xf32>) -> tensor<32x32xf32>
        return %0 : tensor<32x32xf32>
      }
      func.func private @outlined_composite_tt.some_composite.impl(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %cst = stablehlo.constant {reoutline.comp_attrs = {approximate = "tanh"}, reoutline.orig_name = "tt.some_composite"} dense<5.000000e-01> : tensor<f32>
        %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %1 = stablehlo.multiply %arg0, %0 : tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
      }
    }
    ```

    But this pass wouldn't re-outline if the group is not closed:
    ```mlir
    module @ReoutlineNotInGroup attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
      sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
      func.func @main(%arg0: tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %cst = stablehlo.constant {reoutline.comp_attrs = {approximate = "tanh"}, reoutline.group = "composite_tt.some_composite.impl", reoutline.orig_name = "tt.some_composite", reoutline.seed} dense<5.000000e-01> : tensor<f32>
        %0 = stablehlo.broadcast_in_dim %cst, dims = [] {reoutline.group = "composite_tt.some_composite.impl"} : (tensor<f32>) -> tensor<32x32xf32>
        %1 = stablehlo.add %arg0, %0 : tensor<32x32xf32> // not in group
        %2 = stablehlo.multiply %arg0, %1 {reoutline.group = "composite_tt.some_composite.impl"} : tensor<32x32xf32>
        return %2 : tensor<32x32xf32>
      }
    }
    ```
    }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def FuseDistributedCustomCallsPass : Pass<"fuse-distributed-custom-calls", "::mlir::ModuleOp">
{
  let summary = "Fuse custom_call ops with surrounding CCL ops into distributed op variants.";
  let description = [{
    This pass pattern-matches custom_call ops marked with `tt.has_custom_sharding`
    which are surrounded by CCL ops and fuses them into a single distributed
    custom_call variant, that handles cross-device communication internally.

    Example.
    Before fusing:
      all_gather(local_input) -> gathered_input
      all_gather(local_weight) -> gathered_weight
      custom_call @tenstorrent.rms_norm(gathered_input, gathered_weight) -> gathered_result
      sdy.all_slice(gathered_result) -> local_result
    After fusing:
      custom_call @tenstorrent.distributed_rms_norm(local_input, local_weight) -> local_result
  }];

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect"
  ];
}

#endif
