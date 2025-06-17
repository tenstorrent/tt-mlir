// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TRANSFORMS_PASSES_TD
#define TTMLIR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConstEvalHoistTransform: Pass<"const-eval-hoist-transform", "::mlir::ModuleOp"> {
  let summary = "Transform to hoist const-eval subgraph into separate func(s)";
  let description = [{
    Transform pass which runs an analysis pass to find const-eval subgraphs in an input
    and hoist them into separate functions.
    This pass is dialect-agnostic, although ops with TT traits TTCoreCreationOpTrait and TTCoreDuplicateConstEvalTrait have special logic.
    If this pass is run on IR which has already been const-eval'ed, it will first undo existing const-eval hoisting and then re-run.

    Example:

    func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
      %0 = tensor.empty() : tensor<32x32xbf16>
      %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %2 = tensor.empty() : tensor<32x32xbf16>
      %3 = "ttir.subtract"(%arg2, %arg3, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %4 = tensor.empty() : tensor<32x32xbf16>
      %5 = "ttir.add"(%1, %3, %4) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %5 : tensor<32x32xbf16>
    }

    becomes:

    func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
      %0 = ttcore.load_cached(@forward_const_eval_0, [%arg2, %arg3]) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %1 = tensor.empty() : tensor<32x32xbf16>
      %2 = "ttir.add"(%arg0, %arg1, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %3 = tensor.empty() : tensor<32x32xbf16>
      %4 = "ttir.add"(%2, %0, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %4 : tensor<32x32xbf16>
    }
    func.func @forward_const_eval_0(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {const_eval} {
      %0 = tensor.empty() : tensor<32x32xbf16>
      %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %1 : tensor<32x32xbf16>
    }

  }];

  let dependentDialects = ["::mlir::tt::TTCoreDialect"];
}

def UndoConstEvalTransform : Pass<"undo-const-eval", "mlir::ModuleOp"> {
  let summary = "Undo const-eval hoisting by inlining const-eval functions";
  let description = [{
    This pass inlines all functions marked with the "const_eval" attribute,
    effectively undoing the transformations performed by ConstEvalHoistTransform.
  }];

  let dependentDialects = ["::mlir::tt::TTCoreDialect"];
}

#endif
