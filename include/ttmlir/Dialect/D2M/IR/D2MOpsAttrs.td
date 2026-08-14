// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD
#define TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsEnums.td"
include "ttmlir/Dialect/D2M/IR/D2MBase.td"

// Enum-backed attribute wrappers
def D2M_ReduceDimAttr : EnumAttr<D2M_Dialect, D2M_ReduceDim, "reduce_dim">;

def D2M_TileBcastTypeAttr : EnumAttr<D2M_Dialect, D2M_TileBcastType, "tile_bcast_type">;

def D2M_ResolutionStageAttr : EnumAttr<D2M_Dialect, D2M_ResolutionStage, "resolution_stage">;

def D2M_ThreadAttr : AttrDef<D2M_Dialect, "Thread", [], "::mlir::Attribute"> {
  let mnemonic = "thread";
  let summary = "Thread information for a generic op.";
  let description = [{
    Holds the thread information corresponding to a single generic op region.
    For data movement threads, `dmCoreIndex` indicates the assigned hardware DM
    core.

    The default DM core -> NoC mapping is defined in getDmCoreDefaultNoc:
    on Wormhole/Blackhole DM core 1 maps to NoC0 and DM core 0 maps to NoC1;
    on Quasar every DM core maps to NoC0.

    A value of -1 means unassigned.
  }];
  let parameters = (ins
    "ThreadType":$threadType,
    OptionalParameter<"SymbolRefAttr">:$kernelSymbol,
    DefaultValuedParameter<"int32_t", "-1">:$dmCoreIndex
  );

  // Format: `<` threadType (`,` kernelSymbol)?
  //             (`,` `dm_core` `=` dmCoreIndex)? `>`.
  // The optional groups all start with `,`, which the declarative format cannot
  // disambiguate when kernelSymbol is absent, so we hand-roll the parser to
  // peek for the keyword after a comma and pick the right branch.
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
      static ThreadAttr get(::mlir::MLIRContext *context, ThreadType threadType) {
        return get(context, threadType, nullptr, -1);
      }
      static ThreadAttr get(::mlir::MLIRContext *context, ThreadType threadType,
                            SymbolRefAttr kernelSymbol) {
        return get(context, threadType, kernelSymbol, -1);
      }
  }];
}

def D2M_ThreadArrayAttr : TypedArrayAttrBase<D2M_ThreadAttr, "">;

def D2M_TopKPlacementAttr : AttrDef<D2M_Dialect, "TopKPlacement", [], "::mlir::Attribute"> {
  let mnemonic = "topk_placement";
  let summary = "One buffer of a multi-core topk, already placed on cores.";
  let description = [{
    A single buffer of a planned topk, with its grid already folded onto
    physical cores by `d2m-grid-selection`.

    `type` is the finished device type, carried verbatim because its dim
    alignments come from a grid-dependent search that a logical shape plus a
    grid cannot reproduce.

    `grid` is the consuming `d2m.generic`'s grid. `vgmForward`/`vgmInverse` are
    the wider `d2m.empty` virtual grid mapping over grid and shard dims, absent
    when the grid needs neither virtualization nor a spatial-region offset.
  }];
  let parameters = (ins
    "RankedTensorType":$type,
    "::mlir::tt::ttcore::GridAttr":$grid,
    OptionalParameter<"::mlir::AffineMapAttr">:$vgmForward,
    OptionalParameter<"::mlir::AffineMapAttr">:$vgmInverse
  );

  let assemblyFormat = [{ `<` struct(params) `>` }];
}

def D2M_TopKPlanAttr : AttrDef<D2M_Dialect, "TopKPlan", [], "::mlir::Attribute"> {
  let mnemonic = "topk_plan";
  let summary = "How a topk is split across the worker grid, and where every buffer lands.";
  let description = [{
    Written by `d2m-grid-selection` onto the placeholder leaf `d2m.generic`, and
    consumed by `d2m-build-topk-chain`, which erases that leaf and re-emits the
    whole chain from here. It is the whole handoff, so the chain build derives
    no shape and folds nothing, single core or not. The laid-out input grid
    selection emits itself stays out of it, since only what is left to build
    travels here.

    `numShards` bands the reduction dim, `ntShards` slices the non-target dim.
    `mergeSchedule` holds the group count surviving *after* each merge round,
    outermost first, so its last element is always 1.

    `placements` is consumed in order, so it must stay in the order
    `planTopKBuffers` builds it.
  }];
  let parameters = (ins
    "int64_t":$numShards,
    "int64_t":$ntShards,
    ArrayRefParameter<"int64_t">:$mergeSchedule,
    ArrayRefParameter<"TopKPlacementAttr">:$placements
  );

  let assemblyFormat = [{ `<` struct(params) `>` }];
}

def D2M_SenderReceiversAttr : AttrDef<D2M_Dialect, "SenderReceivers"> {
  let mnemonic = "sender_receivers";
  let summary = "One sender core and its receiver CoreRange set.";
  let description = [{
    One entry of a tt-metal GlobalCircularBuffer sender_receiver_core_mapping:
    a sender CoreCoord and the CoreRanges of cores that receive from it.
  }];
  let parameters = (ins
    "::mlir::tt::ttcore::CoreCoordAttr":$sender,
    ArrayRefParameter<"::mlir::tt::ttcore::CoreRangeAttr">:$receivers
  );
  let assemblyFormat =
      "`<` custom<CoordBracketStyle>($sender) `,` `[` $receivers `]` `>`";
  let genVerifyDecl = 1;
}

def D2M_GlobalCBMappingAttr : AttrDef<D2M_Dialect, "GlobalCBMapping", [],
                                      "::mlir::Attribute"> {
  let mnemonic = "global_cb_mapping";
  let summary = "Compact sender-to-receiver mapping for a global circular buffer.";
  let description = [{
    Encodes tt-metal `vector<pair<CoreCoord, CoreRangeSet>>` without listing
    every pair when the pattern is regular.

    zip: equal-count CoreRanges, 1:1 in row-major (y, x) order.
    row_fanout: sender width is 1; sender (y, xs) maps to receiver row y.
    explicit: irregular list of #d2m.sender_receivers.

    Pair lists are expanded in the verifier and later lowering, not printed
    for zip/row_fanout.
  }];
  let parameters = (ins
    "GlobalCBMappingKind":$kind,
    OptionalParameter<"::mlir::tt::ttcore::CoreRangeAttr">:$sender,
    OptionalParameter<"::mlir::tt::ttcore::CoreRangeAttr">:$receiver,
    OptionalArrayRefParameter<"SenderReceiversAttr">:$pairs
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Expand zip/row_fanout/explicit into per-sender pairs.
    ::mlir::LogicalResult expand(
        ::llvm::SmallVectorImpl<SenderReceiversAttr> &expanded) const;
  }];
}

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD
