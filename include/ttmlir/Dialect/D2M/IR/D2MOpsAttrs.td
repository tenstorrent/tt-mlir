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

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD
