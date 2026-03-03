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

def D2M_ThreadAttr : AttrDef<D2M_Dialect, "Thread", [], "::mlir::Attribute"> {
  let mnemonic = "thread";
  let summary = "Thread information for a generic op.";
  let description = [{
    Holds the thread information corresponding to a single generic op region.
    For datamovement threads, nocIndex indicates the assigned NoC (0 or 1).
    A value of -1 means unassigned.
  }];
  let parameters = (ins
    "ThreadType":$threadType,
    OptionalParameter<"SymbolRefAttr">:$kernelSymbol,
    DefaultValuedParameter<"int32_t", "-1">:$nocIndex
  );

  let assemblyFormat = [{ `<` $threadType (`,` $kernelSymbol^)? (`,` `noc` `=` $nocIndex^)? `>` }];

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
