// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD
#define TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
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
    For datamovement threads, processorIndex indicates the assigned hardware
    datamovement processor. ProcessorIndex 0 is a mapped to NoC1 and
    ProcessorIndex 1 is mapped to NoC0.

    Compute thread regions also carry
    numComputeThreads, the number of compute threads used in a Neo L1 region.
    
    A value of -1 means unassigned.
  }];
  let parameters = (ins
    "ThreadType":$threadType,
    OptionalParameter<"SymbolRefAttr">:$kernelSymbol,
    DefaultValuedParameter<"int32_t", "-1">:$processorIndex,
    DefaultValuedParameter<"int64_t", "1">:$numComputeThreads
  );

  // Format: `<` threadType (`,` kernelSymbol)?
  //             (`,` `processor` `=` processorIndex)? `>`.
  // The optional groups all start with `,`, which the declarative format cannot
  // disambiguate when kernelSymbol is absent, so we hand-roll the parser to
  // peek for the keyword after a comma and pick the right branch.
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
      static ThreadAttr get(::mlir::MLIRContext *context, ThreadType threadType) {
        return get(context, threadType, nullptr, -1, 1);
      }
      static ThreadAttr get(::mlir::MLIRContext *context, ThreadType threadType,
                            SymbolRefAttr kernelSymbol) {
        return get(context, threadType, kernelSymbol, -1, 1);
      }
      static ThreadAttr get(::mlir::MLIRContext *context, ThreadType threadType,
                            SymbolRefAttr kernelSymbol, int32_t nocIndex) {
        return get(context, threadType, kernelSymbol, nocIndex, 1);
      }
  }];
}

def D2M_ThreadArrayAttr : TypedArrayAttrBase<D2M_ThreadAttr, "">;

def D2M_ComputeThreadMappingAttr : AttrDef<D2M_Dialect, "ComputeThreadMapping",
    [DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>], "::mlir::Attribute"> {
  let mnemonic = "compute_thread";
  let summary = "scf.forall mapping attribute for Quasar Tensix threads.";
  let description = [{
    Marks an scf.forall as describing parallelism across N Tensix threads
    in one Neo. D2MMaterializeComputeThreadForall recognizes this attribute 
    and lowers the enclosing scf.forall to single-program SPMD form in D2M IR.
  }];
  let parameters = (ins "int64_t":$num);
  let assemblyFormat = "`<` `num` `=` $num `>`";
}

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD
