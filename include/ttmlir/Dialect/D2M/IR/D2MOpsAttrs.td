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

def D2M_ThreadAttr : AttrDef<D2M_Dialect, "Thread", [], "::mlir::Attribute"> {
  let mnemonic = "thread";
  let summary = "Thread information for a generic op.";
  let description = [{
    Holds the thread information corresponding to a single generic op region.
  }];
  let parameters = (ins
    "ThreadType":$threadType,
    OptionalParameter<"SymbolRefAttr">:$kernelSymbol
  );

  let assemblyFormat = [{ `<` $threadType (`,` $kernelSymbol^)? `>` }];

  let extraClassDeclaration = [{
      static ThreadAttr get(::mlir::MLIRContext *context, ThreadType threadType) {
        return get(context, threadType, nullptr);
      }
  }];
}

def D2M_ThreadArrayAttr : TypedArrayAttrBase<D2M_ThreadAttr, "">;

def D2M_KernelArgsAttr : AttrDef<D2M_Dialect, "KernelArgs", [], "::mlir::Attribute"> {
  let mnemonic = "kernel_args";
  let summary = "Kernel arguments.";
  let description = [{
    A list of argument attibutes, of which form the argument specification for this kernel.
  }];
  let parameters = (ins OptionalArrayRefParameter<"uint32_t">:$ct_args,
                        OptionalArrayRefParameter<"uint32_t">:$rt_args);
  let assemblyFormat = "`<` (`ct_args` `=` `[` $ct_args^ `]`)? ` ` (`rt_args` `=` `[` $rt_args^ `]`)? `>`";
}

def D2M_ProgramArgsAttr : AttrDef<D2M_Dialect, "ProgramArgs", [], "::mlir::Attribute"> {
  let mnemonic = "program_args";
  let summary = "program arguments.";
  let description = [{
    A list of argument attibutes, of which form the argument specification for this kernel.
  }];
  let parameters = (ins OptionalArrayRefParameter<"KernelArgsAttr">:$program_args);
  let assemblyFormat = "`<` (`[` $program_args^ `]`)? `>`";
}

def D2M_PerDeviceProgramArgsAttr : AttrDef<D2M_Dialect, "PerDeviceProgramArgs", [], "::mlir::Attribute"> {
  let mnemonic = "per_device_program_args";
  let summary = "Per-device program arguments.";
  let description = [{
    A list of argument attibutes, of which form the argument specification for this kernel.
  }];
  let parameters = (ins OptionalArrayRefParameter<"ProgramArgsAttr">:$per_device_program_args);
  let assemblyFormat = "`<` (`[` $per_device_program_args^ `]`)? `>`";
}

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSATTRS_TD
