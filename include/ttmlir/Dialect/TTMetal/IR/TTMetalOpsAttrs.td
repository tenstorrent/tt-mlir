// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSATTRS_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalBase.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsEnums.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalAttrInterfaces.td"

//===----------------------------------------------------------------------===//
// TTMetal attr definitions
//===----------------------------------------------------------------------===//

class TTMetal_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TTMetal_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "ttmetal." # attrMnemonic;
}

def TTMetal_CoreRangeAttr : TTMetal_Attr<"CoreRange", "core_range"> {
  let summary = "TTMetal grid attribute";
  let description = [{
    TTMetal grid attribute
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$offset,
                        ArrayRefParameter<"int64_t">:$size);
  let assemblyFormat = "`<` custom<DimensionList>($offset) `,` custom<DimensionList>($size) `>`";

  let extraClassDeclaration = [{
      static CoreRangeAttr get(::mlir::MLIRContext *context, ::mlir::tt::ttcore::GridAttr grid);
  }];
}

def TTMetal_KernelArgAttr : TTMetal_Attr<"KernelArg", "kernel_arg"> {
  let summary = "TTMetal kernel argument attribute.";
  let description = [{
    Holds a kernel argument binding info, comprised of:
      - type: the type of the argument to be bound.
      - operand_index: the index of the operand to the top level enqueue_program to query this argument from.
  }];

  let parameters = (ins "ttkernel::ArgType":$type, "size_t":$operand_index);
  let assemblyFormat = "`<` $type `[` $operand_index `]` `>`";
}

def TTMetal_KernelArgsAttr : TTMetal_Attr<"KernelArgs", "kernel_args"> {
  let summary = "TTMetal kernel arguments attribute.";
  let description = [{
    Holds two lists of kernel arguments, one for runtime and one for compile time.
  }];

  let parameters = (ins OptionalArrayRefParameter<"KernelArgAttr">:$rt_args,
                        OptionalArrayRefParameter<"KernelArgAttr">:$ct_args);
  let assemblyFormat = "`<` (`rt_args` `=` `[` $rt_args^ `]`)? ` ` (`ct_args` `=` `[` $ct_args^ `]`)? `>`";
}


def TTMetal_ComputeConfigAttr: TTMetal_Attr<"ComputeConfig", "compute_config", [TTMetal_KernelConfigInterface]> {
  let summary = "TTMetal ComputeConfig attribute.";
  let description = [{
    Structure for configuring compute kernels.
  }];
  let parameters = (ins "SymbolRefAttr":$kernel_symbol,
                        "CoreRangeAttr":$core_range,
                        "KernelArgsAttr":$kernel_args,
                        "MathFidelity":$math_fidelity,
                        "bool":$fp32_dest_acc_en,
                        "bool":$dst_full_sync_en,
                        "bool":$math_approx_mode,
                        ArrayRefParameter<"UnpackToDestMode">:$unpack_to_dest_mode);
  let assemblyFormat =  "`<` $kernel_symbol `,` qualified($core_range) `,` qualified($kernel_args) `,` $math_fidelity`,` $fp32_dest_acc_en`,` $dst_full_sync_en`,` $math_approx_mode`,` `[` $unpack_to_dest_mode `]` `>`";

  let extraClassDeclaration = [{
    static ComputeConfigAttr get(mlir::MLIRContext *context,
                                 mlir::SymbolRefAttr kernelSymbol,
                                 CoreRangeAttr coreRange,
                                 KernelArgsAttr kernelArgs,
                                 MathFidelity mathFid,
                                 bool fp32DestAccEn,
                                 ::llvm::ArrayRef<UnpackToDestMode> unpackToDestMode) {
      return ComputeConfigAttr::get(context,
                                    kernelSymbol,
                                    coreRange,
                                    kernelArgs,
                                    mathFid,
                                    fp32DestAccEn,
                                    false,
                                    false,
                                    unpackToDestMode);
    };

    mlir::tt::ttkernel::ThreadType getThreadType() const { return mlir::tt::ttkernel::ThreadType::Compute; }
  }];
}

def TTMetal_NocConfigAttr: TTMetal_Attr<"NocConfig", "noc_config", [TTMetal_KernelConfigInterface]> {
  let summary = "TTMetal NocConfig attribute.";
  let description = [{
    Structure for configuring noc kernels.
  }];
  let parameters = (ins "SymbolRefAttr":$kernel_symbol,
                        "CoreRangeAttr":$core_range,
                        "KernelArgsAttr":$kernel_args,
                        "NocIndex":$noc_index);
  let assemblyFormat = "`<` $kernel_symbol `,` qualified($core_range) `,` qualified($kernel_args) `,` $noc_index `>`";
  let extraClassDeclaration = [{
    mlir::tt::ttkernel::ThreadType getThreadType() const { return mlir::tt::ttkernel::ThreadType::Noc; }
  }];
}

def TTMetal_EthernetConfigAttr: TTMetal_Attr<"EthernetConfig", "ethernet_config", [TTMetal_KernelConfigInterface]> {
  let summary = "TTMetal EthernetConfig attribute.";
  let description = [{
    Structure for configuring ethernet kernels.
  }];
  let parameters = (ins "SymbolRefAttr":$kernel_symbol,
                        "CoreRangeAttr":$core_range,
                        "KernelArgsAttr":$kernel_args,
                        "EthType":$eth_type,
                        "NocIndex":$noc_index);
  let assemblyFormat = "`<` $kernel_symbol `,` qualified($core_range) `,` qualified($kernel_args) `,` $eth_type`,` $noc_index `>`";
  let extraClassDeclaration = [{
    mlir::tt::ttkernel::ThreadType getThreadType() const { return mlir::tt::ttkernel::ThreadType::Ethernet; }
  }];
}

def TTMetal_KernelConfigAttr : AnyAttrOf<[TTMetal_ComputeConfigAttr, TTMetal_NocConfigAttr, TTMetal_EthernetConfigAttr]>;

def TTMetal_KernelConfigArrayAttr : TypedArrayAttrBase<TTMetal_KernelConfigAttr, "">;

#endif
