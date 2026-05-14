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
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td"

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
      static CoreRangeAttr getPhysicalCoreRange(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> physicalGridShape);
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

    `num_threads_per_cluster` is the per-kernel compute-thread fan-out on
    Quasar (1..hardware ceiling). Propagated from D2M_ThreadAttr.
    Default 1 keeps single-hart (WH/BH) compute kernels unaffected; the
    custom printer elides the field when it equals 1 so existing IR
    round-trips unchanged.
  }];
  let parameters = (ins "SymbolRefAttr":$kernel_symbol,
                        "CoreRangeAttr":$core_range,
                        "KernelArgsAttr":$kernel_args,
                        "MathFidelity":$math_fidelity,
                        "bool":$fp32_dest_acc_en,
                        "bool":$dst_full_sync_en,
                        "bool":$math_approx_mode,
                        ArrayRefParameter<"UnpackToDestMode">:$unpack_to_dest_mode,
                        DefaultValuedParameter<"uint32_t", "1">:$num_threads_per_cluster);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static ComputeConfigAttr get(mlir::MLIRContext *context,
                                 mlir::SymbolRefAttr kernelSymbol,
                                 CoreRangeAttr coreRange,
                                 KernelArgsAttr kernelArgs,
                                 MathFidelity mathFid,
                                 bool fp32DestAccEn,
                                 bool dstFullSyncEn,
                                 ::llvm::ArrayRef<UnpackToDestMode> unpackToDestMode) {
      return ComputeConfigAttr::get(context,
                                    kernelSymbol,
                                    coreRange,
                                    kernelArgs,
                                    mathFid,
                                    fp32DestAccEn,
                                    dstFullSyncEn,
                                    false,
                                    unpackToDestMode,
                                    1u);
    };

    static ComputeConfigAttr get(mlir::MLIRContext *context,
                                 mlir::SymbolRefAttr kernelSymbol,
                                 CoreRangeAttr coreRange,
                                 KernelArgsAttr kernelArgs,
                                 MathFidelity mathFid,
                                 bool fp32DestAccEn,
                                 bool dstFullSyncEn,
                                 bool mathApproxMode,
                                 ::llvm::ArrayRef<UnpackToDestMode> unpackToDestMode) {
      return ComputeConfigAttr::get(context,
                                    kernelSymbol,
                                    coreRange,
                                    kernelArgs,
                                    mathFid,
                                    fp32DestAccEn,
                                    dstFullSyncEn,
                                    mathApproxMode,
                                    unpackToDestMode,
                                    1u);
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
                        "mlir::tt::ttcore::NocIndex":$noc_index);
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
                        "mlir::tt::ttcore::NocIndex":$noc_index);
  let assemblyFormat = "`<` $kernel_symbol `,` qualified($core_range) `,` qualified($kernel_args) `,` $eth_type`,` $noc_index `>`";
  let extraClassDeclaration = [{
    mlir::tt::ttkernel::ThreadType getThreadType() const { return mlir::tt::ttkernel::ThreadType::Ethernet; }
  }];
}

def TTMetal_KernelConfigAttr : AnyAttrOf<[TTMetal_ComputeConfigAttr, TTMetal_NocConfigAttr, TTMetal_EthernetConfigAttr]>;

def TTMetal_KernelConfigArrayAttr : TypedArrayAttrBase<TTMetal_KernelConfigAttr, "">;

def TTMetal_DataflowBufferConfigAttr : TTMetal_Attr<"DataflowBufferConfig", "dfb_config"> {
  let summary = "TTMetal Dataflow Buffer config attribute.";
  let description = [{
    Configuration for a Quasar Dataflow Buffer (DFB), mirroring the
    DataflowBufferConfig struct in tt-metal's experimental DFB API
    (`experimental::dfb::CreateDataflowBuffer`).

    - entry_size: size of one DFB entry in bytes.
    - num_entries: number of entries in the buffer.
    - num_producers, num_consumers: cardinality.
    - producer_risc_mask, consumer_risc_mask: bitmask of participating
      RISCs (bits 0-7: DM RISCs, bits 8-15: Tensix compute RISCs).
    - producer_pattern, consumer_pattern: STRIDED vs BLOCKED access pattern.
    - enable_implicit_sync: when true, use transaction-id-based sync
      instead of explicit reserve/push/wait/pop (currently always false in
      the compiler — implicit sync is out of scope for the MVP).
    - data_format: element data format.
  }];

  let parameters = (ins "uint32_t":$entry_size,
                        "uint32_t":$num_entries,
                        "uint32_t":$num_producers,
                        "uint32_t":$num_consumers,
                        "uint32_t":$producer_risc_mask,
                        "uint32_t":$consumer_risc_mask,
                        "mlir::tt::ttcore::DFBAccessPatternAttr":$producer_pattern,
                        "mlir::tt::ttcore::DFBAccessPatternAttr":$consumer_pattern,
                        "bool":$enable_implicit_sync,
                        "mlir::tt::ttcore::DataTypeAttr":$data_format);

  let assemblyFormat = "`<` `entry_size` `=` $entry_size `,` `num_entries` `=` $num_entries `,` `num_producers` `=` $num_producers `,` `num_consumers` `=` $num_consumers `,` `producer_risc_mask` `=` $producer_risc_mask `,` `consumer_risc_mask` `=` $consumer_risc_mask `,` `producer_pattern` `=` $producer_pattern `,` `consumer_pattern` `=` $consumer_pattern `,` `enable_implicit_sync` `=` $enable_implicit_sync `,` `data_format` `=` $data_format `>`";
}

#endif
