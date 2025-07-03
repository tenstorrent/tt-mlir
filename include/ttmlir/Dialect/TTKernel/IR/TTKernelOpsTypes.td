// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPSTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPSTYPES_TD

include "ttmlir/Dialect/TTKernel/IR/TTKernelBase.td"
include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTKernel type definitions
//===----------------------------------------------------------------------===//

class TTKernel_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTKernel_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTKernel_CB : TTKernel_Type<"CB", "cb"> {
    let summary = "TTKernel cb";
    let description = "Circular buffer type in TTKernel dialect";
    let parameters = (ins "MemRefType":$memref);
    let assemblyFormat = "`<` $memref `>`";

    let extraClassDeclaration = [{
        ::llvm::ArrayRef<int64_t> getShape() const {
          return getMemref().getShape();
        }

        int64_t getNumElements() const {
          return getMemref().getNumElements();
        }

        int64_t getNumTiles() const {
          assert(mlir::isa<mlir::tt::ttcore::TileType>(getMemref().getElementType()) &&
                 "getNumTiles only supports tile element type");
          return getNumElements();
        }
    }];
}

def TTKernel_Semaphore : TTKernel_Type<"Semaphore", "semaphore"> {
  let summary = "TTKernel semaphore";
  let description = "Semaphore type in TTKernel dialect";
}

def TTKernel_NocAddr : TTKernel_Type<"NocAddr", "noc_addr"> {
    let summary = "TTKernel noc address";
    let description = "Noc address type in TTKernel dialect";
}

def TTKernel_ThreadTypeAttr : EnumAttr<TTKernel_Dialect, TTKernel_ThreadType, "thread"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTKernel_ThreadTypeArrayAttr : TypedArrayAttrBase<TTKernel_ThreadTypeAttr, "">;

def TTKernel_BcastTypeAttr : EnumAttr<TTKernel_Dialect, TTKernel_BcastType, "bcast_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTKernel_BcastBinaryOpAttr : EnumAttr<TTKernel_Dialect, TTKernel_BcastBinaryOp, "bcast_binary_op"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTKernel_ReduceTypeAttr : EnumAttr<TTKernel_Dialect, TTKernel_ReduceType, "reduce_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTKernel_ReduceDimAttr : EnumAttr<TTKernel_Dialect, TTKernel_ReduceDim, "reduce_dim"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTKernel_L1Addr : TTKernel_Type<"L1Addr", "l1_addr"> {
    let summary = "TTKernel l1 address";
    let description = "L1 address type in TTKernel dialect";
}

def TTKernel_L1AddrPtr : TTKernel_Type<"L1AddrPtr", "l1_addr_ptr"> {
    let summary = "TTKernel l1 address pointer";
    let description = "L1 pointer address type in TTKernel dialect";
}

def TTKernel_InterleavedAddrGenFast : TTKernel_Type<"InterleavedAddrGenFast", "interleaved_addr_gen_fast"> {
    let summary = "TTKernel InterleavedAddrGenFast type";
    let description = "InterleavedAddrGenFast type in TTKernel dialect";
}

def TTKernel_TensorAccessorArgs : TTKernel_Type<"TensorAccessorArgs", "TensorAccessorArgs"> {
  let summary = "TensorAccessorArgs type";
  let description = "TensorAccessor args type that stores compile + runtime information";
}

def TTKernel_TensorAccessor : TTKernel_Type<"TensorAccessor", "TensorAccessor"> {
  let summary = "TensorAccessor type";
  let description = "Accessor that encapsulates logic to access tensor information";
}

def TTKernel_TensorAccessorPageMapping : TTKernel_Type<"TensorAccessorPageMapping", "PageMapping"> {
  let summary = "TensorAccessor PageMapping struct";
  let description = "TensorAccessor struct that holds bank_id and bank_page_offset";
}

def TTKernel_DataFormat : TTKernel_Type<"DataFormat", "DataFormat"> {
    let summary = "TTKernel compute data format type";
    let description = "Data format type in TTKernel dialect";
}

class TTKernel_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TTKernel_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "ttkernel." # attrMnemonic;
}

def TTKernel_ArgAttr : TTKernel_Attr<"Arg", "arg"> {
  let summary = "Kernel argument.";
  let description = [{
  }];
  let parameters = (ins "ArgType":$arg_type, "size_t":$operand_index,
                        DefaultValuedParameter<"bool", "true">:$is_uniform);
  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    static ArgAttr get(::mlir::MLIRContext *context, ArgType argType, size_t index) {
      return ArgAttr::get(context, argType, index, true);
    }
  }];
}

def TTKernel_ArgSpecAttr : TTKernel_Attr<"ArgSpec", "arg_spec"> {
  let summary = "Kernel argument specification.";
  let description = [{
    A list of argument attibutes, of which form the argument specification for this kernel.
  }];
  let parameters = (ins OptionalArrayRefParameter<"ArgAttr">:$rt_args,
                        OptionalArrayRefParameter<"ArgAttr">:$ct_args);
  let assemblyFormat = "`<` (`rt_args` `=` `[` $rt_args^ `]`)? ` ` (`ct_args` `=` `[` $ct_args^ `]`)? `>`";

  let extraClassDeclaration = [{
    static ArgSpecAttr setArgSpec(func::FuncOp func, ArgSpecAttr argSpec);
    // Returns the appended argument index.
    static size_t appendCompileTimeArg(func::FuncOp func, ArgAttr arg);
    // Returns the appended argument index.
    static size_t appendRuntimeArg(func::FuncOp func, ArgAttr arg);
  }];
}

#endif
