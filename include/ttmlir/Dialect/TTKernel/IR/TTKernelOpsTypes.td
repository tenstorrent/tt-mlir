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
    let parameters = (ins "int64_t":$num_elements, "Type":$element_type);
    let assemblyFormat = "`<` $num_elements `,` $element_type `>`";

    let builders = [
      TypeBuilderWithInferredContext<(ins "mlir::MemRefType":$memref), [{
        return $_get(memref.getContext(), memref.getNumElements(), memref.getElementType());
      }]>,
    ];

    let extraClassDeclaration = [{
        int64_t getNumTiles() const {
          assert(mlir::isa<mlir::tt::ttcore::TileType>(getElementType()) &&
                 "getNumTiles only supports tile element type");
          return getNumElements();
        }
    }];
}

def TTKernel_Semaphore : TTKernel_Type<"Semaphore", "semaphore"> {
  let summary = "TTKernel semaphore";
  let description = "Semaphore type in TTKernel dialect";
}

def TTKernel_GlobalSemaphore : TTKernel_Type<"GlobalSemaphore", "global_semaphore"> {
  let summary = "TTKernel global semaphore";
  let description = "Global semaphore type in TTKernel dialect";
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

def TTKernel_BinaryDestReuseTypeAttr : EnumAttr<TTKernel_Dialect, TTKernel_BinaryDestReuseType, "reuse_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTKernel_EltwiseBinaryTypeAttr : EnumAttr<TTKernel_Dialect, TTKernel_EltwiseBinaryType, "eltwise_binary_type"> {
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

def TTKernel_FabricConnectionManager : TTKernel_Type<"FabricConnectionManager", "fabric_connection_manager"> {
    let summary = "TTKernel fabric connection manager";
    let description = "Fabric connection manager type in TTKernel dialect";
}

def TTKernel_MeshPosition : TTKernel_Type<"MeshPosition", "mesh_position"> {
    let summary = "TTKernel logical mesh position";
    let description = [{
      A logical mesh position representing a coordinate in the device mesh.
    }];
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
                        DefaultValuedParameter<"bool", "true">:$is_uniform,
                        DefaultValuedParameter<"StringAttr", "StringAttr::get($_ctxt, \"\")">:$argument_name);
  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    static ArgAttr get(::mlir::MLIRContext *context, ArgType argType, size_t index) {
      return ArgAttr::get(context, argType, index, true);
    }

    static ArgAttr get(::mlir::MLIRContext *context, ArgType argType, size_t index, bool isUniform) {
      return ArgAttr::get(context, argType, index, isUniform, StringAttr::get(context, ""));
    }

    static ArgAttr get(::mlir::MLIRContext *context, StringAttr argumentName) {
      return ArgAttr::get(context, ArgType::NamedArgument, 0, true, argumentName);
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
