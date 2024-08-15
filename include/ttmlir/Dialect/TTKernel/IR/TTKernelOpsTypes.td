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
    let parameters = (ins "uint64_t":$address,
                          "uint64_t":$port,
                          "MemRefType":$memref,
                          "uint64_t":$page_size,
                          "uint64_t":$num_buffers);
    let assemblyFormat = "`<` $address`,` $port`,` $memref`,` $page_size`,` $num_buffers `>`";

    let extraClassDeclaration = [{
        static CBType get(::mlir::MLIRContext *context,
                              uint64_t address,
                              uint64_t port,
                              MemRefType memref) {
          uint64_t numBuffers = 1;
          uint64_t pageSize = 0;
          if (::mlir::isa<::mlir::tt::TileType>(memref.getElementType())) {
            pageSize = ::mlir::cast<::mlir::tt::TileType>(memref.getElementType()).getSizeBytes();
          } else {
            pageSize = memref.getShape().back() * (memref.getElementType().getIntOrFloatBitWidth() / 8);
          }
          return CBType::get(context, address, port, memref, pageSize, numBuffers);
        }

        ::llvm::ArrayRef<int64_t> getShape() const {
          return getMemref().getShape();
        }
    }];
}

def TTKernel_NocAddr : TTKernel_Type<"NocAddr", "noc_addr"> {
    let summary = "TTKernel noc address";
    let description = "Noc address type in TTKernel dialect";
}

def TTKernel_ThreadTypeAttr : EnumAttr<TTKernel_Dialect, TTKernel_ThreadType, "thread"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTKernel_ThreadTypeArrayAttr : TypedArrayAttrBase<TTKernel_ThreadTypeAttr, "">;

#endif
