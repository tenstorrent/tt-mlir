// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPSENUMS_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTKernel_Noc0 : I32EnumAttrCase<"Noc0", 0, "noc0">;
def TTKernel_Noc1 : I32EnumAttrCase<"Noc1", 1, "noc1">;
def TTKernel_Tensix : I32EnumAttrCase<"Tensix", 2, "tensix">;
def TTKernel_Ethernet : I32EnumAttrCase<"Ethernet", 3, "ethernet">;

def TTKernel_ThreadType : I32EnumAttr<"ThreadType", "TTKernel ThreadTypes", [
  TTKernel_Noc0,
  TTKernel_Noc1,
  TTKernel_Tensix,
  TTKernel_Ethernet,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

#endif
