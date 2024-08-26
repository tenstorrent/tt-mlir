// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTNN type definitions
//===----------------------------------------------------------------------===//

class TTNN_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTNN_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTNN_MemoryConfig : TTNN_Type<"MemoryConfig", "memory_config"> {
  let summary = "TTNN memory configuration attribute";
  let description = [{
    TTNN memory configuration attribute
  }];

  let parameters = (ins EnumParameter<TTNN_TensorMemoryLayout>:$tensorMemoryLayout,
                        EnumParameter<TTNN_BufferType>:$bufferType);
                        // TODO(bug #620): ShardSpec
  let assemblyFormat = "`<` $tensorMemoryLayout `,` $bufferType `>`";
}


#endif  // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSTYPES_TD
