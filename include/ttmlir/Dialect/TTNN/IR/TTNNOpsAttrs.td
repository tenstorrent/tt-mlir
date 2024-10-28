// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSATTRS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTNN attr definitions
//===----------------------------------------------------------------------===//

class TTNN_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TTNN_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "ttnn." # attrMnemonic;
}

def TTNN_CoreRangeAttr : TTNN_Attr<"CoreRange", "core_range"> {
  let summary = "TTNN grid attribute";
  let description = [{
    TTNN grid attribute
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$offset,
                        ArrayRefParameter<"int64_t">:$size);
  let assemblyFormat = "`<` custom<DimensionList>($offset) `,` custom<DimensionList>($size) `>`";

  let extraClassDeclaration = [{
      static CoreRangeAttr get(::mlir::MLIRContext *context, ::mlir::tt::GridAttr grid, SmallVector<int64_t> offset = {0, 0})
      {
        assert(grid.getShape().size() == 2 && "Grid shape must be 2D for now");
        return CoreRangeAttr::get(context, {0, 0}, grid.getShape());
      }
  }];
}

def TTNN_LayoutAttr : EnumAttr<TTNN_Dialect, TTNN_Layout, "layout"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNN_TensorMemoryLayoutAttr : EnumAttr<TTNN_Dialect, TTNN_TensorMemoryLayout, "tensor_memory_layout"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNN_BufferTypeAttr : EnumAttr<TTNN_Dialect, TTNN_BufferType, "buffer_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNN_ShapeAttr : TTNN_Attr<"Shape", "shape"> {
  let summary = "TTNN Shape attribute";
  let description = [{
    TTNN shape attribute
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` custom<DimensionList>($shape) `>`";
}

def TTNN_ShardSpecAttr : TTNN_Attr<"ShardSpec", "shard_spec"> {
  let summary = "TTNN ShardSpec attribute";
  let description = [{
    TTNN ShardSpec attribute
  }];

  // TODO (#620): Add other fields like core_ranges, shard orientation etc.
  let parameters = (ins AttrParameter<"ShapeAttr", "">:$shardShape);
  let assemblyFormat = "`<` params `>`";
}

def TTNN_MemoryConfigAttr : TTNN_Attr<"MemoryConfig", "memory_config"> {
  let summary = "TTNN MemoryConfig attribute";
  let description = [{
    TTNN memory config attribute
  }];

  let parameters = (ins AttrParameter<"TensorMemoryLayoutAttr", "">:$tensorMemoryLayout,
                        AttrParameter<"BufferTypeAttr", "">:$bufferType,
                        AttrParameter<"ShardSpecAttr", "">:$shardSpec);

  let assemblyFormat = "`<` params `>`";

  let extraClassDeclaration = [{
    ::llvm::ArrayRef<int64_t> getShardShapeArray() const
    {
      return this->getShardSpec().getShardShape().getShape();
    }
  }];
}

def TTNN_MeshShapeAttr : TTNN_Attr<"MeshShape", "mesh_shape"> {
  let summary = "TTNN Mesh Shape";
  let description = [{
    TTNN mesh shape
  }];

  let parameters = (ins "int64_t":$y, "int64_t":$x);
  let assemblyFormat = "custom<VargDimensionList>($y, $x)";
}

#endif  // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSATTRS_TD
