// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPU_IR_SFPUOPSTYPES_TD
#define TTMLIR_DIALECT_SFPU_IR_SFPUOPSTYPES_TD

include "ttmlir/Dialect/SFPU/IR/SFPUBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// SFPU Vector Types
//===----------------------------------------------------------------------===//

// SFPU 64-element float32 vector type (maps to SFPI vFloat)
def SFPU_VFloat : Type<CPred<"::mlir::tt::sfpu::isSFPUVectorType($_self, \"f32\")">,
                      "SFPU 64-element float32 vector", "::mlir::VectorType">;

// SFPU 64-element int32 vector type (maps to SFPI vInt)                      
def SFPU_VInt : Type<CPred<"::mlir::tt::sfpu::isSFPUVectorType($_self, \"i32\")">,
                    "SFPU 64-element int32 vector", "::mlir::VectorType">;

// SFPU 64-element uint32 vector type (maps to SFPI vUInt)
def SFPU_VUInt : Type<CPred<"::mlir::tt::sfpu::isSFPUVectorType($_self, \"ui32\")">,
                     "SFPU 64-element uint32 vector", "::mlir::VectorType">;

// Any SFPU vector type
def SFPU_VectorType : AnyTypeOf<[SFPU_VFloat, SFPU_VInt, SFPU_VUInt]>;

//===----------------------------------------------------------------------===//
// SFPU Register Types
//===----------------------------------------------------------------------===//

// SFPU destination register type (maps to SFPI dst_reg)
def SFPU_DstReg : Type<CPred<"::mlir::tt::sfpu::isDstRegType($_self)">,
                      "SFPU destination register">;
                      
// SFPU local register type (maps to SFPI l_reg)
def SFPU_LReg : Type<CPred<"::mlir::tt::sfpu::isLRegType($_self)">,
                    "SFPU local register">;

//===----------------------------------------------------------------------===//
// SFPU Attributes
//===----------------------------------------------------------------------===//

// SFPU comparison predicate attribute
def SFPU_CmpPredicateAttr : I32EnumAttr<"CmpPredicate", "SFPU comparison predicate",
  [
    I32EnumAttrCase<"eq", 0, "eq">,   // Equal
    I32EnumAttrCase<"ne", 1, "ne">,   // Not equal  
    I32EnumAttrCase<"lt", 2, "lt">,   // Less than
    I32EnumAttrCase<"le", 3, "le">,   // Less than or equal
    I32EnumAttrCase<"gt", 4, "gt">,   // Greater than
    I32EnumAttrCase<"ge", 5, "ge">,   // Greater than or equal
    I32EnumAttrCase<"uo", 6, "uo">,   // Unordered
    I32EnumAttrCase<"o", 7, "o">      // Ordered
  ]> {
  let cppNamespace = "::mlir::tt::sfpu";
}

// SFPU lookup table attribute (for sfplut operation)
// Use the standard MLIR ArrayAttr for now
// def SFPU_LutTableAttr : ArrayAttr;  // This will be defined later if needed

// SFPU architecture variant attribute
def SFPU_ArchVariantAttr : I32EnumAttr<"ArchVariant", "SFPU architecture variant",
  [
    I32EnumAttrCase<"wormhole", 0, "wormhole">,
    I32EnumAttrCase<"blackhole", 1, "blackhole">
  ]> {
  let cppNamespace = "::mlir::tt::sfpu";
}

#endif // TTMLIR_DIALECT_SFPU_IR_SFPUOPSTYPES_TD