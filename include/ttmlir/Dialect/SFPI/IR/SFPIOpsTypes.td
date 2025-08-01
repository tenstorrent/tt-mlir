// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPI_IR_SFPIOPSTYPES_TD
#define TTMLIR_DIALECT_SFPI_IR_SFPIOPSTYPES_TD

include "ttmlir/Dialect/SFPI/IR/SFPIBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// SFPI Vector Types
//===----------------------------------------------------------------------===//

// SFPI 4x8 float32 vector type (maps to SFPI vFloat)
def SFPI_VFloat : Type<CPred<"::mlir::tt::sfpi::isSFPIVectorType($_self, \"f32\")">,
                      "SFPI 4x8 float32 vector", "::mlir::VectorType">;

// SFPI 4x8 int32 vector type (maps to SFPI vInt)                      
def SFPI_VInt : Type<CPred<"::mlir::tt::sfpi::isSFPIVectorType($_self, \"i32\")">,
                    "SFPI 4x8 int32 vector", "::mlir::VectorType">;

// SFPI 4x8 uint32 vector type (maps to SFPI vUInt)
def SFPI_VUInt : Type<CPred<"::mlir::tt::sfpi::isSFPIVectorType($_self, \"ui32\")">,
                     "SFPI 4x8 uint32 vector", "::mlir::VectorType">;

// Any SFPI vector type
def SFPI_VectorType : AnyTypeOf<[SFPI_VFloat, SFPI_VInt, SFPI_VUInt]>;

//===----------------------------------------------------------------------===//
// SFPI Register Types
//===----------------------------------------------------------------------===//

// SFPI destination register type (maps to SFPI dst_reg)
def SFPI_DstReg : Type<CPred<"::mlir::tt::sfpi::isDstRegType($_self)">,
                      "SFPI destination register">;
                      
// SFPI local register type (maps to SFPI l_reg)
def SFPI_LReg : Type<CPred<"::mlir::tt::sfpi::isLRegType($_self)">,
                    "SFPI local register">;

//===----------------------------------------------------------------------===//
// SFPI Attributes
//===----------------------------------------------------------------------===//

// SFPI comparison predicate attribute
def SFPI_CmpPredicateAttr : I32EnumAttr<"CmpPredicate", "SFPI comparison predicate",
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
  let cppNamespace = "::mlir::tt::sfpi";
}

// SFPI lookup table attribute (for sfplut operation)
// Use the standard MLIR ArrayAttr for now
// def SFPI_LutTableAttr : ArrayAttr;  // This will be defined later if needed

// SFPI architecture variant attribute
def SFPI_ArchVariantAttr : I32EnumAttr<"ArchVariant", "SFPI architecture variant",
  [
    I32EnumAttrCase<"wormhole", 0, "wormhole">,
    I32EnumAttrCase<"blackhole", 1, "blackhole">
  ]> {
  let cppNamespace = "::mlir::tt::sfpi";
}

#endif // TTMLIR_DIALECT_SFPI_IR_SFPIOPSTYPES_TD