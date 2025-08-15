// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPI_IR_SFPIBASE_TD
#define TTMLIR_DIALECT_SFPI_IR_SFPIBASE_TD

include "mlir/IR/OpBase.td"
include "ttmlir/Dialect/SFPI/IR/SFPIEnums.td"

//===----------------------------------------------------------------------===//
// SFPI dialect definition.
//===----------------------------------------------------------------------===//

def SFPI_Dialect : Dialect {
    let name = "sfpi";
    let summary = "SFPI (SFPU Programming Interface) dialect for Tenstorrent accelerators";
    let description = [{
        This dialect provides a wrapper around Tenstorrent SFPI functionality,
        enabling direct access to SFPU hardware capabilities through MLIR operations.

        The SFPI dialect provides a programming interface for the SFPU and
        provides vector-based operations on 4x8 tile-based floating-point vectors.

        Operations in this dialect correspond directly to SFPU instructions as defined
        in the SFPI GCC backend (rvtt-insn.h), providing a 1:1 mapping between
        MLIR operations and hardware instructions.

        A lot of documentation for this dialect has been taken from [tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation).
        Please refer to tt-isa-documentation for additional and up to date docs.
    }];
    let cppNamespace = "::mlir::tt::sfpi";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::vector::VectorDialect"
    ];

    // Disable bytecode features to avoid compatibility issues
    let hasConstantMaterializer = 0;
}

//===----------------------------------------------------------------------===//
// Base SFPI operation definition.
//===----------------------------------------------------------------------===//

class SFPI_Op<string mnemonic, list<Trait> traits = []> :
        Op<SFPI_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base SFPI trait definitions.
//===----------------------------------------------------------------------===//

class SFPI_Trait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::tt::sfpi::Trait";
}

// Trait for unary SFPI operations
def SFPI_UnaryOpTrait : SFPI_Trait<"SFPIUnaryOpTrait">;

// Trait for binary SFPI operations
def SFPI_BinaryOpTrait : SFPI_Trait<"SFPIBinaryOpTrait">;

// Trait for ternary SFPI operations
def SFPI_TernaryOpTrait : SFPI_Trait<"SFPITernaryOpTrait">;

// Trait for SFPI comparison operations
def SFPI_ComparisonOpTrait : SFPI_Trait<"SFPIComparisonOpTrait">;

// Trait for SFPI type conversion operations
def SFPI_ConversionOpTrait : SFPI_Trait<"SFPIConversionOpTrait">;

//===----------------------------------------------------------------------===//
// Base SFPI operation classes.
//===----------------------------------------------------------------------===//

class SFPI_UnaryOp<string mnemonic, list<Trait> traits = []> :
        SFPI_Op<mnemonic, [SFPI_UnaryOpTrait] # traits>;

class SFPI_BinaryOp<string mnemonic, list<Trait> traits = []> :
        SFPI_Op<mnemonic, [SFPI_BinaryOpTrait] # traits>;

class SFPI_TernaryOp<string mnemonic, list<Trait> traits = []> :
        SFPI_Op<mnemonic, [SFPI_TernaryOpTrait] # traits>;

class SFPI_ComparisonOp<string mnemonic, list<Trait> traits = []> :
        SFPI_Op<mnemonic, [SFPI_ComparisonOpTrait] # traits>;

class SFPI_ConversionOp<string mnemonic, list<Trait> traits = []> :
        SFPI_Op<mnemonic, [SFPI_ConversionOpTrait] # traits>;

#endif // TTMLIR_DIALECT_SFPI_IR_SFPIBASE_TD
