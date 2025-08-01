// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPU_IR_SFPUBASE_TD
#define TTMLIR_DIALECT_SFPU_IR_SFPUBASE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SFPU dialect definition.
//===----------------------------------------------------------------------===//

def SFPU_Dialect : Dialect {
    let name = "sfpu";
    let summary = "SFPU (Special Function Processing Unit) dialect for Tenstorrent accelerators";
    let description = [{
        This dialect provides a wrapper around Tenstorrent SFPU-specific functionality,
        enabling direct access to SFPU hardware capabilities through MLIR operations.
        
        The SFPU dialect maps closely to the SFPI (SFPU Programming Interface) and
        provides vector-based operations on 64-element floating-point vectors.
        
        Operations in this dialect correspond directly to SFPU instructions as defined
        in the SFPI GCC backend (rvtt-insn.h), providing a 1:1 mapping between
        MLIR operations and hardware instructions.
    }];
    let cppNamespace = "::mlir::tt::sfpu";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::vector::VectorDialect", 
      "::mlir::tt::ttcore::TTCoreDialect"
    ];
}

//===----------------------------------------------------------------------===//
// Base SFPU operation definition.
//===----------------------------------------------------------------------===//

class SFPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<SFPU_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base SFPU trait definitions.
//===----------------------------------------------------------------------===//

class SFPU_Trait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::tt::sfpu::Trait";
}

// Trait for unary SFPU operations
def SFPU_UnaryOpTrait : SFPU_Trait<"SFPUUnaryOpTrait">;

// Trait for binary SFPU operations  
def SFPU_BinaryOpTrait : SFPU_Trait<"SFPUBinaryOpTrait">;

// Trait for ternary SFPU operations
def SFPU_TernaryOpTrait : SFPU_Trait<"SFPUTernaryOpTrait">;

// Trait for SFPU comparison operations
def SFPU_ComparisonOpTrait : SFPU_Trait<"SFPUComparisonOpTrait">;

// Trait for SFPU type conversion operations
def SFPU_ConversionOpTrait : SFPU_Trait<"SFPUConversionOpTrait">;

//===----------------------------------------------------------------------===//
// Base SFPU operation classes.
//===----------------------------------------------------------------------===//

class SFPU_UnaryOp<string mnemonic, list<Trait> traits = []> :
        SFPU_Op<mnemonic, [SFPU_UnaryOpTrait] # traits>;

class SFPU_BinaryOp<string mnemonic, list<Trait> traits = []> :
        SFPU_Op<mnemonic, [SFPU_BinaryOpTrait] # traits>;

class SFPU_TernaryOp<string mnemonic, list<Trait> traits = []> :
        SFPU_Op<mnemonic, [SFPU_TernaryOpTrait] # traits>;

class SFPU_ComparisonOp<string mnemonic, list<Trait> traits = []> :
        SFPU_Op<mnemonic, [SFPU_ComparisonOpTrait] # traits>;

class SFPU_ConversionOp<string mnemonic, list<Trait> traits = []> :
        SFPU_Op<mnemonic, [SFPU_ConversionOpTrait] # traits>;

#endif // TTMLIR_DIALECT_SFPU_IR_SFPUBASE_TD