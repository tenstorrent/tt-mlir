// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_D2M_D2MGENERICREGIONOPS_TD
#define TTMLIR_TTMLIR_DIALECT_D2M_D2MGENERICREGIONOPS_TD

include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsTypes.td"
include "ttmlir/Dialect/D2M/IR/D2MBase.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.td"

include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Generic Region Op Traits and Classes
//===----------------------------------------------------------------------===//

def IsDeviceL1MemorySpace : CPred<"::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::DeviceL1">;

def IsDeviceDRAMMemorySpace : CPred<"::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::DeviceDRAM">;

def IsDeviceMemorySpace : Or<[IsDeviceL1MemorySpace, IsDeviceDRAMMemorySpace]>;

def IsRegisterDstMemorySpace : CPred<"::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::RegisterDst">;

def DeviceL1MemRef : Type<
  And<[IsDeviceL1MemorySpace, HasRankGreaterOrEqualPred<1>]>,
  "device l1 memoryspace memref type",
  "::mlir::MemRefType">;

def DeviceMemRef : Type<
  And<[IsDeviceMemorySpace, HasRankGreaterOrEqualPred<1>]>,
  "device memoryspace memref type",
  "::mlir::MemRefType">;

def RegisterDstMemRef : Type<
  And<[IsRegisterDstMemorySpace, HasRankGreaterOrEqualPred<1>]>,
  "register memoryspace memref type",
  "::mlir::MemRefType">;

def DeviceOrRegisterMemRef : AnyTypeOf<[DeviceMemRef, RegisterDstMemRef]>;

// Allow either device/register memrefs or ranked tensors for generic region block ops.
def DeviceOrRegisterMemRefOrAnyRankedTensor : AnyTypeOf<[
  DeviceOrRegisterMemRef,
  AnyRankedTensor
]>;

class D2M_GenericRegionOp<string mnemonic, list<Trait> traits = []> :
    D2M_Op<mnemonic, [D2M_GenericParent] # traits> {}

class D2M_GenericRegionComputeOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionOp<mnemonic, [D2M_GenericRegionComputeOpTrait, D2M_OperandLoadStoreRegisterOpInterface] # traits> {}

class D2M_GenericRegionComputeUnaryDstOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionComputeOp<mnemonic, traits> {

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {0};
    }
     bool getDstRegInPlace() { return true; }
  }];
}

class D2M_GenericRegionComputeBinaryDstOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionComputeOp<mnemonic, traits> {

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      // Both operands are tiles, load both from DST
      return {0, 1};
    }
  }];
}

// Binary ops that support tile-tile OR tile-scalar operations
class D2M_GenericRegionComputeBinaryWithScalarOptionDstOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionComputeOp<mnemonic, traits> {

  let extraClassDeclaration = [{
    bool supportsTileOrScalarRhs() { return true; }

    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      assert(this->getNumOperands() == 2 && "Binary operation must have 2 operands");
      auto rhsType = this->getOperand(1).getType();
      // If RHS is a scalar (not a tile), only load LHS from DST
      if (!mlir::isa<::mlir::tt::ttcore::TileType>(rhsType)) {
        return {0};
      }
      // Both operands are tiles, load both from DST
      return {0, 1};
    }
  }];
}

// FPU binary ops that support tile-tile OR tile-scalar operations
class D2M_GenericRegionComputeFPUOrSFPUBinary<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionComputeOp<mnemonic, traits> {

  let extraClassDeclaration = [{
    bool supportsTileOrScalarRhs() { return true; }
    bool isFPUBinaryOp() { return true; }

    // Returns the operand indices that need to be loaded from the dst register.
    // For FPU, we don't load from DST.
    // For SFPU (either via f32 operands or scalar operands) we load from DST.
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister();
  }];
}

class D2M_GenericRegionComputeTernaryDstOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionComputeOp<mnemonic, traits> {

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      // All three operands are tiles, load all from DST
      return {0, 1, 2};
    }
  }];
}

class D2M_GenericRegionDatamovementOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionOp<mnemonic, [D2M_GenericRegionDatamovementOpTrait] # traits> {}

//===----------------------------------------------------------------------===//
// D2M Generic Region Trait Bundles
//===----------------------------------------------------------------------===//

// Shared trait bundle for TTIR tile compute ops.
defvar D2M_TileComputeOpTraitBundle = [
  Pure
];

//===----------------------------------------------------------------------===//
// D2M Generic Region Math Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

// Type constraint that accepts either a Tile or a scalar value (float/int)
def TileOrScalar : AnyTypeOf<[TTCore_Tile, AnyFloat, AnyInteger]>;

def D2M_TileAddOp : D2M_GenericRegionComputeFPUOrSFPUBinary<"tile_add", [Pure]> {
    let summary = "D2M Tile Add Op";
    let description = [{
        The `tile_add` operation adds two tiles element-wise on the FPU (or SFPU if one of the operands is a float32),
        or adds a scalar value to each element in the input tile if rhs is a scalar (SFPU path).
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
    let hasFolder = 1;
}

def D2M_TileSubOp : D2M_GenericRegionComputeFPUOrSFPUBinary<"tile_sub", [Pure]> {
    let summary = "D2M Tile Sub Op";
    let description = [{
        The `tile_sub` operation subtracts two tiles element-wise on the FPU (or SFPU if one of the operands is a float32),
        or subtracts a scalar value from each element in the input tile if rhs is a scalar (SFPU path).
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
    let hasFolder = 1;
}

def D2M_TileMulOp : D2M_GenericRegionComputeFPUOrSFPUBinary<"tile_mul", [Pure]> {
    let summary = "D2M Tile Mul Op";
    let description = [{
        The `tile_mul` operation multiplies two tiles element-wise on the FPU (or SFPU if one of the operands is a float32),
        or multiplies each element in the input tile by a scalar value if rhs is a scalar (SFPU path).
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
    let hasFolder = 1;
}

def D2M_TileDivOp : D2M_GenericRegionComputeBinaryWithScalarOptionDstOp<"tile_div", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Div Op";
    let description = [{
        The `tile_div` operation divides two tiles element-wise,
        or divides each element in the input tile by a scalar value if rhs is a scalar.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
    let hasFolder = 1;
}

def D2M_TileRecipOp : D2M_GenericRegionComputeUnaryDstOp<"tile_recip", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Recip Op";
    let description = [{
        The `tile_recip` operation computes the reciprocal of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TilePowOp : D2M_GenericRegionComputeBinaryWithScalarOptionDstOp<"tile_pow", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Pow Op";
    let description = [{
        The `tile_pow` operation raises two tiles element-wise,
        or raises each element in the input tile to a scalar power if rhs is a scalar.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
    let hasFolder = 1;
}

def D2M_TileExpOp : D2M_GenericRegionComputeUnaryDstOp<"tile_exp", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Exp Op";
    let description = [{
        The `tile_exp` operation computes the exponential of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileExp2Op : D2M_GenericRegionComputeUnaryDstOp<"tile_exp2", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Exp2 Op";
    let description = [{
        The `tile_exp2` operation computes the base-2 exponential of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileExpm1Op : D2M_GenericRegionComputeUnaryDstOp<"tile_expm1", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Expm1 Op";
    let description = [{
        The `tile_expm1` operation computes exp(x)-1 of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLogOp : D2M_GenericRegionComputeUnaryDstOp<"tile_log", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Log Op";
    let description = [{
        The `tile_log` operation computes the natural logarithm of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLog1pOp : D2M_GenericRegionComputeUnaryDstOp<"tile_log1p", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Log1p Op";
    let description = [{
        The `tile_log1p` operation computes log(1+x) of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileNegativeOp : D2M_GenericRegionComputeUnaryDstOp<"tile_negative", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Negative Op";
    let description = [{
        The `tile_negative` operation computes the negative of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileCosOp : D2M_GenericRegionComputeUnaryDstOp<"tile_cos", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Cos Op";
    let description = [{
        The `tile_cos` operation computes the cosine function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileAcosOp : D2M_GenericRegionComputeUnaryDstOp<"tile_acos", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Acos Op";
    let description = [{
        The `tile_acos` operation computes the arccosine of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileTanOp : D2M_GenericRegionComputeUnaryDstOp<"tile_tan", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Tan Op";
    let description = [{
        The `tile_tan` operation computes the tangent function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileAtanOp : D2M_GenericRegionComputeUnaryDstOp<"tile_atan", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Atan Op";
    let description = [{
        The `tile_atan` operation computes the arctangent of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileTanhOp : D2M_GenericRegionComputeUnaryDstOp<"tile_tanh", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Tanh Op";
    let description = [{
        The `tile_tanh` operation computes the hyperbolic tangent function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSqrtOp : D2M_GenericRegionComputeUnaryDstOp<"tile_sqrt", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sqrt Op";
    let description = [{
        The `tile_sqrt` operation computes the sqrt function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSquareOp : D2M_GenericRegionComputeUnaryDstOp<"tile_square", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Square Op";
    let description = [{
        The `tile_square` operation computes the square function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileRsqrtOp : D2M_GenericRegionComputeUnaryDstOp<"tile_rsqrt", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Rsqrt Op";
    let description = [{
        The `tile_rsqrt` operation computes the reciprocal sqrt function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSinOp : D2M_GenericRegionComputeUnaryDstOp<"tile_sin", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sin Op";
    let description = [{
        The `tile_sin` operation computes the sine function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileAsinOp : D2M_GenericRegionComputeUnaryDstOp<"tile_asin", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Asin Op";
    let description = [{
        The `tile_asin` operation computes the arcsine of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSigmoidOp : D2M_GenericRegionComputeUnaryDstOp<"tile_sigmoid", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sigmoid Op";
    let description = [{
        The `tile_sigmoid` operation computes the sigmoid of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSoftsignOp : D2M_GenericRegionComputeUnaryDstOp<"tile_softsign", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Softsign Op";
    let description = [{
        The `tile_softsign` operation computes x/(1+|x|) for each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileHardsigmoidOp : D2M_GenericRegionComputeUnaryDstOp<"tile_hardsigmoid", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sigmoid Op";
    let description = [{
        The `tile_hardsigmoid` operation computes the hardsigmoid of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSiluOp : D2M_GenericRegionComputeUnaryDstOp<"tile_silu", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sliu Op";
    let description = [{
        The `tile_silu` operation computes the silu of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileReluOp : D2M_GenericRegionComputeUnaryDstOp<"tile_relu", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Relu Op";
    let description = [{
        The `tile_relu` operation computes the relu of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileGeluOp : D2M_GenericRegionComputeUnaryDstOp<"tile_gelu", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile GELU Op";
    let description = [{
        The `tile_gelu` operation computes the GELU (Gaussian Error Linear Unit) of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileErfOp : D2M_GenericRegionComputeUnaryDstOp<"tile_erf", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Erf Op";
    let description = [{
        The `tile_erf` operation computes the error function (erf) of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileErfcOp : D2M_GenericRegionComputeUnaryDstOp<"tile_erfc", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Erfc Op";
    let description = [{
        The `tile_erfc` operation computes the complementary error function (erfc) of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSignOp : D2M_GenericRegionComputeUnaryDstOp<"tile_sign", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sign Op";
    let description = [{
        The `tile_sign` operation computes the sign of each element in the input tile.
        Returns 1 for positive values, -1 for negative values, and 0 for zero.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSignbitOp : D2M_GenericRegionComputeUnaryDstOp<"tile_signbit", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Signbit Op";
    let description = [{
        The `tile_signbit` operation extracts the IEEE-754 sign bit of each element,
        producing 0.0 or 1.0 in the tile element type.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileCeilOp : D2M_GenericRegionComputeUnaryDstOp<"tile_ceil", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Ceil Op";
    let description = [{
        The `tile_ceil` operation computes the ceiling function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileFloorOp : D2M_GenericRegionComputeUnaryDstOp<"tile_floor", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Floor Op";
    let description = [{
        The `tile_floor` operation computes the floor function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileFracOp : D2M_GenericRegionComputeUnaryDstOp<"tile_frac", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Frac Op";
    let description = [{
        The `tile_frac` operation computes the fractional part of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileTruncOp : D2M_GenericRegionComputeUnaryDstOp<"tile_trunc", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Trunc Op";
    let description = [{
        The `tile_trunc` operation truncates each element of the input tile toward zero.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileAbsOp : D2M_GenericRegionComputeUnaryDstOp<"tile_abs", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Abs Op";
    let description = [{
        The `tile_abs` operation computes the absolute value of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileBitwiseNotOp : D2M_GenericRegionComputeUnaryDstOp<"tile_bitwise_not", D2M_TileComputeOpTraitBundle>{
    let summary = "D2M Tile Bitwise Not Op";
    let description = [{
        The `tile_bitwise_not` operation computes the bitwise negation of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileBitwiseAndOp : D2M_GenericRegionComputeBinaryWithScalarOptionDstOp<"tile_bitwise_and", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Bitwise And Op";
    let description = [{
        The `tile_bitwise_and` operation computes bitwise AND between two tiles.
    }];

    let arguments = (ins TileOrScalar:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileBitwiseOrOp : D2M_GenericRegionComputeBinaryWithScalarOptionDstOp<"tile_bitwise_or", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Bitwise Or Op";
    let description = [{
        The `tile_bitwise_or` operation computes bitwise OR between two tiles.
    }];

    let arguments = (ins TileOrScalar:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileBitwiseXorOp : D2M_GenericRegionComputeBinaryWithScalarOptionDstOp<"tile_bitwise_xor", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Bitwise Xor Op";
    let description = [{
        The `tile_bitwise_xor` operation computes bitwise XOR between two tiles.
    }];

    let arguments = (ins TileOrScalar:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLogicalNotOp : D2M_GenericRegionComputeUnaryDstOp<"tile_logical_not", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Logical Not Op";
    let description = [{
        The `tile_logical_not` operation computes the logical negation of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileEqzOp : D2M_GenericRegionComputeUnaryDstOp<"tile_eqz", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Eqz Op";
    let description = [{
        The `tile_eqz` operation checks if each element in the input tile == 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileNezOp : D2M_GenericRegionComputeUnaryDstOp<"tile_nez", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Nez Op";
    let description = [{
        The `tile_nez` operation checks if each element in the input tile != 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileGtzOp : D2M_GenericRegionComputeUnaryDstOp<"tile_gtz", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Gtz Op";
    let description = [{
        The `tile_gtz` operation checks if each element in the input tile > 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileGezOp : D2M_GenericRegionComputeUnaryDstOp<"tile_gez", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Gez Op";
    let description = [{
        The `tile_gez` operation checks if each element in the input tile >= 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLtzOp : D2M_GenericRegionComputeUnaryDstOp<"tile_ltz", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Ltz Op";
    let description = [{
        The `tile_ltz` operation checks if each element in the input tile < 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLezOp : D2M_GenericRegionComputeUnaryDstOp<"tile_lez", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Lez Op";
    let description = [{
        The `tile_lez` operation checks if each element in the input tile <= 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileEqOp : D2M_GenericRegionComputeBinaryDstOp<"tile_eq", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Eq Op";
    let description = [{
        The `tile_eq` operation performs element-wise equality comparison between two tiles.
        Result element is 1.0 where lhs == rhs, otherwise 0.0.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileNeOp : D2M_GenericRegionComputeBinaryDstOp<"tile_ne", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Ne Op";
    let description = [{
        The `tile_ne` operation performs element-wise inequality comparison between two tiles.
        Result element is 1.0 where lhs != rhs, otherwise 0.0.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileGtOp : D2M_GenericRegionComputeBinaryDstOp<"tile_gt", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Gt Op";
    let description = [{
        The `tile_gt` operation performs element-wise greater-than comparison between two tiles.
        Result element is 1.0 where lhs > rhs, otherwise 0.0.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLtOp : D2M_GenericRegionComputeBinaryDstOp<"tile_lt", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Lt Op";
    let description = [{
        The `tile_lt` operation performs element-wise less-than comparison between two tiles.
        Result element is 1.0 where lhs < rhs, otherwise 0.0.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileGeOp : D2M_GenericRegionComputeBinaryDstOp<"tile_ge", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Ge Op";
    let description = [{
        The `tile_ge` operation performs element-wise greater-than-or-equal comparison
        between two tiles. Result element is 1.0 where lhs >= rhs, otherwise 0.0.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLeOp : D2M_GenericRegionComputeBinaryDstOp<"tile_le", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Le Op";
    let description = [{
        The `tile_le` operation performs element-wise less-than-or-equal comparison
        between two tiles. Result element is 1.0 where lhs <= rhs, otherwise 0.0.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileMaximumOp : D2M_GenericRegionComputeBinaryDstOp<"tile_maximum", D2M_TileComputeOpTraitBundle>{
    let summary = "D2M Tile Maximum Op";
    let description = [{
        The `tile_maximum` operation calculates the maximum of two tensors element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileMinimumOp : D2M_GenericRegionComputeBinaryDstOp<"tile_minimum", D2M_TileComputeOpTraitBundle>{
    let summary = "D2M Tile Minimum Op";
    let description = [{
        The `tile_minimum` operation calculates the minimum of two tensors element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileClampScalarOp : D2M_GenericRegionComputeUnaryDstOp<"tile_clamp_scalar", D2M_TileComputeOpTraitBundle>{
    let summary = "D2M Tile Clamp Scalar Op";
    let description = [{
        The `tile_clamp_scalar` operation clamps all elements of a tile to be within
        the range [min, max] specified by scalar attributes.
    }];

    let arguments = (ins TTCore_Tile:$input, AnyAttrOf<[F32Attr, I32Attr]>:$min, AnyAttrOf<[F32Attr, I32Attr]>:$max);
    let results = (outs TTCore_Tile:$result);

    let hasVerifier = 1;
}

def D2M_TileSeluOp : D2M_GenericRegionComputeUnaryDstOp<"tile_selu", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile SELU Op";
    let description = [{
        The `tile_selu` operation applies SELU activation with `scale` and `alpha` given as
        f32 attributes (passed to the kernel as IEEE-754 bit patterns).
    }];

    let arguments = (ins TTCore_Tile:$input, F32Attr:$scale, F32Attr:$alpha);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileWhereOp : D2M_GenericRegionComputeTernaryDstOp<"tile_where", D2M_TileComputeOpTraitBundle>{
    let summary = "D2M Tile Where Op";
    let description = [{
        The `tile_where` operation performs element-wise conditional selection.
        For each element position, it selects between two values based on a boolean
        condition in the first tensor:
        - If the condition is true (non-zero), it selects the corresponding element
          from the second tensor (true_value)
        - If the condition is false (zero), it selects the corresponding element
          from the third tensor (false_value)
    }];

    let arguments = (ins TTCore_Tile:$condition, TTCore_Tile:$true_value, TTCore_Tile:$false_value);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileReduceSumOp : D2M_GenericRegionComputeOp<"tile_reduce_sum", D2M_TileComputeOpTraitBundle> {
  let summary = "D2M Tile Reduce Sum Op";
  let description = [{
        The `tile_reduce_sum` operation computes the sum of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- sum<dims>(A * B, C).
        Valid only for float element types; integer sum reductions use `tile_sfpu_reduce_sum` instead.
    }];

  let arguments = (ins TTCore_Tile:$a,
                       TTCore_Tile:$b,
                       TTCore_Tile:$c,
                       D2M_ReduceDimAttr:$reduce_dim);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {2};
    }
    bool getDstRegInPlace() {
      return true;
    }
  }];

  let hasVerifier = 1;
}

def D2M_TileReduceMaxOp : D2M_GenericRegionComputeOp<"tile_reduce_max", D2M_TileComputeOpTraitBundle> {
  let summary = "D2M Tile Reduce Max Op";
  let description = [{
        The `tile_reduce_max` operation computes the max of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- max<dims>(A * B, C).
        Valid only for float element types; integer max reductions use `tile_sfpu_reduce_max` instead.
    }];

  let arguments = (ins TTCore_Tile:$a,
                       TTCore_Tile:$b,
                       TTCore_Tile:$c,
                       D2M_ReduceDimAttr:$reduce_dim);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {2};
    }
    bool getDstRegInPlace() {
      return true;
    }
  }];

  let hasVerifier = 1;
}

def D2M_TileReduceMeanOp : D2M_GenericRegionComputeOp<"tile_reduce_mean", D2M_TileComputeOpTraitBundle> {
  let summary = "D2M Tile Reduce Mean Op";
  let description = [{
        The `tile_reduce_mean` operation computes the mean of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- avg<dims>(A * B, C).
        Valid only for float element types; integer mean reduction is not supported yet.
    }];

  let arguments = (ins TTCore_Tile:$a,
                       TTCore_Tile:$b,
                       TTCore_Tile:$c,
                       D2M_ReduceDimAttr:$reduce_dim);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {2};
    }
    bool getDstRegInPlace() {
      return true;
    }
  }];

  let hasVerifier = 1;
}

def D2M_TileSFPUReduceSumOp : D2M_GenericRegionComputeOp<"tile_sfpu_reduce_sum", D2M_TileComputeOpTraitBundle> {
  let summary = "D2M Tile SFPU Reduce Sum Op";
  let description = [{
        The `tile_sfpu_reduce_sum` operation computes the sum of the input A over the specified reduction dim(s), accumulating into C: result <- sum<dims>(A, C).
        Valid only for integer element types; float sum reductions use `tile_reduce_sum` instead.

        `dst_scratch_index` is populated by `d2m-insert-dst-register-access`
        (see `getNumDstScratchSlices`) and is -1 before the pass runs.
    }];

  let arguments = (ins TTCore_Tile:$a,
                       TTCore_Tile:$c,
                       D2M_ReduceDimAttr:$reduce_dim,
                       DefaultValuedAttr<I64Attr, "-1">:$dst_scratch_index);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {1};
    }
    bool getDstRegInPlace() {
      return true;
    }
    int64_t getNumDstScratchSlices() {
      return 1;
    }
  }];

  let hasVerifier = 1;
}

def D2M_TileSFPUReduceMaxOp : D2M_GenericRegionComputeOp<"tile_sfpu_reduce_max", D2M_TileComputeOpTraitBundle> {
  let summary = "D2M Tile SFPU Reduce Max Op";
  let description = [{
        The `tile_sfpu_reduce_max` operation computes the max of the input A over the specified reduction dim(s), accumulating into C: result <- max<dims>(A, C).
        Valid only for integer element types; float max reductions use `tile_reduce_max` instead.

        `dst_scratch_index` is populated by `d2m-insert-dst-register-access`
        (see `getNumDstScratchSlices`) and is -1 before the pass runs.
    }];

  let arguments = (ins TTCore_Tile:$a,
                       TTCore_Tile:$c,
                       D2M_ReduceDimAttr:$reduce_dim,
                       DefaultValuedAttr<I64Attr, "-1">:$dst_scratch_index);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {1};
    }
    bool getDstRegInPlace() {
      return true;
    }
    int64_t getNumDstScratchSlices() {
      return 1;
    }
  }];

  let hasVerifier = 1;
}

// TODO(#5968): fusion w/ implicit bcast causes multiple failure scenarios.
def D2M_TileBcastOp : D2M_GenericRegionComputeOp<"tile_bcast", [D2M_SkipOpEltwiseFusionTrait]> {
  let summary = "D2M Tile Broadcast Op";
  let description = [{
    The `tile_bcast` operation broadcasts a row/col/scalar tile into a full tile.
  }];

  let arguments = (ins TTCore_Tile:$input,
                       D2M_TileBcastTypeAttr:$bcast_type);
  let results = (outs TTCore_Tile:$result);
}

def D2M_TileMatmulOp : D2M_GenericRegionComputeOp<"tile_matmul"> {
  let summary = "D2M Tile Matmul Op";
  let description = [{
        The `tile_matmul` operation computes the matrix multiplication of A and B input tiles and element-wise adds C tile: result <- a @ b + c.
    }];

  let arguments = (ins TTCore_Tile:$a, TTCore_Tile:$b, TTCore_Tile:$c);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {2};
    }
    bool getDstRegInPlace() {
      return true;
    }
  }];
}

def D2M_TileMatmulBlockOp : D2M_GenericRegionComputeOp<"tile_matmul_block",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "D2M Tile Matmul Block Op";
    let description = [{
        The `tile_matmul_block` operation computes the matrix multiplication of two input blocks.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$a,
                         AnyRankedTensorOrMemRef:$b,
                         AnyRankedTensorOrMemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def D2M_TileTilizeBlockOp : D2M_GenericRegionComputeOp<"tile_tilize_block",
  [ DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  , D2M_SkipOpEltwiseFusionTrait
  ]> {
    let summary = "D2M Tile Tilize Block Op";
    let description = [{
        The `tile_tilize_block` operation tilizes the input row major memref block and outputs the memref containing the tilized data.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input,
                         AnyRankedTensorOrMemRef:$output);
    let results = (outs AnyRankedTensorOrMemRef:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def D2M_TileUntilizeBlockOp : D2M_GenericRegionComputeOp<"tile_untilize_block",
  [ DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  , D2M_SkipOpEltwiseFusionTrait
  ]> {
    let summary = "D2M Tile Untilize Block Op";
    let description = [{
        The `tile_untilize_block` operation untilizes the input tilized memref block and outputs the memref containing the row major data.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input,
                         AnyRankedTensorOrMemRef:$output);
    let results = (outs AnyRankedTensorOrMemRef:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def D2M_TileTypecastOp : D2M_GenericRegionComputeUnaryDstOp<"tile_typecast", [D2M_SkipOpEltwiseFusionTrait]> {
    let summary = "D2M Tile Typecast Op";
    let description = [{
        The `tile_typecast` operation casts the input tile to the desired dataformat.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

//===----------------------------------------------------------------------===//
// D2M Masking Ops
//===----------------------------------------------------------------------===//

def D2M_TileFillOp : D2M_GenericRegionComputeOp<"tile_fill"> {
    let summary = "D2M Fill Tile Op";
    let description = [{
        The `tile_fill` operation creates a tile filled with a constant scalar value.
        All elements in the resulting tile will have the same value as the input scalar.

        This operation abstracts away the complexity of creating a constant-filled tile,
        providing a simple interface that takes a scalar value and produces a tile suitable
        for element-wise operations.
    }];

    let arguments = (ins AnyTypeOf<[AnyFloat, AnyInteger]>:$value);
    let results = (outs TTCore_Tile:$result);

    let assemblyFormat = [{
        `(` $value `)` attr-dict `:` type($value) `->` type($result)
    }];

    let extraClassDeclaration = [{
        // This operation doesn't load from DST (it generates a new tile)
        mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
          return {};
        }
        bool getDstRegInPlace() { return false; }
        // Operand 0 is the fill scalar (not a tile). Other indices must be
        // false so passes that query isScalarOperand(1) for tile+scalar rhs do
        // not treat tile_fill as in-place on store.
        bool isScalarOperand(int64_t operandIdx) {
          return operandIdx == 0;
        }
    }];
}

def D2M_TileRandOp : D2M_GenericRegionComputeOp<"tile_rand"> {
    let summary = "D2M Rand Tile Op";
    let description = [{
        The `tile_rand` operation creates a tile filled with random float values
        drawn from a uniform distribution over the range [from, from + scale).

        Attributes:
          - `seed` (UI32): Seed passed to `rand_tile_init` to initialize the PRNG.
          - `from` (F32): Lower bound of the uniform distribution (inclusive).
          - `scale` (F32): Width of the uniform distribution (must be > 0).

        Lowers to `ttkernel.rand_tile_init(seed)` + `ttkernel.rand_tile(dst_idx,
        from_bits, scale_bits)` where the `from`/`scale` parameters are the
        bit-cast f32 values.
    }];

    let arguments = (ins DefaultValuedAttr<UI32Attr, "0">:$seed,
                         DefaultValuedAttr<F32Attr, "0.0">:$from,
                         DefaultValuedAttr<F32Attr, "1.0">:$scale);
    let results = (outs TTCore_Tile:$result);

    let assemblyFormat = [{
        `(` `)` attr-dict `:` type($result)
    }];

    let extraClassDeclaration = [{
        // This operation doesn't load from DST (it generates a new tile).
        mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
          return {};
        }
        bool getDstRegInPlace() { return false; }
    }];
}

def D2M_BlockMaskOp : D2M_GenericRegionComputeOp<"block_mask",
  [ DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  , D2M_SkipOpEltwiseFusionTrait
  , AttrSizedOperandSegments
  ]> {
    let summary = "Mask block elements outside logical bounds with fill value";
    let description = [{
        The `block_mask` operation masks elements in a block/memref that fall outside
        the logical tensor bounds, replacing them with a specified fill value.

        This op operates at the block level (memref of tiles) and handles both complete
        tile masking (tiles entirely outside bounds) and partial tile masking (tiles
        that straddle the boundary).

        The logical_rows and logical_cols are Index values specifying the logical
        dimensions for this core's shard. These can be computed dynamically based on
        the core's position in the grid, enabling multicore support.

        For partial tile masking, row_mask_cb and col_mask_cb contain scratch mask
        tiles written by a corresponding BlockMaskWriteOp in the DM region. These are
        single-tile CBs where each element is 1.0 if valid or 0.0 if OOB.

        This is a high-level op that gets decomposed in a later pass into multiple
        loop nests that handle interior tiles, edge tiles, and corner tiles separately.

        Example:
        ```mlir
        // With mask CBs for partial masking:
        d2m.block_mask %input, %output, %row_mask_cb, %col_mask_cb,
                       %logical_rows, %logical_cols, <zero>
            : (memref<2x3x!ttcore.tile<32x32, f32>>, memref<2x3x!ttcore.tile<32x32, f32>>,
               memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>)
        ```
    }];

    let arguments = (ins
        AnyRankedTensorOrMemRef:$input,
        AnyRankedTensorOrMemRef:$output,
        Optional<AnyRankedTensorOrMemRef>:$row_mask_cb,
        Optional<AnyRankedTensorOrMemRef>:$col_mask_cb,
        Index:$logical_rows,
        Index:$logical_cols,
        TTCore_OOBValAttr:$fill_value);

    let results = (outs AnyRankedTensorOrMemRef:$result);

    // Use default format with AttrSizedOperandSegments - the operand_segment_sizes
    // attribute is automatically added/parsed to track optional operand presence.

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
      bool hasMaskCBs() { return getRowMaskCb() != nullptr && getColMaskCb() != nullptr; }
    }];

    let hasVerifier = 1;
}

def D2M_DstReinterpretCastOp : D2M_GenericRegionOp<"dst_reinterpret_cast"> {
    let summary = "D2M Destination Register Reinterpret Cast Op";
    let description = [{
        The `dst_reinterpret_cast` operation reinterprets a tile value as a different
        tile type for the purpose of storing/loading from the destination register.

        This models the hardware behavior that destination register memory is untyped
        and can be accessed with different type interpretations. It is inserted during
        the InsertDstRegisterAccess pass to reconcile type mismatches between compute
        operations and destination register allocations, and is removed during lowering
        to TTKernel as a no-op.

        Example:
        ```mlir
        %0 = affine.load %dst : memref<1x1x4x!ttcore.tile<32x32, f32>, #dst>
        %1 = d2m.tile_typecast %0 : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
        %2 = d2m.dst_reinterpret_cast %1 : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
        affine.store %2, %dst : memref<1x1x4x!ttcore.tile<32x32, f32>, #dst>
        ```
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

//===----------------------------------------------------------------------===//
// D2M Destination Control Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_AcquireDstOp : D2M_GenericRegionComputeOp<"acquire_dst",
  [ MemoryEffects<[MemRead, MemWrite]>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
    let summary = "Acquire Destination Register op.";
    let description = [{
      This op returns a memref with memory space dest that models the destination register
      resource on tensorrent hardware. Example IR:

      ```mlir
      %dst = d2m.acquire_dst() : memref<2x4x!tt.tile<32x32, f32>, #tt.memory_space<dst>>
      ```
    }];

    let results = (outs RegisterDstMemRef:$result);

    let assemblyFormat = [{ `(` `)` attr-dict `:` type($result) }];
}

def D2M_SetL1AccumulateOp : D2M_GenericRegionOp<"set_l1_accumulate"> {
    let summary = "Enable L1 accumulation mode.";
    let description = [{
      Configures HW state such that all subsequent stores to L1 memory space
      will be accumulated to in place.
    }];

    let arguments = (ins I32:$enable);

    let assemblyFormat = [{ `(` $enable `)` attr-dict }];
}

def D2M_UnpackStallOnPackOp : D2M_GenericRegionOp<"unpack_stall_on_pack"> {
  let summary = "Stall UNPACK thread until previous PACK write is committed to L1.";
  let description = [{
    Inserted between consecutive linalg.generics in a fused region when
    the intermediate result is written to L1 by one generic and immediately
    read by the next, and the usual tile_regs_acquire semwait is not present
    (single-tile shard path).
  }];
  let assemblyFormat = [{ attr-dict }];
}

def D2M_TileTransposeOp : D2M_GenericRegionComputeOp<"tile_transpose", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Transpose Op";
    let description = [{
        The `tile_transpose` operation computes the transpose of the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Datamovement Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_DMAWriteOp : D2M_GenericRegionDatamovementOp<"dma_write",
  [ AttrSizedOperandSegments
  , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , D2M_DMAOpInterface
  ]> {

    let summary = "Lowered D2M DMA Write Op";
    let description = [{
      This op performs a DMA write operation from src memref to dst memref. This operation supports multicast; providing an empty multicast shape
      argument _implies a unicast write_.

      It supports two forms distinguished by numElems:
      Fully indexed form (numElems > 0): src and dst indices are fully
      resolved (all view and memspace-specific affine maps applied). Corresponds
      1:1 to a single concrete DMA write operation.

      ```mlir
      %tx = d2m.dma_write %src[%offset], %dst[%gridy, %gridx, %offset], <%size>
        : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x2x4x!ttcore.tile<32x32, f32>, #dram>) -> !d2m.mem_tx
      ```

      Shard-level form (numElems == 0): indices are grid-level only.
      Represents "write entire shard at grid position". The indexing expansion
      is deferred to D2MLowerDMAToFullyIndexedForm.

      ```mlir
      %tx = d2m.dma_write %src, %dst[%grid_i, %grid_j], <0>
        : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>,
           memref<2x2x2x4x!ttcore.tile<32x32, f32>, #dram>) -> !d2m.mem_tx
      ```

      Constraints:
      - src MUST be a LOCAL memref!
      - src and dst must have the same element type.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$src, Variadic<Index>:$srcIndices,
                         AnyRankedTensorOrMemRef:$dst, Variadic<Index>:$dstIndices,
                         I64Attr:$numElems,
                         Variadic<Index>:$mcastStartIndex, Variadic<Index>:$mcastShape,
                         Variadic<Index>:$startDevice, Variadic<Index>:$deviceMcastShape);
    let results = (outs D2M_MemTx:$result);

    let builders =
    [
      // Fully indexed unicast - multicast args are defaulted
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "size_t": $numElems),
      [{
        build($_builder, $_state, MemTxType::get($_builder.getContext(), DMAType::Write), src, srcIndices, dst, dstIndices, $_builder.getI64IntegerAttr(numElems), ValueRange(), ValueRange(), /*startDevice=*/ValueRange(), /*deviceMcastShape=*/ValueRange());
      }]>,

      // Fully indexed multicast - multicast args must be provided
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape, "size_t": $numElems),
      [{
        build($_builder, $_state, MemTxType::get($_builder.getContext(), DMAType::McastWrite), src, srcIndices, dst, dstIndices, $_builder.getI64IntegerAttr(numElems), mcastStartIndex, mcastShape, /*startDevice=*/ValueRange(), /*deviceMcastShape=*/ValueRange());
      }]>,

      // Fully indexed unicast, with destination device range
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "size_t": $numElems, "ValueRange": $startDevice, "ValueRange": $deviceMcastShape),
      [{
        build($_builder, $_state, MemTxType::get($_builder.getContext(), DMAType::Write), src, srcIndices, dst, dstIndices, $_builder.getI64IntegerAttr(numElems), ValueRange(), ValueRange(), startDevice, deviceMcastShape);
      }]>,

      // Shard-level unicast (no src indices, grid indices on dst, numElems = 0)
      OpBuilder<(ins "Value": $src, "Value": $dst, "ValueRange": $dstIndices),
      [{
        build($_builder, $_state, MemTxType::get($_builder.getContext(), DMAType::Write), src, ValueRange(), dst, dstIndices, $_builder.getI64IntegerAttr(0), ValueRange(), ValueRange(), /*startDevice=*/ValueRange(), /*deviceMcastShape=*/ValueRange());
      }]>,

      // Shard-level multicast (no indices on either, numElems = 0)
      OpBuilder<(ins "Value": $src, "Value": $dst, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape),
      [{
        build($_builder, $_state, MemTxType::get($_builder.getContext(), DMAType::McastWrite), src, ValueRange(), dst, ValueRange(), $_builder.getI64IntegerAttr(0), mcastStartIndex, mcastShape, /*startDevice=*/ValueRange(), /*deviceMcastShape=*/ValueRange());
      }]>,

      // Shard-level unicast (no src indices, grid indices on dst, numElems = 0), with destination device range
      OpBuilder<(ins "Value": $src, "Value": $dst, "ValueRange": $dstIndices, "ValueRange": $startDevice, "ValueRange": $deviceMcastShape),
      [{
        build($_builder, $_state, MemTxType::get($_builder.getContext(), DMAType::Write), src, ValueRange(), dst, dstIndices, $_builder.getI64IntegerAttr(0), ValueRange(), ValueRange(), startDevice, deviceMcastShape);
      }]>
    ];

    let assemblyFormat = [{ $src (`[` $srcIndices^ `]`)? `,` $dst (`[` $dstIndices^ `]`)? (`core` `[` $mcastStartIndex^ `]` `mcast` `[` $mcastShape `]`)? (`startDevice` `[` $startDevice^ `]`)? (`deviceMcastShape` `[` $deviceMcastShape^ `]`)? `,` `<` $numElems `>` attr-dict `:` `(` type($src) `,` type($dst) `)` `->` qualified(type($result))}];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      MemRefType getSrcMemRefType() { return cast<MemRefType>(getSrc().getType()); }
      MemRefType getDstMemRefType() { return cast<MemRefType>(getDst().getType()); }
      bool isShardLevel() { return getNumElems() == 0; }
      bool isFullyIndexed() { return getNumElems() > 0; }
      size_t getSizeBytes() {
        assert(isFullyIndexed() && "getSizeBytes() requires fully indexed form");
        auto elementSizeBytes =
          ttcore::getElementSizeBytes(getSrcMemRefType().getElementType());
        return getNumElems() * elementSizeBytes;
      }
      bool isDstLocal() { return !isDstRemote(); }
      bool isDstRemote() { return ttcore::hasDeviceLayout(getDst()); }
      bool isMcast() { return !getMcastShape().empty(); }
      ::mlir::tt::ttcore::MemorySpace getDstMemorySpace() {
        return mlir::cast<::mlir::tt::ttcore::MemorySpaceAttr>(getDstMemRefType().getMemorySpace()).getValue();
      }
    }];
}

def D2M_DMAReadOp : D2M_GenericRegionDatamovementOp<"dma_read",
  [ AttrSizedOperandSegments
  , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , D2M_DMAOpInterface
  ]> {

    let summary = "Lowered D2M DMA Read Op.";
    let description = [{
      This op performs a DMA read operation from src memref to dst memref.
      The dst memref must be local. The src memref can either be remote or local.

      It supports two forms distinguished by numElems:
      Fully indexed form (numElems > 0): src and dst indices are fully
      resolved (all view and memspace-specific affine maps applied). Corresponds
      1:1 to a single concrete DMA read operation.

      Remote-to-local fully indexed form:
      ```mlir
      %tx = d2m.dma_read %src[%gridy, %gridx, %offset], %dst[%offset], <8>
        : (memref<2x2x2x4x!ttcore.tile<32x32, f32>, #dram>,
           memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
      ```

      Local-to-local fully indexed form:
      ```mlir
      %tx = d2m.dma_read %src[%offset], %dst[%offset], <8>
        : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>,
           memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
      ```

      Shard-level form (numElems == 0): indices are grid-level only on the
      remote src operand, no indices on the local dst. Represents "read entire
      shard at grid position". The indexing expansion is deferred to
      D2MLowerDMAToFullyIndexedForm.

      ```mlir
      %tx = d2m.dma_read %src[%grid_i, %grid_j], %dst, <0>
        : (memref<2x2x2x4x!ttcore.tile<32x32, f32>, #dram>,
           memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
      ```

      This operation only supports unicast (multicast are write-only).

      Constraints:
      - dst MUST be a LOCAL memref!
      - src and dst must have the same element type.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$src, Variadic<Index>:$srcIndices,
                         AnyRankedTensorOrMemRef:$dst, Variadic<Index>:$dstIndices,
                         I64Attr:$numElems);
    let results = (outs D2M_MemTx:$result);

    let builders = [
      // Shard-level form (grid indices on src, no indices on dst, numElems = 0)
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst),
      [{
        build($_builder, $_state, MemTxType::get($_builder.getContext(), DMAType::Read), src, srcIndices, dst, ValueRange(), $_builder.getI64IntegerAttr(0));
      }]>,
      // Fully indexed form
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "IntegerAttr": $numElems),
      [{
        build($_builder, $_state, MemTxType::get($_builder.getContext(), DMAType::Read), src, srcIndices, dst, dstIndices, numElems);
      }]>,
    ];

    let assemblyFormat    = [{ $src `[` $srcIndices `]` `,` $dst (`[` $dstIndices^ `]`)? `,` `<` $numElems `>` attr-dict `:` `(` type($src) `,` type($dst) `)` `->` qualified(type($result))}];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      bool isSrcLocal() { return !isSrcRemote(); }
      bool isSrcRemote() { return ttcore::hasDeviceLayout(getSrc()); }
      MemRefType getSrcMemRefType() { return cast<MemRefType>(getSrc().getType()); }
      MemRefType getDstMemRefType() { return cast<MemRefType>(getDst().getType()); }
      bool isShardLevel() { return getNumElems() == 0; }
      bool isFullyIndexed() { return getNumElems() > 0; }
      size_t getSizeBytes() {
        assert(isFullyIndexed() && "getSizeBytes() requires fully indexed form");
        auto elementSizeBytes =
          ttcore::getElementSizeBytes(getSrcMemRefType().getElementType());
        return getNumElems() * elementSizeBytes;
      }
      ::mlir::tt::ttcore::MemorySpace getSrcMemorySpace() {
        return mlir::cast<::mlir::tt::ttcore::MemorySpaceAttr>(getSrcMemRefType().getMemorySpace()).getValue();
      }
    }];
}

def D2M_LocalCopyOp : D2M_GenericRegionDatamovementOp<"local_copy",
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]>
  , DestinationStyleOpInterface
  , D2M_ShardDMAOpInterface
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType",
      "hasTensorSemantics"
    ]>
  ]> {
    let summary = "Local DMA copy with affine map indexing";
    let description = [{
      Copies data between two local L1 buffers using affine maps to describe
      the source/destination indexing relationship.  The iteration domain is
      the destination shape; `LowerDMAToFullyIndexedForm` generates the
      actual loops and point-access `d2m.dma_read` ops from the maps.

      Both operands must be local (non-DRAM).

      The operation has three surface forms:

      Tensor form (pre-bufferization):
      ```mlir
      %result = d2m.local_copy %src, %dst indexing_maps = [#src_map, #dst_map]
        : tensor<...>, tensor<...> -> tensor<...>
      ```

      Memref form (post-bufferization, no result):
      ```mlir
      d2m.local_copy %src, %dst indexing_maps = [#src_map, #dst_map]
        : memref<...>, memref<...>
      ```

      Explicit CB form (after LowerLoadStoreOpsToExplicitCBForm):
      ```mlir
      d2m.local_copy %src into %cb indexing_maps = [#src_map, #dst_map]
        : memref<...> into !d2m.cb<memref<...>>
      ```

      Memref form with MemTx result (after LowerLoadStoreOpsToDMA):
      ```mlir
      %tx = d2m.local_copy %src, %dst indexing_maps = [#src_map, #dst_map]
        : memref<...>, memref<...> -> !d2m.mem_tx
      d2m.dma_wait %tx
      ```
    }];

    let arguments = (ins
      AnyRankedTensorOrMemRef:$src,
      Optional<AnyRankedTensorOrMemRef>:$dst,
      Optional<D2M_CB>:$cb,
      AffineMapArrayAttr:$indexing_maps
    );

    let results = (outs Optional<AnyTypeOf<[AnyRankedTensorOrMemRef, D2M_MemTx]>>:$result);

    let builders = [
      // Implicit form: infers result from dst type (tensor → result, memref → no result).
      OpBuilder<(ins "Value":$src, "Value":$dst, "ArrayAttr":$indexingMaps), [{
        Type dstType = dst.getType();
        TypeRange resultTypes = mlir::isa<RankedTensorType>(dstType)
            ? TypeRange(dstType) : TypeRange{};
        build($_builder, $_state, resultTypes, src, dst, /*cb=*/Value{},
              indexingMaps);
      }]>,
      // MemTx result form (post-LowerLoadStoreOpsToDMA): memref operands, !d2m.mem_tx result.
      OpBuilder<(ins "Type":$resultType, "Value":$src, "Value":$dst, "ArrayAttr":$indexingMaps), [{
        build($_builder, $_state, TypeRange(resultType), src, dst,
              /*cb=*/Value{}, indexingMaps);
      }]>
    ];

    let assemblyFormat = [{
      $src (`,` $dst^)? (`into` $cb^)?
      `indexing_maps` `=` $indexing_maps
      attr-dict `:` type($src) (`,` type($dst)^)? (`into` type($cb)^)?
      (`->` type($result)^)?
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      // Required by DestinationStyleOpInterface.
      MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }

      // Returns true if in explicit CB form (has CB, no dst).
      bool isExplicitCBForm() { return static_cast<bool>(getCb()); }

      // Returns true if in implicit form (has dst, no CB).
      bool isImplicitForm() { return static_cast<bool>(getDst()); }

      // Returns true if the result is a !d2m.mem_tx (post-LowerLoadStoreOpsToDMA).
      bool hasMemTxResult() {
        return getResult() && mlir::isa<MemTxType>(getResult().getType());
      }

      // Returns the CB type, or null if CB is not present.
      CBType getCbType() {
        Value cb = getCb();
        if (!cb) return CBType();
        return mlir::cast<CBType>(cb.getType());
      }


      // Returns the source memref type. Only valid post-bufferization.
      MemRefType getSrcMemRefType() {
        return mlir::cast<MemRefType>(getSrc().getType());
      }

      // Returns the destination memref type from dst or CB underlying type.
      // Only valid post-bufferization or in explicit CB form.
      MemRefType getDstMemRefType() {
        if (getDst())
          return mlir::cast<MemRefType>(getDst().getType());
        if (getCb())
          return getCbType().template getUnderlyingAs<MemRefType>();
        llvm_unreachable("LocalCopyOp has neither dst nor cb");
      }

      // Returns the destination shaped type (works for both tensor and memref).
      ShapedType getDstShapedType() {
        if (getDst())
          return mlir::cast<ShapedType>(getDst().getType());
        if (getCb())
          return getCbType().template getUnderlyingAs<ShapedType>();
        llvm_unreachable("LocalCopyOp has neither dst nor cb");
      }

      /// Returns the CB port number. Op must be in explicit CB form.
      unsigned getCBPort();
    }];

  let extraClassDefinition = [{
    unsigned $cppClass::getCBPort() {
      assert(isExplicitCBForm() && "getCBPort requires explicit CB form");
      auto getCBOp = getCb().getDefiningOp<GetCBOp>();
      assert(getCBOp && "CB operand must be defined by a GetCBOp");
      return getCBOp.getPort();
    }
  }];
}

def D2M_NullTxOp : D2M_GenericRegionDatamovementOp<"null_tx", [Pure]> {
    let summary = "Create a null transaction.";
    let description = [{
      Utility op to create a null transaction.  This is required for creating a sentinel
      starting transaction for a DMA nested inside of a loop nest.
    }];

    let results = (outs D2M_MemTx:$result);

    let builders =
    [
      OpBuilder<(ins "DMAType":$dmaType),
      [{
        build($_builder, $_state, MemTxType::get($_builder.getContext(), dmaType));
      }]>
    ];

    let assemblyFormat = [{ attr-dict `:` qualified(type($result)) }];
}

def D2M_DMAWaitOp : D2M_GenericRegionDatamovementOp<"dma_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "D2M DMA wait Op";
    let description = [{
      Waits for the producer DMA memory transaction to complete.
    }];

    let arguments = (ins D2M_MemTx:$mem_tx);

    let assemblyFormat = [{ $mem_tx attr-dict `:` qualified(type($mem_tx)) }];
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Remote Load/Store Ops (Based on Affine Load/Store)
//===----------------------------------------------------------------------===//

def D2M_RemoteLoadOp : D2M_GenericRegionDatamovementOp<"remote_load",
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]>
  , DestinationStyleOpInterface
  , D2M_ShardDMAOpInterface
  , DeclareOpInterfaceMethods<AffineReadOpInterface>
  , DeclareOpInterfaceMethods<AffineMapAccessInterface>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType",
      "hasTensorSemantics",
      "isNotConflicting"
    ]>
  ]> {
    let summary = "D2M Remote/Aliased Load Op";
    let description = [{
      Loads an _entire shard_ from remote or local GenericOp operand into a
      local L1 buffer. The memref/tensor argument _must_ be an operand of the
      GenericOp.

      RemoteLoadOp indices correspond to the _grid dimensions only_ (first
      N/2 dimensions of the device shape). The RemoteLoadOp always loads an
      entire shard from the corresponding operand.

      This operation is well-defined for both remote and local tensors. Backend
      DMA passes opportunistically lower this operation to aliased CB operations
      if possible, but the semantics for remote and local tensors are identical.

      The operation requires an explicit destination buffer parameter. The
      operation has multiple modes; the explicit CB form and unicast/multicast
      modes are both orthogonal and can be mixed and matched.

      The basic form takes a local buffer as input and loads data into it:

      ```mlir
      d2m.remote_load %buffer %generic_operand[%i, %j]
        : memref<SY x SX x f32, ...>, memref<GY x GX x SY x SX x f32, ...>
        -> memref<SY x SX x f32, ...>
      ```

      The explicit CB form takes a CB (from `d2m.get_cb`) as additional input;
      the load is produced into the CB. The CB provided must be the CB associated
      with the operand. In explicit CB form, no local buffer is required.

      ```mlir
      d2m.remote_load %generic_operand[%i, %j] into %cb
        : memref<GY x GX x SY x SX x f32, ...> into !d2m.cb<memref<SY x SX x f32, ...>>
      ```

      In implicit form, the result matches the shard memref/tensor type and aliases
      the input buffer.

      ## High-Level Multicast Mode

      When a high-level multicast dimension list (mcast) is provided, this
      operation performs a cooperative multicast along the parallel dimension
      specified. All locations in the compute grid that share the same mcast dim value
      will form a multicast group *if all non-mcast dims are reduction dims*.

      The sender vs receiver roles in the multicast group are determined by convention.

      NOTE: If implementing the high-level multicast is not feasible, the
      operation will be lowered to an equivalent unicast remote load instead.

      Example with high-level multicast (implicit form):

      ```mlir
      // Cooperative multicast along dimension 0
      d2m.remote_load %buffer %memref[%i, %j] mcast[%c0]
        : memref<4x6xf32, #l1_>, memref<2x4x4x6xf32, #ttcore.shard<...>, #dram>
        -> memref<4x6xf32, #l1_>
      ```

      ## Low-Level Multicast Mode

      WARNING: This form is currently supported mostly for internal use within
      middle-end pipeline. Picking a concrete multicast shape implicitly
      constrains that the layout of the operand, but this isn't handled properly
      in the frontend grid selection.

      When low-level multicast parameters (mcore/mshape) are provided, this
      operation performs a gather-multicast pattern:
        - By convention, core 0 along each multicast dimension is the sender
        - The sender reads the remote data into local memory
        - The sender then multicasts to all other cores in the mcast group
        starting at `mcastStartIndex` with shape `mcastShape`
        - Receivers wait for the sender to complete

      Example with low-level multicast (implicit form):

      ```mlir
      // Core 0 gathers and multicasts to a 1x4 region starting at mcore[0, 1]
      d2m.remote_load %buffer %memref[%i, %j] mcore[%c0, %c1] mshape[%c1, %c4]
        : memref<4x6xf32, #l1_>, memref<2x4x4x6xf32, #ttcore.shard<...>, #dram>
        -> memref<4x6xf32, #l1_>
      ```

      Constraints:
      - The number of indices must equal N/2, where N is the memref/tensor rank.
      - The operation loads the entire shard into the provided local L1 buffer
      - If in implicit form (no CB):
        - The localBuffer type must match the shard shape of the memref operand
        - The localBuffer is required
      - If in explicit CB form:
        - The operand provided must be a memref
        - The localBuffer must NOT be present
        - The CB underlying type must match the shard shape
      - The two multicast forms are mutually exclusive: either use low-level
      (mcore/mshape) or high-level (mcast dims), not both
    }];

    let arguments = (ins Arg<Optional<AnyRankedTensorOrMemRef>, "the local buffer to load into",
        [MemWrite]>:$localBuffer,
        Arg<AnyRankedTensorOrMemRef, "the reference to load from",
        [MemRead]>:$memref,
        Variadic<Index>:$indices,
        Optional<D2M_CB>:$cb,
        Variadic<Index>:$mcastStartIndex,
        Variadic<Index>:$mcastShape,
        Variadic<Index>:$mcastDims);

    let results = (outs Optional<AnyRankedTensorOrMemRef>:$result);

    let builders = [
      // With CB (no result, no localBuffer)
      OpBuilder<(ins "Value":$memref, "ValueRange":$indices, "Value":$cb),
      [{
        build($_builder, $_state, /*result=*/TypeRange{}, /*localBuffer=*/Value{}, memref, indices, cb, ValueRange(), ValueRange(), ValueRange());
      }]>,
      // With CB and low-level multicast (no result, no localBuffer)
      OpBuilder<(ins "Value":$memref, "ValueRange":$indices, "Value":$cb,
                     "ValueRange":$mcastStartIndex, "ValueRange":$mcastShape),
      [{
        build($_builder, $_state, /*result=*/TypeRange{}, /*localBuffer=*/Value{}, memref, indices, cb, mcastStartIndex, mcastShape, ValueRange());
      }]>,
      // With CB and high-level multicast (no result, no localBuffer)
      OpBuilder<(ins "Value":$memref, "ValueRange":$indices, "Value":$cb,
                     "ValueRange":$mcastDims),
      [{
        build($_builder, $_state, /*result=*/TypeRange{}, /*localBuffer=*/Value{}, memref, indices, cb, ValueRange(), ValueRange(), mcastDims);
      }]>,
      // Without CB (with result, localBuffer required)
      OpBuilder<(ins "Type":$resultType, "Value":$localBuffer, "Value":$memref, "ValueRange":$indices),
      [{
        build($_builder, $_state, resultType, localBuffer, memref, indices, /*cb=*/Value{}, ValueRange(), ValueRange(), ValueRange());
      }]>,
      // Without CB, with low-level multicast (with result, localBuffer required)
      OpBuilder<(ins "Type":$resultType, "Value":$localBuffer, "Value":$memref, "ValueRange":$indices,
                     "ValueRange":$mcastStartIndex, "ValueRange":$mcastShape),
      [{
        build($_builder, $_state, resultType, localBuffer, memref, indices, /*cb=*/Value{}, mcastStartIndex, mcastShape, ValueRange());
      }]>,
      // Without CB, with high-level multicast (with result, localBuffer required)
      OpBuilder<(ins "Type":$resultType, "Value":$localBuffer, "Value":$memref, "ValueRange":$indices,
                     "ValueRange":$mcastDims),
      [{
        build($_builder, $_state, resultType, localBuffer, memref, indices, /*cb=*/Value{}, ValueRange(), ValueRange(), mcastDims);
      }]>
    ];

    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      // Required by DestinationStyleOpInterface.
      // The localBuffer operand is the DPS init (destination) for the load.
      MutableOperandRange getDpsInitsMutable() {
        return getLocalBufferMutable();
      }

      /// Returns the operand index (for this op!) for the local buffer parameter.
      /// Returns UINT_MAX if localBuffer is not present (explicit CB form).
      unsigned getLocalBufferOperandIndex() {
        if (!getLocalBuffer()) return UINT_MAX;
        return 0;
      }

      /// Returns the operand index (for this op!) for the memref/tensor parameter.
      /// In explicit CB form, this is 0; in implicit form, this is 1.
      unsigned getMemRefOperandIndex() {
        return getLocalBuffer() ? 1 : 0;
      }

      void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

      /// Returns the affine map for AffineReadOpInterface. Returns an identity
      /// map over the actual index operands, since the indices on this op are
      /// already the final computed grid indices (not raw iteration variables).
      /// The parent GenericOp's indexing maps describe the iteration-space-to-
      /// operand relationship and may have a different dimensionality (e.g.
      /// after loop fusion), so they cannot be used here.
      AffineMap getAffineMap() {
        unsigned numIndices = getIndices().size();
        if (numIndices == 0) {
          return AffineMap();
        }
        return AffineMap::getMultiDimIdentityMap(numIndices, getContext());
      }

      /// Returns the CB type, or null CBType if CB is not present.
      /// Caller should check getCb() before calling this if null check is needed.
      CBType getCbType() {
        Value cb = getCb();
        if (!cb) return CBType();
        return mlir::cast<CBType>(cb.getType());
      }

      /// Returns the CB port number. Op must be in explicit CB form.
      unsigned getCBPort() {
        assert(isExplicitCBForm() && "getCBPort requires explicit CB form");
        auto getCBOp = getCb().getDefiningOp<GetCBOp>();
        assert(getCBOp && "CB operand must be defined by a GetCBOp");
        return getCBOp.getPort();
      }

      /// Returns true if the circular buffer operand is present (explicit CB form).
      bool isExplicitCBForm() { return static_cast<bool>(getCb()); }

      /// Returns true if the operation is in implicit form (no CB, has localBuffer).
      bool isImplicitForm() { return static_cast<bool>(getLocalBuffer()); }

      /// Returns true if the result is present (result form).
      bool hasResultForm() { return static_cast<bool>(getResult()); }

      /// Returns the shaped type (memref or tensor) of the memref operand.
      ShapedType getShapedType() { return cast<ShapedType>(getMemref().getType()); }

      /// Returns the memref type, or fails if the operand is a tensor.
      MemRefType getMemRefType() {
        if (auto memrefType = dyn_cast<MemRefType>(getMemref().getType())) {
          return memrefType;
        }
        llvm_unreachable("getMemRefType() called on tensor operand, use getShapedType() instead");
      }

      /// Returns true if this is a multicast operation (either form).
      bool isMcast() { return !getMcastShape().empty() || !getMcastDims().empty(); }

      /// Returns true if using high-level mcast form (dimension indices).
      bool isHighLevelMcast() { return !getMcastDims().empty(); }

      /// Returns true if using low-level mcast form (mcore/mshape).
      bool isLowLevelMcast() { return !getMcastShape().empty(); }

      // --- AffineReadOpInterface / AffineMapAccessInterface support ---

      /// Returns the affine map operands (the grid indices).
      Operation::operand_range getMapOperands() { return getIndices(); }

      /// Returns the value read by this operation.
      Value getValue() { return getResult(); }

      /// Returns an AffineMapAttr wrapping the affine map for this operation.
      AffineMapAttr getAffineMapAttr() {
        return AffineMapAttr::get(getAffineMap());
      }

      /// Returns the name of the map attribute.
      static StringRef getMapAttrStrName() { return "map"; }

      /// Returns the AffineMapAttr associated with the given memref value.
      NamedAttribute getAffineMapAttrForMemRef(Value memref) {
        assert(memref == getMemref() &&
               "Expected memref argument to match memref operand");
        return {StringAttr::get(getContext(), getMapAttrStrName()),
                getAffineMapAttr()};
      }
    }];
}

def D2M_RemoteStoreOp : D2M_GenericRegionDatamovementOp<"remote_store",
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]>
  , DestinationStyleOpInterface
  , D2M_ShardDMAOpInterface
  , DeclareOpInterfaceMethods<AffineWriteOpInterface>
  , DeclareOpInterfaceMethods<AffineMapAccessInterface>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType",
      "hasTensorSemantics"
    ]>
  ]> {
    let summary = "D2M Remote/Aliased Store Op";
    let description = [{
      Stores an _entire shard_ from a local buffer to remote or local GenericOp
      operand. The memref/tensor argument _must_ be an operand of the GenericOp.

      RemoteStoreOp indices correspond to the _grid dimensions only_ (first N/2
      dimensions of the device shape). The RemoteStoreOp always stores an entire
      shard to the corresponding operand.

      This operation is well-defined for both remote and local tensors. Backend
      DMA passes opportunistically lower this operation to aliased CB operations
      if possible, but the semantics for remote and local tensors are identical.

      The operation has two forms:
      - Implicit form takes a plain memref/tensor as input to store; it doesn't
      assume a particular buffering implementation (CBs, scratchpad, etc).

      ```mlir
      // For an operand with device shape (GY x GX) x (SY x SX)
      d2m.remote_store %generic_operand[%i, %j] %buf
        : memref<GY x GX x SY x SX x f32, ...>, memref<SY x SX x f32, ...>
        -> memref<GY x GX x SY x SX x f32, ...>
      ```

      - Explicit CB form stores data from a concrete CB (from `d2m.get_cb`) to the remote operand.

      ```mlir
      d2m.remote_store %generic_operand[%i, %j] from %cb
        : memref<GY x GX x SY x SX x f32, ...>, !d2m.cb<memref<SY x SX x f32, ...>>
        -> memref<GY x GX x SY x SX x f32, ...>
      ```

      In both forms, the result matches the type of the operand memref/tensor.
      This allows using the result as yield value in a generic region, as well
      as forming a complete use-chain from remote_load to remote_store.

      Constraints:
      - The number of indices must equal N/2, where N is the memref/tensor rank.
      - The operation stores the entire shard from a local buffer or circular buffer
      - If in explicit CB form:
        - The operand provided must be a memref
        - The CB and operand memref must have compatible shard shapes
      - Exactly one of localBuffer or cb must be present
    }];

    let arguments = (ins Arg<AnyRankedTensorOrMemRef, "the reference to store to",
        [MemWrite]>:$memref,
        Variadic<Index>:$indices,
        Arg<Optional<AnyRankedTensorOrMemRef>, "the local buffer to store from",
        [MemRead]>:$localBuffer,
        Optional<D2M_CB>:$cb,
        Variadic<Index>:$startDevice, Variadic<Index>:$deviceMcastShape,
        Optional<D2M_GlobalSemaphore>:$semaphore,
        Variadic<Index>:$semaphoreIndices
        );
    let results = (outs Optional<AnyRankedTensorOrMemRef>:$result);

    let builders = [
      // Explicit CB form (with CB, no result)
      OpBuilder<(ins "Value":$memref, "ValueRange":$indices, "Value":$cb),
      [{
        build($_builder, $_state, /*result=*/TypeRange{}, memref, indices, /*localBuffer=*/Value{}, cb, /*startDevice=*/ValueRange{}, /*deviceMcastShape=*/ValueRange{}, /*semaphore=*/Value{}, /*semaphoreIndices=*/ValueRange());
      }]>,
      // Implicit form with local buffer and result (required)
      OpBuilder<(ins "Type":$resultType, "Value":$memref, "ValueRange":$indices, "Value":$localBuffer),
      [{
        build($_builder, $_state, resultType, memref, indices, localBuffer, /*cb=*/Value{}, /*startDevice=*/ValueRange{}, /*deviceMcastShape=*/ValueRange{}, /*semaphore=*/Value{}, /*semaphoreIndices=*/ValueRange());
      }]>,
      // Explicit CB form (with CB, no result); with destination device range
      OpBuilder<(ins "Value":$memref, "ValueRange":$indices, "Value":$cb, "ValueRange":$startDevice, "ValueRange":$deviceMcastShape, "Value":$semaphore, "ValueRange":$semaphoreIndices),
      [{
        build($_builder, $_state, /*result=*/TypeRange{}, memref, indices, /*localBuffer=*/Value{}, cb, startDevice, deviceMcastShape, semaphore, semaphoreIndices);
      }]>,
      // Implicit form with local buffer and result (required); with destination device range
      OpBuilder<(ins "Type":$resultType, "Value":$memref, "ValueRange":$indices, "Value":$localBuffer, "ValueRange":$startDevice, "ValueRange":$deviceMcastShape, "Value":$semaphore, "ValueRange":$semaphoreIndices),
      [{
        build($_builder, $_state, resultType, memref, indices, localBuffer, /*cb=*/Value{}, startDevice, deviceMcastShape, semaphore, semaphoreIndices);
      }]>
    ];

    let assemblyFormat = [{
      $memref `[` $indices `]`
      ($localBuffer^)?
      (`from` $cb^)?
      (`devices` `startDevice` `[` $startDevice^ `]`)? (`deviceMcastShape` `[` $deviceMcastShape^ `]`)?
      (`semaphore` `increment` $semaphore^ `[` $semaphoreIndices `]`)?
      attr-dict `:` type($memref)
      (`,` type($localBuffer)^)?
      (`from` type($cb)^)?
      (`->` type($result)^)?
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      // Required by DestinationStyleOpInterface.
      // The memref operand is the DPS init (destination) for the store.
      MutableOperandRange getDpsInitsMutable() {
        return getMemrefMutable();
      }

      /// Returns the operand index of the memref/tensor.
      unsigned getMemRefOperandIndex() { return 0; }

      void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

      /// Returns the affine map for AffineWriteOpInterface. Returns an identity
      /// map over the actual index operands, since the indices on this op are
      /// already the final computed grid indices (not raw iteration variables).
      /// The parent GenericOp's indexing maps describe the iteration-space-to-
      /// operand relationship and may have a different dimensionality (e.g.
      /// after loop fusion), so they cannot be used here.
      AffineMap getAffineMap() {
        unsigned numIndices = getIndices().size();
        if (numIndices == 0) {
          return AffineMap();
        }
        return AffineMap::getMultiDimIdentityMap(numIndices, getContext());
      }

      /// Returns the CB type, or null CBType if CB is not present.
      CBType getCbType() {
        Value cb = getCb();
        if (!cb) return CBType();
        return mlir::cast<CBType>(cb.getType());
      }

      /// Returns the CB port number. Op must be in explicit CB form.
      unsigned getCBPort() {
        assert(isExplicitCBForm() && "getCBPort requires explicit CB form");
        auto getCBOp = getCb().getDefiningOp<GetCBOp>();
        assert(getCBOp && "CB operand must be defined by a GetCBOp");
        return getCBOp.getPort();
      }

      /// Returns true if the circular buffer operand is present (explicit CB form).
      bool isExplicitCBForm() { return static_cast<bool>(getCb()); }

      /// Returns true if the local buffer operand is present (implicit form).
      bool isImplicitForm() { return static_cast<bool>(getLocalBuffer()); }

      /// Returns true if the result is present (result form).
      bool hasResultForm() { return static_cast<bool>(getResult()); }

      /// Returns the shaped type (memref or tensor) of the memref operand.
      ShapedType getShapedType() { return cast<ShapedType>(getMemref().getType()); }

      /// Returns the memref type, or fails if the operand is a tensor.
      MemRefType getMemRefType() {
        if (auto memrefType = dyn_cast<MemRefType>(getMemref().getType())) {
          return memrefType;
        }
        llvm_unreachable("getMemRefType() called on tensor operand, use getShapedType() instead");
      }

      // --- AffineWriteOpInterface / AffineMapAccessInterface support. ---

      /// Returns the affine map operands (the grid indices).
      Operation::operand_range getMapOperands() { return getIndices(); }

      /// Returns the value to store (the local buffer).
      Value getValueToStore() { return getLocalBuffer(); }

      /// Returns the operand index of the value to store.
      unsigned getStoredValOperandIndex() {
        // LocalBuffer follows memref and indices in the operand list.
        // Use the ODS segment info to find the correct index.
        return getLocalBuffer()
          ? static_cast<unsigned>(getODSOperandIndexAndLength(2).first)
          : UINT_MAX;
      }

      /// Returns an AffineMapAttr wrapping the affine map for this operation.
      AffineMapAttr getAffineMapAttr() {
        return AffineMapAttr::get(getAffineMap());
      }

      /// Returns the name of the map attribute.
      static StringRef getMapAttrStrName() { return "map"; }

      /// Returns the AffineMapAttr associated with the given memref value.
      NamedAttribute getAffineMapAttrForMemRef(Value memref) {
        assert(memref == getMemref() &&
               "Expected memref argument to match memref operand");
        return {StringAttr::get(getContext(), getMapAttrStrName()),
                getAffineMapAttr()};
      }
    }];
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Semaphore Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

class D2M_SemaphoreUpdateOp<string mnemonic> : D2M_GenericRegionOp<mnemonic,
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]>
  ]> {
    let summary = "D2M Semaphore Set or Inc Op";
    let description = [{
      Set or increment the semaphore value atomically. This op comes in a few flavors:

      - Set this local core's semaphore value
      ```mlir
      d2m.semaphore_set %sem0, %c1
      ```

      - Update a remote core's semaphore value
      ```mlir
      d2m.semaphore_inc %sem0, %c1 core[%c2, %c2]
      ```

      - Update a remote mcast region of cores' semaphore value
      ```mlir
      d2m.semaphore_set %sem0, %c1 core[%c2, %c2] mcast[%c4, %c4]
      ```
    }];

    let arguments = (ins AnyTypeOf<[D2M_LocalSemaphore, D2M_GlobalSemaphore]>:$semaphore, Index:$value,
                         Variadic<Index>:$dstCoreIndex, Variadic<Index>:$mcastShape,
                         Variadic<Index>:$startDevice, Variadic<Index>:$deviceMcastShape);

    let assemblyFormat = [{ $semaphore `,` $value (`,` `core` `[` $dstCoreIndex^ `]`)? (`mcast` `[` $mcastShape^ `]`)? (`startDevice` `[` $startDevice^ `]`)? (`deviceMcastShape` `[` $deviceMcastShape^ `]`)? attr-dict `:` type($semaphore) }];

    let builders =
    [
      OpBuilder<(ins "Value":$semaphore, "Value":$value),
      [{
        build($_builder, $_state, semaphore, value, ValueRange(), ValueRange(), ValueRange(), ValueRange());
      }]>,
      OpBuilder<(ins "Value":$semaphore, "Value":$value, "ValueRange":$dstCoreIndex),
      [{
        build($_builder, $_state, semaphore, value, dstCoreIndex, ValueRange(), ValueRange(), ValueRange());
      }]>,
      OpBuilder<(ins "Value":$semaphore, "Value":$value, "ValueRange":$dstCoreIndex, "ValueRange":$startDevice, "ValueRange":$deviceMcastShape),
      [{
        build($_builder, $_state, semaphore, value, dstCoreIndex, ValueRange(), startDevice, deviceMcastShape);
      }]>,
    ];
}

def D2M_SemaphoreSetOp : D2M_SemaphoreUpdateOp<"semaphore_set"> {}

def D2M_SemaphoreIncOp : D2M_SemaphoreUpdateOp<"semaphore_inc"> {}

def D2M_SemaphoreWaitOp : D2M_GenericRegionOp<"semaphore_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "D2M Semaphore Set Op.";
    let description = [{
      Wait for the semaphore value to reach the specified value. Optionally supply a reset value as a shorthand syntax.

      ```mlir
      d2m.semaphore_wait %sem1, %c1 reset %c0
      // is equivalent to
      d2m.semaphore_wait %sem1, %c1
      d2m.semaphore_set %sem1, %c0
      ```
    }];

    let arguments = (ins AnyTypeOf<[D2M_LocalSemaphore, D2M_GlobalSemaphore]>:$semaphore, Index:$value, Optional<Index>:$resetValue);

    let assemblyFormat = [{ $semaphore `,` $value (`reset` $resetValue^)? attr-dict `:` type($semaphore) }];

    let builders =
    [
      OpBuilder<(ins "Value":$semaphore, "Value":$value),
      [{
        build($_builder, $_state, semaphore, value, nullptr);
      }]>,
    ];
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Control Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_YieldOp : D2M_GenericRegionOp<"yield", [
    MemoryEffects<[MemRead, MemWrite]>,
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
    let summary = "Yield op.";
    let description = [{
      D2M yield equivalent, required for enforcing pure tensor semantics on the tensor form of the GenericOp.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$values);

    let assemblyFormat = [{ $values attr-dict `:` `(` type($values) `)` }];

    let hasVerifier = 1;

    let extraClassDefinition = [{
      bool $cppClass::bufferizesToMemoryRead(
          mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
        return false;
      }
      bool $cppClass::bufferizesToMemoryWrite(
          mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
        return false;
      }
      mlir::bufferization::AliasingValueList $cppClass::getAliasingValues(
          mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
        mlir::bufferization::AliasingValueList result;
        return result;
      }
      mlir::FailureOr<mlir::bufferization::BufferLikeType> $cppClass::getBufferType(
          mlir::Value, const mlir::bufferization::BufferizationOptions &,
          const mlir::bufferization::BufferizationState &,
          ::llvm::SmallVector<mlir::Value> &) {
        return mlir::failure();
      }
      mlir::LogicalResult $cppClass::bufferize(
          mlir::RewriterBase &rewriter,
          const mlir::bufferization::BufferizationOptions &,
          mlir::bufferization::BufferizationState &state) {
        rewriter.eraseOp(*this);
        return mlir::success();
      }
    }];
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Control Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_SpatialYieldOp : D2M_GenericRegionOp<"spatial_yield", [
    MemoryEffects<[MemRead, MemWrite]>,
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
    let summary = "Yield op for spatial op.";
    let description = [{
      Yield operation for spatial op.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$values);

    let assemblyFormat = [{ $values attr-dict `:` `(` type($values) `)` }];

    let hasVerifier = 1;

    let extraClassDefinition = [{
      bool $cppClass::bufferizesToMemoryRead(
          mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
        return false;
      }
      bool $cppClass::bufferizesToMemoryWrite(
          mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
        return false;
      }
      mlir::bufferization::AliasingValueList $cppClass::getAliasingValues(
          mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
        mlir::bufferization::AliasingValueList result;
        return result;
      }
      mlir::FailureOr<mlir::bufferization::BufferLikeType> $cppClass::getBufferType(
          mlir::Value, const mlir::bufferization::BufferizationOptions &,
          const mlir::bufferization::BufferizationState &,
          ::llvm::SmallVector<mlir::Value> &) {
        return mlir::failure();
      }
      mlir::LogicalResult $cppClass::bufferize(
          mlir::RewriterBase &rewriter,
          const mlir::bufferization::BufferizationOptions &,
          mlir::bufferization::BufferizationState &state) {
        rewriter.eraseOp(*this);
        return mlir::success();
      }
    }];
}
class D2M_CBOp<string mnemonic, list<Trait> traits = []> : D2M_GenericRegionOp<mnemonic,
  traits #
  [ MemoryEffects<[MemRead, MemWrite]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "isNotConflicting"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       , "hasTensorSemantics"
                                                       ]>
  ]> {
  let arguments = (ins D2M_CB:$cb);
  let results = (outs AnyRankedTensorOrMemRef:$result);

  let builders =
  [
    OpBuilder<(ins "Value":$cb),
    [{
      build($_builder, $_state, mlir::cast<CBType>(cb.getType()).getUnderlying(), cb);
    }]>
  ];

  let assemblyFormat = [{ $cb attr-dict `:` type($cb) `->` type($result) }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    CBType getCbType() {
      return mlir::cast<CBType>(getCb().getType());
    }

    ShapedType getUnderlyingType() {
      return getCbType().getUnderlying();
    }

    // Get the wrapped memref type from the circular buffer
    MemRefType getMemRefType() {
      return llvm::cast<MemRefType>(getUnderlyingType());
    }

    // Get the wrapped tensor type from the circular buffer
    RankedTensorType getTensorType() {
      return llvm::cast<RankedTensorType>(getUnderlyingType());
    }
  }];

  let extraClassDefinition = [{
    bool $cppClass::hasTensorSemantics() {
      return getCbType().hasTensorType();
    }

    bool $cppClass::bufferizesToMemoryRead(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      return false;
    }

    bool $cppClass::bufferizesToMemoryWrite(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      return false;
    }

    bool $cppClass::isNotConflicting(mlir::OpOperand *, mlir::OpOperand *,
                                 const mlir::bufferization::AnalysisState &) {
      // Return true to avoid forcing out of place bufferization.
      return true;
    }

    mlir::bufferization::AliasingValueList $cppClass::getAliasingValues(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      mlir::bufferization::AliasingValueList result;
      return result;
    }

    mlir::FailureOr<mlir::bufferization::BufferLikeType> $cppClass::getBufferType(
        mlir::Value value, const mlir::bufferization::BufferizationOptions &,
        const mlir::bufferization::BufferizationState &,
        ::llvm::SmallVector<mlir::Value> &) {
      llvm_unreachable("intentionally unimplemented, this op can only accept block arguments which should have already been converted");
    }

    mlir::LogicalResult $cppClass::bufferize(
        mlir::RewriterBase &rewriter,
        const mlir::bufferization::BufferizationOptions &options,
        mlir::bufferization::BufferizationState &) {
      auto cbBufferType = mlir::cast<bufferization::TensorLikeType>(getCbType()).getBufferType(options, [&] () {
        return this->emitOpError();
      });
      assert(succeeded(cbBufferType));
      auto toBuffer = rewriter.create<bufferization::ToBufferOp>(
          this->getLoc(), *cbBufferType, getCb());
      mlir::bufferization::replaceOpWithNewBufferizedOp<$cppClass>(
          rewriter, *this, toBuffer.getResult());
      return mlir::success();
    }

    mlir::LogicalResult $cppClass::verify() {
      // Verify that the result type matches the wrapped memref type
      auto cbType = llvm::cast<CBType>(getCbType());
      if (cbType.getUnderlying() != getResult().getType()) {
        return emitOpError() << "result type does not match circular buffer's "
                                "wrapped memref type";
      }

      return ::mlir::success();
    }
  }];
}

def D2M_WaitOp : D2M_CBOp<"wait"> {
  let summary = "Wait from circular buffer.";
  let description = [{
    Wait operation, extracts the enclosing memref from a circular buffer.
    This operation is used by consumer threads to access data from the circular buffer.
    It implicitly blocks until a chunk of memref sized underlying type is made
    available by a producer thread via d2m.reserve.

    Each value of !d2m.cb type can be thought of as a shared resource between threads
    akin to a single-producer, single-consumer queue. Where d2m.reserve and d2m.wait
    effectively implement push/pop queue semantics. One distinction which lends itself
    better to DPS style is that both reserve and wait guarantee acquisition of the
    underlying memory after the op has executed.

    The resource is implicitly released at the end of block scope, OR explicitly
    released via d2m.pop.

    Example:
    ```mlir
    %memref = d2m.wait %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    ```
  }];

  let hasCanonicalizer = 1;
}

def D2M_ReserveOp : D2M_CBOp<"reserve"> {
  let summary = "Reserve from circular buffer.";
  let description = [{
    Reserve operation, extracts the enclosing memref from a circular buffer.
    This operation is used by producer threads to reserve a memref sized underlying
    type's worth of space in the circular buffer. It implicitly blocks if there hasn't
    yet been a consumer of the data from another thread via d2m.wait.

    Each value of !d2m.cb type can be thought of as a shared resource between threads
    akin to a single-producer, single-consumer queue. Where d2m.reserve and d2m.wait
    effectively implement push/pop queue semantics. One distinction which lends itself
    better to DPS style is that both reserve and wait guarantee acquisition of the
    underlying memory after the op has executed.

    The resource is implicitly released at the end of block scope, OR explicitly
    released via d2m.push.

    Example:
    ```mlir
    %memref = d2m.reserve %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    ```
  }];
}

def D2M_PushOp : D2M_GenericRegionOp<"push",
  [ MemoryEffects<[MemRead, MemWrite]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
  let summary = "Push to circular buffer (signal producer done).";
  let description = [{
    Push operation, releases the memref acquired via d2m.reserve.
    Signals to consumer threads that data is ready.
    Must be preceded by d2m.reserve in the same block.

    This operation explicitly releases the circular buffer slot acquired by
    d2m.reserve, making the data available to consumer threads. If d2m.push
    is present, the automatic release at block end is skipped for the
    corresponding d2m.reserve operation.

    Example:
    ```mlir
    %memref = d2m.reserve %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    // ... write data to %memref ...
    d2m.push %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    ```
  }];
  let arguments = (ins D2M_CB:$cb);
  let assemblyFormat = [{ $cb attr-dict `:` type($cb) }];

  let extraClassDeclaration = [{
    CBType getCbType() {
      return mlir::cast<CBType>(getCb().getType());
    }

    bool hasTensorSemantics() { return getCbType().hasTensorType(); }
  }];
}

def D2M_PopOp : D2M_GenericRegionOp<"pop",
  [ MemoryEffects<[MemRead, MemWrite]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
  let summary = "Pop from circular buffer (signal consumer done).";
  let description = [{
    Pop operation, releases the memref acquired via d2m.wait.
    Signals to producer threads that space is available.
    Must be preceded by d2m.wait in the same block.

    This operation explicitly releases the circular buffer slot acquired by
    d2m.wait, making space available for producer threads. If d2m.pop is
    present, the automatic release at block end is skipped for the
    corresponding d2m.wait operation.

    Example:
    ```mlir
    %memref = d2m.wait %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    // ... read data from %memref ...
    d2m.pop %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    ```
  }];
  let arguments = (ins D2M_CB:$cb);
  let assemblyFormat = [{ $cb attr-dict `:` type($cb) }];

  let extraClassDeclaration = [{
    CBType getCbType() {
      return mlir::cast<CBType>(getCb().getType());
    }

    bool hasTensorSemantics() { return getCbType().hasTensorType(); }
  }];
}

def D2M_ScratchAllocateOp : D2M_Op<"scratch_allocate", [
    MemoryEffects<[MemAlloc]>
  ]> {
  let summary = "Allocate scratch memory for intermediate spilling.";
  let description = [{
    Allocate a scratch buffer from the L1 scratchpad region for storing
    intermediate results.

    The `slot` attribute is a unique identifier for this scratch allocation
    within the enclosing scope, used for tracking and debugging.

    This op is lowered by `LowerScratchAllocate` to a `memref.subview` into
    the scratch memref anchored by `d2m.scratch_init`.

    Example:
    ```mlir
    %scratch = d2m.scratch_allocate {slot = 0 : i64}
               : memref<1x!ttcore.tile<32x32xbf16>, #ttcore.memory_space<l1>>
    ```
  }];

  let arguments = (ins I64Attr:$slot);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def D2M_ScratchInitOp : D2M_Op<"scratch_init"> {
  let summary = "Anchor op for scratch buffer memref.";
  let description = [{
    Marks a `memref.alloc` as the scratch buffer for intermediate spilling
    inside a `d2m.generic` region. This op consumes the memref to prevent
    canonicalization from removing the otherwise-unused alloc.

    Inserted by `InsertScratchBuffers` and erased by `LowerScratchAllocate`
    after `scratch_allocate` ops are replaced with subviews of this memref.

    Example:
    ```mlir
    %scratch = memref.alloc() : memref<1x32x!ttcore.tile<32x32, f32>,
                                       #ttcore.memory_space<l1>>
    d2m.scratch_init %scratch : memref<1x32x!ttcore.tile<32x32, f32>,
                                       #ttcore.memory_space<l1>>
    ```
  }];

  let arguments = (ins AnyMemRef:$scratch);
  let results = (outs);

  let assemblyFormat = [{ $scratch attr-dict `:` type($scratch) }];
}

def D2M_StoreOp : D2M_GenericRegionOp<"store",
  [ SameTypeOperands
  , MemoryEffects<[MemRead, MemWrite]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "isNotConflicting"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  ]> {
  let summary = "TODO.";
  let description = [{
    TODO
  }];

  let arguments = (ins AnyRankedTensor:$dst, AnyRankedTensor:$src);

  let assemblyFormat = [{ $dst `,` $src attr-dict `:` type($dst) }];

  let extraClassDefinition = [{
    bool mlir::tt::d2m::StoreOp::bufferizesToMemoryRead(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      return false;
    }

    bool mlir::tt::d2m::StoreOp::bufferizesToMemoryWrite(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      return false;
    }

    bool mlir::tt::d2m::StoreOp::isNotConflicting(mlir::OpOperand *, mlir::OpOperand *,
                                 const mlir::bufferization::AnalysisState &) {
      // Return true to avoid forcing out of place bufferization.
      return true;
    }

    mlir::bufferization::AliasingValueList mlir::tt::d2m::StoreOp::getAliasingValues(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      mlir::bufferization::AliasingValueList result;
      return result;
    }

    mlir::FailureOr<mlir::bufferization::BufferLikeType> mlir::tt::d2m::StoreOp::getBufferType(
        mlir::Value value, const mlir::bufferization::BufferizationOptions &,
        const mlir::bufferization::BufferizationState &,
        ::llvm::SmallVector<mlir::Value> &) {
      llvm_unreachable("intentionally unimplemented, this op will be erased during bufferization");
    }

    mlir::LogicalResult mlir::tt::d2m::StoreOp::bufferize(
        mlir::RewriterBase &rewriter,
        const mlir::bufferization::BufferizationOptions &options,
        mlir::bufferization::BufferizationState &state) {
      auto src = bufferization::getBuffer(rewriter, getSrc(), options, state);
      if (failed(src)) {
        return mlir::failure();
      }
      auto dst = bufferization::getBuffer(rewriter, getDst(), options, state);
      if (failed(dst)) {
        return mlir::failure();
      }
      rewriter.replaceAllUsesWith(*src, *dst);
      rewriter.eraseOp(*this);
      return mlir::success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Indexing Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

class D2M_IndexOp<string mnemonic, list<Trait> traits = []> : D2M_GenericRegionOp<mnemonic,
  traits #
  [ Pure
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  , DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
  ]> {
    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim);
    let results = (outs Index:$result);
    let assemblyFormat = [{ `(` $dim `)` attr-dict `:` type($result) }];
    let hasFolder = false;

    let builders =
    [
      OpBuilder<(ins "int64_t":$dim),
      [{
        build($_builder, $_state, $_builder.getIndexType(), $_builder.getI64IntegerAttr(dim));
      }]>
    ];
}

def D2M_IterIndexOp : D2M_IndexOp<"iter_index"> {
    let summary = "Iter Index op.";
    let description = [{
      Return the index of the current element in the iteration for the given generic op dimension.
    }];
}

def D2M_BlockIndexOp : D2M_IndexOp<"block_index"> {
    let summary = "Block Index op.";
    let description = [{
      Return the index for the given block dimension. This op represents
      a symbolic index that requires grid/block calculation during
      loop generation. It is created when expanding indexing map expressions
      and requires combining grid indices with block factors and loop indices
      to compute the final value.
    }];
}

def D2M_BlockOffsetOp : D2M_IndexOp<"block_offset"> {
    let summary = "Block Offset op.";
    let description = [{
      Represents block_factor * core_index as a constant-like operation. Useful
      for representing blocking index math in an affine dialect compatible way.
    }];
    let hasFolder = false;
}

def D2M_GetBlockFactorOp : D2M_IndexOp<"get_block_factor"> {
    let summary = "Get block factor op.";
    let description = [{
      Retrieve the block factor from the block factor array attribute in
      the parent generic op at the provided dimension index.
    }];
    let hasFolder = false;
}

def D2M_CoreIndexOp : D2M_GenericRegionOp<"core_index",
  [ Pure
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  , DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
  ]> {
    let summary = "Core Index op.";
    let description = [{
      Return the index of this core's coordinate inside the generic op's grid dimension.
    }];

    // Optional physical-to-virtual grid mapping. When present, this op returns
    // the virtualized core coordinate for `dim`.
    let arguments = (ins
      ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim,
      OptionalAttr<AffineMapAttr>:$phys_to_virt_map
    );

    let results = (outs Index:$result);
    let assemblyFormat = [{ `(` $dim `)` attr-dict `:` type($result) }];
    let hasFolder = false;

    let builders =
    [
      OpBuilder<(ins "int64_t":$dim),
      [{
        build($_builder, $_state, $_builder.getIndexType(),
              $_builder.getI64IntegerAttr(dim), nullptr);
      }]>,
      OpBuilder<(ins "int64_t":$dim, "mlir::AffineMap":$map),
      [{
        build($_builder, $_state, $_builder.getIndexType(),
              $_builder.getI64IntegerAttr(dim), mlir::AffineMapAttr::get(map));
      }]>
    ];

    let hasVerifier = 1;
}

def D2M_MeshPositionOp : D2M_GenericRegionOp<"mesh_position", [ Pure ]> {
    let summary = "Mesh position op.";
    let description = [{
      Return the mesh position for the current device at the given dimension.
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim);
    let results = (outs Index:$result);
}


//===----------------------------------------------------------------------===//
// D2M Remote Access ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_GetCBOp : D2M_GenericRegionOp<"get_cb", [Pure]> {
    let summary = "Get a circular buffer for the given port.";
    let description = [{
      Obtain a CB handle for the specified port number.  Each unique port maps
      to exactly one physical circular buffer.

      The port number is a local identifier — it does not correspond to the
      parent generic op's operand index.  The binding between a CB port and a
      generic operand is established by `remote_load`/`remote_store` ops.

      ```mlir
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>
      %buf = d2m.wait %cb0 : !d2m.cb<memref<...>> -> memref<...>
      ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$port,
                         OptionalAttr<I64Attr>:$operand_index,
                         OptionalAttr<D2M_ResolutionStageAttr>:$resolution_stage);
    let results = (outs D2M_CB:$result);

    let assemblyFormat = [{ `(` $port `)` (`operand_index` `=` $operand_index^)? (`resolution_stage` `=` $resolution_stage^)? attr-dict `:` type($result) }];
}

def D2M_GetArgOp : D2M_GenericRegionOp<"get_arg", [Pure]> {
    let summary = "Get a generic argument for the given operand index.";
    let description = [{
      Obtain a handle for the specified operand index. The result type determines
      how the argument is interpreted: a memref type indicates a buffer address,
      a local/global semaphore type indicates a semaphore.

      Used to reference additional arguments passed to the parent generic op
      from within a thread region or a standalone kernel function.

      ```mlir
      %buf   = d2m.get_arg(0) : memref<...>
      %lsem  = d2m.get_arg(3) : !d2m.local_semaphore
      %gsem  = d2m.get_arg(4) : !d2m.global_semaphore
      ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$operand_index,
                         OptionalAttr<D2M_ResolutionStageAttr>:$resolution_stage);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ `(` $operand_index `)` (`resolution_stage` `=` $resolution_stage^)? attr-dict `:` type($result) }];
}

//===----------------------------------------------------------------------===//
// D2M Write Mask Tile Ops (Used for partial tile masking)
//===----------------------------------------------------------------------===//

def D2M_WriteRowMaskTileOp : D2M_GenericRegionOp<"write_row_mask_tile",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Write row mask pattern to a tile CB";
    let description = [{
        The `write_row_mask_tile` operation writes a row mask pattern
        to a tile CB, where element[i,j] = 1.0 if i < validRows, else 0.0.

        This is used for partial tile OOB masking. The operation writes directly
        to L1 memory, avoiding DST register pressure.
    }];

    let arguments = (ins AnyTypeOf<[Index, AnyInteger]>:$validRows,
                         AnyTypeOf<[AnyRankedTensorOrMemRef, D2M_CB]>:$output);

    let assemblyFormat = [{ `(` $validRows `)` `to` $output attr-dict `:` type($validRows) `,` type($output) }];
}

def D2M_WriteColMaskTileOp : D2M_GenericRegionOp<"write_col_mask_tile",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Write column mask pattern to a tile CB";
    let description = [{
        The `write_col_mask_tile` operation writes a column mask pattern
        to a tile CB, where element[i,j] = 1.0 if j < validCols, else 0.0.

        This is used for partial tile OOB masking. The operation writes directly
        to L1 memory, avoiding DST register pressure.
    }];

    let arguments = (ins AnyTypeOf<[Index, AnyInteger]>:$validCols,
                         AnyTypeOf<[AnyRankedTensorOrMemRef, D2M_CB]>:$output);

    let assemblyFormat = [{ `(` $validCols `)` `to` $output attr-dict `:` type($validCols) `,` type($output) }];
}

//===----------------------------------------------------------------------===//
// D2M Arange Ops
//===----------------------------------------------------------------------===//

def D2M_FillArangeTileOp : D2M_GenericRegionOp<"fill_arange_tile",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Write full linear index pattern to a tile CB";
  let description = [{
      The `fill_arange_tile` operation writes a linear index pattern
      to a tile CB, where element[i,j] = i * 32 + j in the tile's native data
      format (f32, bf16, or i32).

      This is used for arange operations. Each element gets its linear index value
      (0-1023) within the tile, which can be used to compute global indices.

      The resulting tile looks like:
      [[  0,   1,   2, ...,  31],
        [ 32,  33,  34, ...,  63],
        [ 64,  65,  66, ...,  95],
        ...
        [992, 993, 994, ..., 1023]]
  }];

  let arguments = (ins AnyTypeOf<[AnyRankedTensorOrMemRef, D2M_CB]>:$output);

  let assemblyFormat = [{ `to` $output attr-dict `:` type($output) }];
}

def D2M_ArangeBlockOp : D2M_GenericRegionComputeOp<"arange_block",
  [DestinationStyleOpInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<BufferizableOpInterface,
    ["bufferizesToMemoryRead", "bufferizesToMemoryWrite", "bufferize",
    "getAliasingValues", "getBufferType"]>,
  D2M_SkipOpEltwiseFusionTrait]> {
  let summary = "Generate arange values in a block/memref";
  let description = [{
      The `arange_block` operation fills a block/memref with arange values:
      output[i] = start + step * i, where i is the linear index.

      This op operates at the block level (memref of tiles).

      The index_tile_tensor is a scratch tensor (1x1 tile) handled by
      the FillArangeTileOp.

      For each tile at position (tile_row, tile_col), the computation is:
      1. tile_offset = (tile_row * num_tile_cols * 32 + tile_col) * 32
      2. global_idx = tile_add(tile_offset, local_idx)
      3. result = tile_add(tile_mul(global_idx, step), start)

      Example:
      ```mlir
      %result = d2m.arange_block %output, %index_tile
                        {num_elements = 64, start = 0, step = 1}
          : (tensor<1x2x!ttcore.tile<32x32, f32>>,
              tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
      ```
  }];

  let arguments = (ins
      AnyRankedTensorOrMemRef:$index_tile_tensor,
      AnyRankedTensorOrMemRef:$output,
      I64Attr:$num_elements,
      I64Attr:$start,
      I64Attr:$step);

  let results = (outs AnyRankedTensorOrMemRef:$result);

  let builders = [
    OpBuilder<(ins "Value":$indexTileTensor, "Value":$output,
                    "int64_t":$numElements, "int64_t":$start, "int64_t":$step), [{
      // Result type is the same as the output type.
      Type resultType = output.getType();
      build($_builder, $_state, resultType, indexTileTensor, output,
            $_builder.getI64IntegerAttr(numElements),
            $_builder.getI64IntegerAttr(start),
            $_builder.getI64IntegerAttr(step));
    }]>
  ];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];

  let hasVerifier = 1;
}

def D2M_DeviceSynchronizeOp: D2M_GenericRegionOp<"device_synchronize",
  [AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Device synchronize op.";
  let description = [{
    The different devices in a mesh are not running ops in a synchronized manner
    (devices can be running different ops in the program) so in ops where we are transferring data between devices,
    we need to ensure that receiving devices have started the op before senders send data to them.

    This op allows us to do this in two steps:
    1) receiver devices inform senders that they have started the op by incrementing the sync_semaphore on the sender device
    2) senders wait for all receivers that they have to send to using a semaphore wait

    To ensure correct behaviour, we have to correctly set senderStartDevice, senderDeviceMcastShape, and
    numReceivers based on which devices are sending to which receivers. Note that the senders and receivers
    to be specified are in terms of the own device's perspective.

    Example 1: All gather on a 1x8 mesh along cluster axis 1:
      We have to set senderStartDevice to [0, 0], senderDeviceMcastShape to [1, 8], and numReceivers to 7 for ALL devices.

    Example 2: broadcast CCL from device x to all other devices in 1x8 mesh:
      - on device x, senderStartDevice/senderDeviceMcastShape will be empty since it is not receiving any data from other devices, but
      numReceivers will be 7 since it is sending to all other devices
      - on other devices, senderStartDevice will be device x, senderDeviceMcastShape will be empty, and numReceivers will be 0 (not sending to anyone)

    Pre-condition: sync_semaphore is initialized to 0.
    Post-condition: Receiver devices are guaranteed to have started their op
      and so we can send data to them.
  }];

  let arguments = (ins
    D2M_GlobalSemaphore:$sync_semaphore,
    Variadic<Index>:$senderStartDevice,
    Variadic<Index>:$senderDeviceMcastShape,
    I32Attr:$numReceivers,
    Variadic<Index>:$coreIndices
  );
}

def D2M_PrintOp : D2M_GenericRegionOp<"print", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Print values from a kernel thread.";
    let description = [{
        std::format-style format string with `{}` placeholders, lowered to
        ttkernel.dprint. Zero or more values may be interpolated.

        Example:
        ```
        d2m.print("semaphore addr = {}\n", %sem) : (!d2m.local_semaphore)
        d2m.print("x={} y={}\n", %x, %y) : (ui32, ui32)
        d2m.print("done\n") : ()
        ```

        Notes:
          - Only the trivial `{}` format specifier is currently supported.
          - Must double-escape newlines and other special characters.
    }];

    let arguments = (ins StrAttr:$fmt, Variadic<AnyType>:$argv);

    let assemblyFormat = "`(` $fmt (`,` $argv^)? `)` attr-dict `:` functional-type($argv, results)";

    let extraClassDeclaration = [{
      template <typename... Args>
      static void build(::mlir::OpBuilder &b, ::mlir::OperationState &state,
                        const char *fmt, Args &&...argv) {
        return build(b, state, ::mlir::StringRef(fmt),
                     ::mlir::ValueRange{std::forward<Args>(argv)...});
      }
    }];
}

#endif // TTMLIR_TTMLIR_DIALECT_D2M_D2MGENERICREGIONOPS_TD
