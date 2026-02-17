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

    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      assert(this->getNumOperands() == 2 && "Binary operation must have 2 operands");
      auto rhsType = this->getOperand(1).getType();
      // If RHS is a scalar (not a tile), only load LHS from DST (SFPU path)
      if (!mlir::isa<::mlir::tt::ttcore::TileType>(rhsType)) {
        return {0};
      }
      // Both operands are tiles - FPU reads directly from CBs, no DST load needed
      return {};
    }
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

def D2M_TileAddOp : D2M_GenericRegionComputeFPUOrSFPUBinary<"tile_add", [Pure, D2M_SkipOpEltwiseFusionTrait]> {
    let summary = "D2M Tile Add Op";
    let description = [{
        The `tile_add` operation adds two tiles element-wise on the FPU,
        or adds a scalar value to each element in the input tile if rhs is a scalar (SFPU path).
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSubOp : D2M_GenericRegionComputeFPUOrSFPUBinary<"tile_sub", [Pure, D2M_SkipOpEltwiseFusionTrait]> {
    let summary = "D2M Tile Sub Op";
    let description = [{
        The `tile_sub` operation subtracts two tiles element-wise on the FPU,
        or subtracts a scalar value from each element in the input tile if rhs is a scalar (SFPU path).
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileMulOp : D2M_GenericRegionComputeFPUOrSFPUBinary<"tile_mul", [Pure, D2M_SkipOpEltwiseFusionTrait]> {
    let summary = "D2M Tile Mul Op";
    let description = [{
        The `tile_mul` operation multiplies two tiles element-wise on the FPU,
        or multiplies each element in the input tile by a scalar value if rhs is a scalar (SFPU path).
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
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
}

def D2M_TileExpOp : D2M_GenericRegionComputeUnaryDstOp<"tile_exp", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Exp Op";
    let description = [{
        The `tile_exp` operation computes the exponential of each element in the input tile.
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

def D2M_TileTanOp : D2M_GenericRegionComputeUnaryDstOp<"tile_tan", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Tan Op";
    let description = [{
        The `tile_tan` operation computes the tangent function of each element in the input tile.
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

def D2M_TileSigmoidOp : D2M_GenericRegionComputeUnaryDstOp<"tile_sigmoid", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sigmoid Op";
    let description = [{
        The `tile_sigmoid` operation computes the sigmoid of each element in the input tile.
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

    let arguments = (ins TTCore_Tile:$input, F32Attr:$min, F32Attr:$max);
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
        The `tile_reduce_sum` operation computes the sum of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- sum<dims>(A * B, C)
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
}

def D2M_TileReduceMaxOp : D2M_GenericRegionComputeOp<"tile_reduce_max", D2M_TileComputeOpTraitBundle> {
  let summary = "D2M Tile Reduce Max Op";
  let description = [{
        The `tile_reduce_max` operation computes the max of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- max<dims>(A * B, C)
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
}

// TODO(#5968): fusion w/ implicit bcast causes multiple failure scenarios.
def D2M_TileBcastOp : D2M_GenericRegionComputeOp<"tile_bcast", [Pure, D2M_SkipOpEltwiseFusionTrait]> {
  let summary = "D2M Tile Broadcast Op";
  let description = [{
    The `tile_bcast` operation broadcasts a row/col/scalar tile into a full tile.
  }];

  let arguments = (ins TTCore_Tile:$input,
                       D2M_TileBcastTypeAttr:$bcast_type);
  let results = (outs TTCore_Tile:$result);
}

def D2M_PackerMaskResetOp : D2M_GenericRegionComputeOp<"packer_mask_reset"> {
  let summary = "D2M Packer Mask Reset Op";
  let description = [{
        The `packer_mask_reset` operation resets the packer mask. This should be removed once LLK refactors the reduce API.
        See tt-metal#22904 and tt-metal -> reduce.h.
    }];
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
        The `tile_untilize_block` operation untilizes the input tilized memref block and outputs the memref contianing the row major data.
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
    let summary = "D2M Tile Fill Op";
    let description = [{
        The `tile_fill` operation creates a tile filled with a constant scalar value.
        All elements in the resulting tile will have the same value as the input scalar.

        This operation abstracts away the complexity of creating a constant-filled tile,
        providing a simple interface that takes a scalar value and produces a tile suitable
        for element-wise operations.
    }];

    let arguments = (ins AnyFloat:$value);
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
        // The operand is a scalar (float), not a tile
        bool isScalarOperand(int64_t operandIdx) {
          return true;
        }
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

    let summary = "Fully indexed and lowered D2M DMA Op for writes";
    let description = [{
      This op performs a fully indexed DMA write operation from src memref to dst memref, where src and dst indices are fully resolved
      (all view and memspace-specific affine maps applied). It corresponds 1:1 to a single concrete DMA write operation.

      This operation supports multicast; providing an empty multicast shape argument _implies a unicast write_.

      ```mlir
      %tx = d2m.dma_write %src[%gridy, %gridx, %shardy, %shardx], %dst[%gridy, %gridx, %shardy, %shardx], <%size>
        : (memref<2x2x3x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x3x4x!ttcore.tile<32x32, f32>, #dram>) -> !d2m.mem_tx
      ```

      Constraints:
      - src MUST be a LOCAL memref!
      - src and dst must have the same element type.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$src, Variadic<Index>:$srcIndices,
                         AnyRankedTensorOrMemRef:$dst, Variadic<Index>:$dstIndices,
                         I64Attr:$numElems,
                         Variadic<Index>:$mcastStartIndex, Variadic<Index>:$mcastShape);
    let results = (outs D2M_MemTx:$result);

    let builders =
    [
      // Unicast - multicast args are defaulted
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "size_t": $numElems),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, srcIndices, dst, dstIndices, $_builder.getI64IntegerAttr(numElems), ValueRange(), ValueRange());
      }]>,

      // Multicast - multicast args must be provided
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape, "size_t": $numElems),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, srcIndices, dst, dstIndices, $_builder.getI64IntegerAttr(numElems), mcastStartIndex, mcastShape);
      }]>,
    ];

    let assemblyFormat = [{ $src `[` $srcIndices `]` `,` $dst `[` $dstIndices `]` (`core` `[` $mcastStartIndex^ `]` `mcast` `[` $mcastShape `]`)?  `,` `<` $numElems `>` attr-dict `:` `(` type($src) `,` type($dst) `)` `->` type($result)}];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      MemRefType getSrcMemRefType() { return cast<MemRefType>(getSrc().getType()); }
      MemRefType getDstMemRefType() { return cast<MemRefType>(getDst().getType()); }
      size_t getSizeBytes() {
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

    let summary = "Fully indexed and lowered D2M DMA Op for read";
    let description = [{
      This op performs a fully indexed DMA read operation from src memref to dst memref, where src and dst indices are fully resolved
      (all view and memspace-specific affine maps applied). It corresponds 1:1 to a single concrete DMA read operation. The dst memref _must_ be local!

      This operation only supports unicast (mulicast are write-only).

      ```mlir
      %tx = d2m.dma_read %src[%gridy, %gridx, %shardy, %shardx], %dst[%gridy, %gridx, %shardy, %shardx], %size
        : (memref<2x2x3x4x!ttcore.tile<32x32, f32>, #dram>, memref<2x2x3x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
      ```

      Constraints:
      - src MUST be a REMOTE memref! if both operands are local, use DMAWriteOp
      - dst MUST be a LOCAL memref!
      - src and dst must have the same element type.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$src, Variadic<Index>:$srcIndices,
                         AnyRankedTensorOrMemRef:$dst, Variadic<Index>:$dstIndices,
                         I64Attr:$numElems);
    let results = (outs D2M_MemTx:$result);

    let assemblyFormat    = [{ $src `[` $srcIndices `]` `,` $dst `[` $dstIndices `]` `,` `<` $numElems `>` attr-dict `:` `(` type($src) `,` type($dst) `)` `->` type($result)}];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      MemRefType getSrcMemRefType() { return cast<MemRefType>(getSrc().getType()); }
      MemRefType getDstMemRefType() { return cast<MemRefType>(getDst().getType()); }
      size_t getSizeBytes() {
        auto elementSizeBytes =
          ttcore::getElementSizeBytes(getSrcMemRefType().getElementType());
        return getNumElems() * elementSizeBytes;
      }
      ::mlir::tt::ttcore::MemorySpace getSrcMemorySpace() {
        return mlir::cast<::mlir::tt::ttcore::MemorySpaceAttr>(getSrcMemRefType().getMemorySpace()).getValue();
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
      OpBuilder<(ins),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>());
      }]>
    ];

    let assemblyFormat = [{ attr-dict }];
}

def D2M_DMAWaitOp : D2M_GenericRegionDatamovementOp<"dma_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "D2M DMA wait Op";
    let description = [{
      Waits for the producer DMA memory transaction to complete.
    }];

    let arguments = (ins D2M_MemTx:$mem_tx);

    let assemblyFormat = [{ $mem_tx attr-dict }];
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Remote Load/Store Ops (Based on Affine Load/Store)
//===----------------------------------------------------------------------===//

def D2M_RemoteLoadOp : D2M_GenericRegionDatamovementOp<"remote_load",
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]>
  , DestinationStyleOpInterface
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType",
      "hasTensorSemantics"
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

      The explicit CB form takes a CB block arg as additional input; the load is
      produced into the CB. The CB provided must be the CB block arg associated
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

    let arguments = (ins Optional<AnyRankedTensorOrMemRef>:$localBuffer,
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

      /// Returns the identity affine map used to index the memref/tensor for this operation.
      /// The map always has N/2 results where N is the memref/tensor rank (grid dimensions).
      AffineMap getAffineMap() {
        auto shapedType = getShapedType();
        int64_t gridRank = shapedType.getRank() / 2;
        return AffineMap::getMultiDimIdentityMap(gridRank, getContext());
      }

      /// Returns the CB type, or null CBType if CB is not present.
      /// Caller should check getCb() before calling this if null check is needed.
      CBType getCbType() {
        Value cb = getCb();
        if (!cb) return CBType();
        return mlir::cast<CBType>(cb.getType());
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
    }];
}

def D2M_RemoteStoreOp : D2M_GenericRegionDatamovementOp<"remote_store",
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]>
  , DestinationStyleOpInterface
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

      - Explicit CB form stores data from a concrete CB block arg to the remote operand.

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
        Optional<AnyRankedTensorOrMemRef>:$localBuffer,
        Optional<D2M_CB>:$cb);

    let results = (outs Optional<AnyRankedTensorOrMemRef>:$result);

    let builders = [
      // Explicit CB form (with CB, no result)
      OpBuilder<(ins "Value":$memref, "ValueRange":$indices, "Value":$cb),
      [{
        build($_builder, $_state, /*result=*/TypeRange{}, memref, indices, /*localBuffer=*/Value{}, cb);
      }]>,
      // Implicit form with local buffer and result (required)
      OpBuilder<(ins "Type":$resultType, "Value":$memref, "ValueRange":$indices, "Value":$localBuffer),
      [{
        build($_builder, $_state, resultType, memref, indices, localBuffer, /*cb=*/Value{});
      }]>
    ];

    let assemblyFormat = [{ $memref `[` $indices `]` ($localBuffer^)? (`from` $cb^)? attr-dict `:` type($memref) (`,` type($localBuffer)^)? (`from` type($cb)^)? (`->` type($result)^)? }];

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

      /// Returns the identity affine map used to index the memref/tensor for this operation.
      /// The map always has N/2 results where N is the memref/tensor rank (grid dimensions).
      AffineMap getAffineMap() {
        auto shapedType = getShapedType();
        int64_t gridRank = shapedType.getRank() / 2;
        return AffineMap::getMultiDimIdentityMap(gridRank, getContext());
      }

      /// Returns the CB type, or null CBType if CB is not present.
      /// Caller should check getCb() before calling this if null check is needed.
      CBType getCbType() {
        Value cb = getCb();
        if (!cb) return CBType();
        return mlir::cast<CBType>(cb.getType());
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

    let arguments = (ins AnyTypeOf<[D2M_Semaphore, D2M_GlobalSemaphore]>:$semaphore, Index:$value,
                         Variadic<Index>:$dstCoreIndex, Variadic<Index>:$mcastShape);

    let assemblyFormat = [{ $semaphore `,` $value (`,` `core` `[` $dstCoreIndex^ `]`)? (`mcast` `[` $mcastShape^ `]`)? attr-dict `:` type($semaphore) }];

    let builders =
    [
      OpBuilder<(ins "Value":$semaphore, "Value":$value),
      [{
        build($_builder, $_state, semaphore, value, ValueRange(), ValueRange());
      }]>,
      OpBuilder<(ins "Value":$semaphore, "Value":$value, "ValueRange":$dstCoreIndex),
      [{
        build($_builder, $_state, semaphore, value, dstCoreIndex, ValueRange());
      }]>,
    ];
}

def D2M_SemaphoreSetOp : D2M_SemaphoreUpdateOp<"semaphore_set"> {}

def D2M_SemaphoreIncOp : D2M_SemaphoreUpdateOp<"semaphore_inc"> {}

def D2M_SemaphoreWaitOp : D2M_GenericRegionOp<"semaphore_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "D2M Semaphore Set Op.";
    let description = [{
      Wait for the semaphore value to reach the specified value. Optionall supply a reset value as a shorthand syntax.

      ```mlir
      d2m.semaphore_wait %sem1, %c1 reset %c0
      // is equivalent to
      d2m.semaphore_wait %sem1, %c1
      d2m.semaphore_set %sem1, %c0
      ```
    }];

    let arguments = (ins AnyTypeOf<[D2M_Semaphore, D2M_GlobalSemaphore]>:$semaphore, Index:$value, Optional<Index>:$resetValue);

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
    let hasFolder = true;

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
    let hasFolder = true;

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
}

//===----------------------------------------------------------------------===//
// D2M Remote Access ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_GetGlobalOperandOp : D2M_GenericRegionOp<"get_global_operand", [Pure]> {
    let summary = "Get global operand op.";
    let description = [{
      Access the global, aka parent generic op, operand at the specified index.

      The following forms are all equivalent, but the latter forms are required
      when moving generic regions to top level func ops in the module.
      ```mlir
      d2m.generic (%arg0, %arg1, %arg2) {
        ^datamovement(%cb0, %cb1, %cb2)
          %i = d2m.iter_index(0) : index
          %j = d2m.iter_index(1) : index
          d2m.remote_load %arg1[%i, %j] into %cb1 // Capture %arg1 from parent scope
      }
      ```

      And:
      ```mlir
      d2m.generic (%arg0, %arg1, %arg2) {
        ^datamovement(%cb0, %cb1, %cb2)
          %operand_arg1 = d2m.get_global_operand 1
          %i = d2m.iter_index(0) : index
          %j = d2m.iter_index(1) : index
          d2m.remote_load %operand_arg1[%i, %j] into %cb0
      }
      ```

      And:
      ```mlir
      func.func @main(...) {
        d2m.generic (%arg0, %arg1, %arg2) { kernel_symbols = [@dm0] }
      }

      func.func private @dm0(%cb0, %cb1, %cb2) {
        %operand_arg1 = d2m.get_global_operand 1
        %i = d2m.iter_index(0) : index
        %j = d2m.iter_index(1) : index
        d2m.remote_load %operand_arg1[%i, %j] into %cb0
      }
      ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$operand_index);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ `(` $operand_index `)` attr-dict `:` type($result) }];
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

#endif // TTMLIR_TTMLIR_DIALECT_D2M_D2MGENERICREGIONOPS_TD
