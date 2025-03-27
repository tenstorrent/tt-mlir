// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"  // check
#include "llvm/ADT/SmallString.h" // check
#include "llvm/ADT/SmallVector.h" // check

#include "mlir/IR/MLIRContext.h" // Include the MLIR context
#include "mlir/IR/Operation.h" // Include the operation definition
#include <iostream>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRAUTOMATICDATAPARALLELIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRAutomaticDataParallelization : public impl::TTIRAutomaticDataParallelizationBase<TTIRAutomaticDataParallelization> {
public:
  using impl::TTIRAutomaticDataParallelizationBase<TTIRAutomaticDataParallelization>::TTIRAutomaticDataParallelizationBase;

  void func0(MLIRContext* context) {
    mlir::BFloat16Type childType = BFloat16Type::get(context);
    
    if(childType.isF16()) {
      llvm::outs() << "wrong!\n"; 
    }
  
    if(childType.isBF16()) {
      llvm::outs() << "right!\n"; 
    }
  
    mlir::Type baseType = cast<mlir::Type>(childType);
  
    if(baseType.isBF16()) {
      llvm::outs() << "right!\n"; 
    }
  
    if(mlir::isa<mlir::BFloat16Type>(baseType)) {
      llvm::outs() << "right!\n"; 
    }
  
    [[ maybe_unused ]] mlir::BFloat16Type childTypeAgain = mlir::dyn_cast<mlir::BFloat16Type>(baseType);
  }

  void func1(MLIRContext* context) {
    mlir::IntegerType childType = IntegerType::get(context, 32, mlir::IntegerType::SignednessSemantics::Signed);

    [[ maybe_unused ]] unsigned width = childType.getWidth();
    [[ maybe_unused ]] mlir::IntegerType::SignednessSemantics someSigned = childType.getSignedness();
  }

  void func2(MLIRContext* context) {
    llvm::SmallVector<int64_t> shape = {1, 4};
    mlir::Float32Type float32Type = Float32Type::get(context);
    [[ maybe_unused ]] mlir::RankedTensorType childType = RankedTensorType::get(shape, float32Type);
    llvm::outs() << childType << "\n";

    llvm::SmallVector<mlir::NamedAttribute> attributes;
    attributes.push_back(NamedAttribute(StringAttr::get(context, "taps"), StringAttr::get(context, "hehe")));
    mlir::DictionaryAttr dictionaryAttr = DictionaryAttr::get(context, attributes);
    mlir::RankedTensorType anotherChildType = RankedTensorType::get(shape, float32Type, dictionaryAttr);
    llvm::outs() << anotherChildType << "\n";

    mlir::ShapedType parentType = mlir::cast<mlir::RankedTensorType>(childType);
    llvm::outs() << "num-elements=" << parentType.getNumElements() << "\n";
    llvm::outs() << "num-rank=" << parentType.getRank() << "\n";
    llvm::outs() << "element-type=" << parentType.getElementType() << "\n";
  }

  void func3(MLIRContext* context) {
    llvm::SmallVector<uint32_t> values = {1, 2, 3, 4};
    mlir::IntegerType integerType = IntegerType::get(context, 32);
    llvm::SmallVector<int64_t> shape = {2, 2};
    mlir::RankedTensorType rankedTensorType = RankedTensorType::get(shape, integerType);
    [[ maybe_unused ]] mlir::DenseElementsAttr denseElementsAttr = DenseElementsAttr::get<uint32_t>(rankedTensorType, values);
  
    llvm::outs() << rankedTensorType << "\n";
  }

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext* context = rootModule.getContext();
    func0(context);
    func1(context);
    func2(context);
    func3(context);
  }
};
} // namespace mlir::tt::ttir
