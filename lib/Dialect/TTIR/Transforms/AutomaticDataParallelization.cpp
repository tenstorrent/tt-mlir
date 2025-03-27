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

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
      registry.insert<mlir::func::FuncDialect>();
      registry.insert<mlir::tt::ttir::TTIRDialect>();
      registry.insert<::mlir::tensor::TensorDialect>();
  }

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

  /*
  module {
    module_1 {
      func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
        %0 = tensor.empty() : tensor<64x96xbf16>
        // CHECK: "ttnn.matmul"
        %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
        return %1 : tensor<64x96xbf16>
      }
    }

    module_2 {
      func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
        %0 = tensor.empty() : tensor<64x96xbf16>
        // CHECK: "ttnn.matmul"
        %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
        return %1 : tensor<64x96xbf16>
      }
    }
  }
  */

  void func4(mlir::ModuleOp& rootModule, MLIRContext* context) {
    mlir::OpBuilder builder(context);

    mlir::StringAttr module_1_name = mlir::StringAttr::get(context, "module_1");
    mlir::LocationAttr module_1_loc_attr = mlir::NameLoc::get(module_1_name, mlir::UnknownLoc::get(context));
    mlir::Location module_1_loc = Location(module_1_loc_attr);
    mlir::ModuleOp module_1 = mlir::ModuleOp::create(module_1_loc, "module_1");

    mlir::StringAttr module_2_name = mlir::StringAttr::get(context, "module_2");
    mlir::LocationAttr module_2_loc_attr = mlir::NameLoc::get(module_2_name, mlir::UnknownLoc::get(context));
    mlir::Location module_2_loc = Location(module_2_loc_attr);
    mlir::ModuleOp module_2 = mlir::ModuleOp::create(module_2_loc, "module_2");

    rootModule.push_back(module_1);
    rootModule.push_back(module_2);

    {
      mlir::RankedTensorType type_1 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));
      mlir::RankedTensorType type_2 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));
      mlir::RankedTensorType output_type = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));

      llvm::SmallVector<mlir::Type> inputs = {type_1, type_2};
      llvm::SmallVector<mlir::Type> outputs = {output_type};
      mlir::FunctionType function_type = mlir::FunctionType::get(context, inputs, outputs);

      mlir::NamedAttribute attribute_1 = mlir::NamedAttribute(mlir::StringAttr::get(context, "fruit"), mlir::StringAttr::get(context, "orange"));
      func::FuncOp module_1_func_1 = func::FuncOp::create(mlir::UnknownLoc::get(context), "func_1", function_type, {attribute_1});
      mlir::Block* module_1_func_1_block = module_1_func_1.addEntryBlock();

      mlir::Block &entryBlock = module_1_func_1.getBody().front();
      builder.setInsertionPointToStart(&entryBlock);

      mlir::RankedTensorType type_3 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));
      llvm::SmallVector<mlir::Type> types = {type_3};
      llvm::SmallVector<mlir::Value> values;
      tensor::EmptyOp emptyOp = builder.create<tensor::EmptyOp>(mlir::UnknownLoc::get(context), types, values);
      
      mlir::OperationState returnState(mlir::UnknownLoc::get(context), mlir::func::ReturnOp::getOperationName());
      returnState.addOperands(emptyOp.getResult());
      
      mlir::Operation *returnOp = mlir::Operation::create(returnState);
      module_1_func_1_block->push_back(returnOp);
      module_1.push_back(module_1_func_1);
      

    }

    {
      mlir::RankedTensorType type_1 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));
      mlir::RankedTensorType type_2 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));
      // mlir::RankedTensorType type_3 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));

      llvm::SmallVector<mlir::Type> inputs = {type_1, type_2};
      llvm::SmallVector<mlir::Type> outputs = {};
      mlir::FunctionType function_type = mlir::FunctionType::get(context, inputs, outputs);

      mlir::NamedAttribute attribute_1 = mlir::NamedAttribute(mlir::StringAttr::get(context, "fruit"), mlir::StringAttr::get(context, "orange"));
      func::FuncOp module_1_func_2 = func::FuncOp::create(mlir::UnknownLoc::get(context), "func_2", function_type, {attribute_1});
      mlir::Block* module_1_func_2_block = module_1_func_2.addEntryBlock();

      mlir::OperationState returnState(mlir::UnknownLoc::get(context), mlir::func::ReturnOp::getOperationName());
      mlir::Operation *returnOp = mlir::Operation::create(returnState);
      module_1_func_2_block->push_back(returnOp);
      module_1.push_back(module_1_func_2);
    }
    

    {
      mlir::RankedTensorType type_1 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));
      mlir::RankedTensorType type_2 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));
      // mlir::RankedTensorType type_3 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));

      llvm::SmallVector<mlir::Type> inputs = {type_1, type_2};
      llvm::SmallVector<mlir::Type> outputs = {};
      mlir::FunctionType function_type = mlir::FunctionType::get(context, inputs, outputs);

      mlir::NamedAttribute attribute_1 = mlir::NamedAttribute(mlir::StringAttr::get(context, "fruit"), mlir::StringAttr::get(context, "orange"));
      func::FuncOp module_2_func_1 = func::FuncOp::create(mlir::UnknownLoc::get(context), "func_1", function_type, {attribute_1});
      mlir::Block* module_2_func_1_block = module_2_func_1.addEntryBlock();

      mlir::OperationState returnState(mlir::UnknownLoc::get(context), mlir::func::ReturnOp::getOperationName());
      mlir::Operation *returnOp = mlir::Operation::create(returnState);
      module_2_func_1_block->push_back(returnOp);
      module_2.push_back(module_2_func_1);
    }

    {
      mlir::RankedTensorType type_1 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));
      mlir::RankedTensorType type_2 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));
      // mlir::RankedTensorType type_3 = mlir::RankedTensorType::get({1, 2}, Float32Type::get(context));

      llvm::SmallVector<mlir::Type> inputs = {type_1, type_2};
      llvm::SmallVector<mlir::Type> outputs = {};
      mlir::FunctionType function_type = mlir::FunctionType::get(context, inputs, outputs);

      mlir::NamedAttribute attribute_1 = mlir::NamedAttribute(mlir::StringAttr::get(context, "fruit"), mlir::StringAttr::get(context, "orange"));
      func::FuncOp module_2_func_2 = func::FuncOp::create(mlir::UnknownLoc::get(context), "func_2", function_type, {attribute_1});
      mlir::Block* module_2_func_2_block = module_2_func_2.addEntryBlock();

      mlir::OperationState returnState(mlir::UnknownLoc::get(context), mlir::func::ReturnOp::getOperationName());
      mlir::Operation *returnOp = mlir::Operation::create(returnState);
      module_2_func_2_block->push_back(returnOp);
      module_2.push_back(module_2_func_2);
    }
  }

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext* context = rootModule.getContext();
    //func0(context);
    //func1(context);
    //func2(context);
    //func3(context);
    func4(rootModule, context);
  }
};
} // namespace mlir::tt::ttir
