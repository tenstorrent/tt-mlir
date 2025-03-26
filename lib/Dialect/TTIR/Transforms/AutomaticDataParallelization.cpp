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
    mlir::DenseElementsAttr denseElementsAttr = DenseElementsAttr::get(context);
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

  void runOnOperation() final {
    [[ maybe_unused ]] ArrayRef<int64_t> copiedMeshShape = *meshShape;
    llvm::outs() << "(" << copiedMeshShape[0] << "," << copiedMeshShape[1] << ")\n";
    // need to assert that dim0 of mesh shape == 1 and dim1 is the actual dim that we want to do batch parallelization on

    mlir::ModuleOp rootModule = getOperation();
    OpBuilder builder(rootModule.getContext());
    mlir::PatternRewriter rewriter(rootModule.getContext());

    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        Block &entryBlock = funcOp.getBody().front();

        // parse arguments and add relevant mesh shard operations
        for (BlockArgument arg : entryBlock.getArguments()) {
          if (arg.use_empty()) {
              continue;
          }

          Operation *firstUser = nullptr;
          for (auto &use : arg.getUses()) {
            Operation *userOp = use.getOwner();
            if (!firstUser || userOp->isBeforeInBlock(firstUser)) {
              firstUser = userOp;
            }
          }

          if (firstUser) {
            builder.setInsertionPoint(firstUser);

            DictionaryAttr attrDict = funcOp.getArgAttrDict(arg.getArgNumber());
            auto argumentTypeAttr = dyn_cast<ArgumentTypeAttr>(attrDict.get(ArgumentTypeAttr::name));

            if (argumentTypeAttr.getValue() != ArgumentType::Input) {
              auto loc = arg.getLoc();
              auto outputType = dyn_cast<RankedTensorType>(arg.getType());
              auto outputTensor = builder.create<tensor::EmptyOp>(
                loc, 
                outputType.getShape(), 
                outputType.getElementType());
              auto newResult = builder.create<ttir::MeshShardOp>(
                loc, 
                outputType, 
                arg, 
                outputTensor, 
                mlir::tt::MeshShardType::Replicate,
                mlir::tt::MeshShardDirection::FullToShard, 
                /*shard_shape*/ llvm::SmallVector<int64_t>{1}, 
                /*shard_dims*/ llvm::SmallVector<int64_t>{-1});
            
              for (auto &use : arg.getUses()) {
                Operation *userOp = use.getOwner();
            
                if (userOp != newResult) {
                  userOp->replaceUsesOfWith(arg, newResult);
                } 
              }
            }
            else {
              auto loc = arg.getLoc();
              auto outputType = dyn_cast<RankedTensorType>(arg.getType());
              llvm::SmallVector<int64_t> newShape(outputType.getShape().begin(), outputType.getShape().end());
              newShape[0] = outputType.getShape()[0] / copiedMeshShape[1];
              auto newOutputType = RankedTensorType::get(newShape, outputType.getElementType());
              auto outputTensor = builder.create<tensor::EmptyOp>(
                loc, 
                newOutputType.getShape(), 
                newOutputType.getElementType());
              auto newResult = builder.create<ttir::MeshShardOp>(
                loc, 
                newOutputType, 
                arg, 
                outputTensor, 
                mlir::tt::MeshShardType::Devices,
                mlir::tt::MeshShardDirection::FullToShard, 
                /*shard_shape*/ llvm::SmallVector<int64_t>{copiedMeshShape[1], 1, 1, 1}, 
                /*shard_dims*/ llvm::SmallVector<int64_t>{-1, 0});
            
              for (auto &use : arg.getUses()) {
                Operation *userOp = use.getOwner();
            
                if (userOp != newResult) {
                  userOp->replaceUsesOfWith(arg, newResult);
                } 
              }
            }
          }
        }

        // parse outputs and add relevant mesh shard operations
        for (auto &op : entryBlock) {
          if (auto returnOp = llvm::dyn_cast<func::ReturnOp>(&op)) {
            auto returnTensors = returnOp.getOperands();

            // need to assert returnTensors is of size 1

            for (auto tensor : returnTensors) {
              auto producerOp = tensor.getDefiningOp();

              builder.setInsertionPoint(returnOp);
              auto loc = tensor.getLoc();
              auto outputType = dyn_cast<RankedTensorType>(tensor.getType());
              auto outputTensor = builder.create<tensor::EmptyOp>(
                loc, 
                outputType.getShape(), 
                outputType.getElementType());
              auto newResult = builder.create<ttir::MeshShardOp>(
                loc, 
                outputType, 
                producerOp->getResult(0), 
                outputTensor, 
                mlir::tt::MeshShardType::Replicate,
                mlir::tt::MeshShardDirection::ShardToFull, 
                /*shard_shape*/ llvm::SmallVector<int64_t>{1}, 
                /*shard_dims*/ llvm::SmallVector<int64_t>{-1});

                for (auto &use : tensor.getUses()) {
                  Operation *userOp = use.getOwner();
    
                  if (userOp != newResult) {
                    userOp->replaceUsesOfWith(tensor, newResult);
                  } 
                }
            }
          }
        }

        // once all the mesh shards have been inserted correctly, we now want to propogate all the new shapes correctly
        // find all the mesh shards in the graph, find their dependent operations, and compute the new shapes for their dependent operations
        for (auto &op : entryBlock) {
          if (auto meshShardOp = llvm::dyn_cast<ttir::MeshShardOp>(&op)) {
            auto opResult = meshShardOp.getResult();
            for (auto resultOperation : opResult.getUsers()) {
              llvm::outs() << "taps was here";

              // replace current op with a new op that has an updated output shape
              if (isa<ttir::AddOp>(resultOperation)) {
                auto addOp = dyn_cast<ttir::AddOp>(resultOperation);
                auto outputType = dyn_cast<RankedTensorType>(addOp->getResult(0).getType());
                auto inputType = dyn_cast<RankedTensorType>(addOp->getOperand(0).getType());
                llvm::SmallVector<int64_t> newShape(inputType.getShape().begin(), inputType.getShape().end());
                auto newOutputType = RankedTensorType::get(newShape, outputType.getElementType());
                rewriter.replaceOpWithNewOp<ttir::AddOp>(addOp, newOutputType, addOp->getOperands());
              }
              else if (isa<ttir::MatmulOp>(resultOperation)) {
                continue;
              }
              else {
                signalPassFailure();
              }



              // rewriter.replaceOp(op, newOp.getOperation());
            }
          }
        }
      }
    }
  }
};
} // namespace mlir::tt::ttir

/*
- for each argument, figure out what type it is
  - input, parameter, constant
- for parameter + constant
  - we are going to replicate this tensor across all devices (so nothing changes)
- for input
  - we are going to split it's batch dimension across all devices
  - we need to calculate it's new output shape based on the split batch
  - we need to propogate this new shape to all places that use this input
*/

/*
  %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #tt.shard_type<devices>}> : (tensor<64x1x1024x2048xf32>, tensor<32x1x1024x2048xf32>) -> tensor<32x1x1024x2048xf32>

*/

mlir::Type
- mlir::BFloat16Type
- mlir::Float16Type
- mlir::Float32Type
- mlir::Float64Type
- mlir::Float80Type
- mlir::Float128Type
- mlir::FloatTF32Type
- mlir::IntegerType
- mlir::RankedTensorType
- mlir::ShapedType

/*
mlir::Type
- TypeID 	getTypeID ()
- MLIRContext * 	getContext () const
- Dialect & 	getDialect () const
- bool 	isBF16 () const
- bool 	isF16 () const
- bool 	isTF32 () const
- bool 	isF32 () const
- bool 	isF64 () const
- bool 	isF80 () const
- bool 	isF128 () const
- bool 	isFloat () const
- bool 	isInteger () const
- bool 	isSignlessInteger () const
- bool 	isSignedInteger () const
- bool 	isUnsignedInteger () const
- void 	print (raw_ostream &os) const

template<typename... Tys>
- bool 	isa () const

template<typename U >
- U 	dyn_cast () const

template<typename U >
- U 	dyn_cast_or_null () const

template<typename U >
- U 	cast () const
*/

/*
mlir::BFloat16Type
mlir::Float16Type
mlir::Float16Type
mlir::Float32Type
mlir::Float64Type
mlir::Float80Type
mlir::Float128Type
mlir::FloatTF32Type
- BFloat16Type BFloat16Type::get(MLIRContext *)
*/

/*
mlir::IntegerType
- IntegerType IntegerType::get(MLIRContext *, unsigned, IntegerType::SignednessSemantics)
- unsigned getWidth()
- SignednessSemantics getSignedness()
*/

/*
mlir::RankedTensorType
- RankedTensorType RankedTensorType::get(MLIRContext *, ::llvm::ArrayRef<int64_t>, Type)
- RankedTensorType RankedTensorType::get(MLIRContext *, ::llvm::ArrayRef<int64_t>, Type, Attribute)
- ::llvm::ArrayRef<int64_t> getShape()
- Type getElementType()
- Attribute getEncoding()
*/


/*
mlir::ShapedType
class ShapedType : public Type
- Type getElementType() const;
- int64_t getNumElements() const;
- int64_t getRank() const;
- bool hasRank() const;
- ArrayRef<int64_t> getShape() const;

class TensorType : public ShapedType
class RankedTensorType : public Type::TypeBase<RankedTensorType, TensorType, detail::RankedTensorTypeStorage>
class UnrankedTensorType : public Type::TypeBase<UnrankedTensorType, TensorType, detail::UnrankedTensorTypeStorage>
*/

mlir::Attribute
- DictionaryAttr
- DenseElementsAttr
- ArrayAttr
- DenseArrayAttr
  - DenseI8ArrayAttr
  - DenseI16ArrayAttr
  - DenseI32ArrayAttr
  - DenseI64ArrayAttr
  - DenseF32ArrayAttr
  - DenseF64ArrayAttr

/*
mlir::Attribute
- TypeID 	getTypeID ()
- MLIRContext * 	getContext () const
- Dialect & 	getDialect () const
- void 	print (raw_ostream &os) const

template<typename... Tys>
- bool 	isa () const

template<typename U >
- U 	dyn_cast () const

template<typename U >
- U 	dyn_cast_or_null () const

template<typename U >
- U 	cast () const
*/

/*
DictionaryAttr
- static DictionaryAttr::get(ArrayRef<NamedAttribute> value, MLIRContext *context)
- ArrayRef< NamedAttribute > 	getValue () const
- Attribute get(StringRef name) const
- bool empty() const
- size_t size () const
- iterator 	begin () const
- iterator 	end () const
*/

/*
DenseElementsAttr
static DenseElementsAttr::get(ShapedType type, ArrayRef<Attribute> values)
*/

/*
mlir::NamedAttribute
NamedAttribute (StringAttr name, Attribute value)
- StringAttr 	getName () const
- Attribute 	getValue () const
- void 	setName (StringAttr newName)
- void 	setValue (Attribute newValue)
*/

/*
mlir::Builder
- MLIRContext * 	getContext () const
- Location 	getUnknownLoc ()
- FloatType 	getBF16Type ()
- FloatType 	getF16Type ()
- FloatType 	getTF32Type ()
- FloatType 	getF32Type ()
- FloatType 	getF64Type ()
- FloatType 	getF80Type ()
- FloatType 	getF128Type ()
- IntegerType 	getI1Type ()
- IntegerType 	getI2Type ()
- IntegerType 	getI4Type ()
- IntegerType 	getI8Type ()
- IntegerType 	getI16Type ()
- IntegerType 	getI32Type ()
- IntegerType 	getI64Type ()

- UnitAttr 	getUnitAttr ()
- BoolAttr 	getBoolAttr (bool value)
- DictionaryAttr 	getDictionaryAttr (ArrayRef< NamedAttribute > value)
- IntegerAttr 	getIntegerAttr (Type type, int64_t value)
- IntegerAttr 	getIntegerAttr (Type type, const APInt &value)
- FloatAttr 	getFloatAttr (Type type, double value)
- FloatAttr 	getFloatAttr (Type type, const APFloat &value)
- StringAttr 	getStringAttr (const Twine &bytes)
- ArrayAttr 	getArrayAttr (ArrayRef< Attribute > value)
- TypedAttr 	getZeroAttr (Type type)
- TypedAttr 	getOneAttr (Type type)
- FloatAttr 	getF16FloatAttr (float value)
- FloatAttr 	getF32FloatAttr (float value)
- FloatAttr 	getF64FloatAttr (double value)
- IntegerAttr 	getI8IntegerAttr (int8_t value)
- IntegerAttr 	getI16IntegerAttr (int16_t value)
- IntegerAttr 	getI32IntegerAttr (int32_t value)
- IntegerAttr 	getI64IntegerAttr (int64_t value)
- IntegerAttr 	getIndexAttr (int64_t value)
- IntegerAttr 	getSI32IntegerAttr (int32_t value)
- IntegerAttr 	getUI32IntegerAttr (uint32_t value)
- DenseIntElementsAttr 	getBoolVectorAttr (ArrayRef< bool > values)
- DenseIntElementsAttr 	getI32VectorAttr (ArrayRef< int32_t > values)
- DenseIntElementsAttr 	getI64VectorAttr (ArrayRef< int64_t > values)
- DenseIntElementsAttr 	getIndexVectorAttr (ArrayRef< int64_t > values)
- DenseFPElementsAttr 	getF32VectorAttr (ArrayRef< float > values)
- DenseFPElementsAttr 	getF64VectorAttr (ArrayRef< double > values)
- DenseIntElementsAttr 	getI32TensorAttr (ArrayRef< int32_t > values)
- DenseIntElementsAttr 	getI64TensorAttr (ArrayRef< int64_t > values)
- DenseIntElementsAttr 	getIndexTensorAttr (ArrayRef< int64_t > values)
- DenseBoolArrayAttr 	getDenseBoolArrayAttr (ArrayRef< bool > values)
- DenseI8ArrayAttr 	getDenseI8ArrayAttr (ArrayRef< int8_t > values)
- DenseI16ArrayAttr 	getDenseI16ArrayAttr (ArrayRef< int16_t > values)
- DenseI32ArrayAttr 	getDenseI32ArrayAttr (ArrayRef< int32_t > values)
- DenseI64ArrayAttr 	getDenseI64ArrayAttr (ArrayRef< int64_t > values)
- DenseF32ArrayAttr 	getDenseF32ArrayAttr (ArrayRef< float > values)
- DenseF64ArrayAttr 	getDenseF64ArrayAttr (ArrayRef< double > values)
- ArrayAttr 	getAffineMapArrayAttr (ArrayRef< AffineMap > values)
- ArrayAttr 	getBoolArrayAttr (ArrayRef< bool > values)
- ArrayAttr 	getI32ArrayAttr (ArrayRef< int32_t > values)
- ArrayAttr 	getI64ArrayAttr (ArrayRef< int64_t > values)
- ArrayAttr 	getIndexArrayAttr (ArrayRef< int64_t > values)
- ArrayAttr 	getF32ArrayAttr (ArrayRef< float > values)
- ArrayAttr 	getF64ArrayAttr (ArrayRef< double > values)
- ArrayAttr 	getStrArrayAttr (ArrayRef< StringRef > values)
- ArrayAttr 	getTypeArrayAttr (TypeRange values)
- NamedAttribute 	getNamedAttr (StringRef name, Attribute val)

template<typename Ty , typename... Args>
- Ty 	getType (Args &&...args)

template<typename Attr , typename... Args>
- Attr 	getAttr (Args &&...args)
*/

/*
mlir::OpBuilder
- void 	clearInsertionPoint ()
- void 	setInsertionPoint (Block *block, Block::iterator insertPoint)
- void 	setInsertionPoint (Operation *op)
- Operation * 	clone (Operation &op, IRMapping &mapper)
- Operation * 	clone (Operation &op)

template<typename OpTy , typename... Args>
- OpTy 	create (Location location, Args &&...args)
*/

/*

*/


mlir::Builder
- mlir::OpBuilder
  - mlir::RewriterBase
    - mlir::IRRewriter
    - mlir::PatternRewriter
      - mlir::ConversionPatternRewriter
    - mlir::transform::TransformRewriter



Type
Value
Attr



// operation
module {
  //region
    //block
      // operation
        //region
          // block (2 arguments)
            // operation
              func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
                %0 = tensor.empty() : tensor<64x96xbf16>
                // CHECK: "ttnn.matmul"
                %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
                return %1 : tensor<64x96xbf16>
              }
      // operation
        // region
          // block (2 arguments)
            // operation
              func.func @matmul_transpose_lhs(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<128x128xbf16> {
                %0 = tensor.empty() : tensor<128x128xbf16>
                // CHECK: "ttnn.matmul"
                %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
                return %1 : tensor<128x128xbf16>
              }
}



module {
  func.func @forward(%arg0: tensor<64x128xbf16> {tt.argument_type = #tt.argument_type<input>}, %arg1: tensor<128x96xbf16> {tt.argument_type = #tt.argument_type<parameter>}) -> tensor<64x96xbf16> {
    %0 = tensor.empty() : tensor<64x96xbf16>
    %1 = "ttir.mesh_shard"(%arg0, %arg0) <{shard_dims = array<i64: -1>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #tt.shard_type<identity>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %2 = "ttir.mesh_shard"(%arg1, %arg1) <{shard_dims = array<i64: -1>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #tt.shard_type<identity>}> : (tensor<128x96xbf16>, tensor<128x96xbf16>) -> tensor<128x96xbf16>
    %3 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    return %3 : tensor<64x96xbf16>
  }
}

/*
- get mesh shape
- get root module
- iterate through all func.funcop's 
- for each func, iterate through all it's arguments
- for each argument, find where it is being used first, and insert a mesh_shard operation right before it's used
*/

/*
module {
  func.func @forward(%arg0: tensor<64x128xbf16> {tt.argument_type = #tt.argument_type<input>}, %arg1: tensor<128x96xbf16> {tt.argument_type = #tt.argument_type<parameter>}) -> tensor<64x96xbf16> {
    %0 = tensor.empty() : tensor<64x96xbf16>
    %1 = "ttir.constant"() <{value = dense<5> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "ttir.constant"() <{value = dense<5> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    return %3 : tensor<64x96xbf16>
  }
}
*/

/*
module {
  func.func @forward(%arg0: tensor<64x128xbf16> {tt.argument_type = #tt.argument_type<input>}, %arg1: tensor<128x96xbf16> {tt.argument_type = #tt.argument_type<parameter>}) -> tensor<64x96xbf16> {
    %0 = tensor.empty() : tensor<64x96xbf16>
    %1 = tensor.empty() : tensor<64x128xbf16>
    %2 = "ttir.mesh_shard"(%arg0, %1) <{shard_dims = array<i64: -1>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #tt.shard_type<identity>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %3 = tensor.empty() : tensor<128x96xbf16>
    %4 = "ttir.mesh_shard"(%arg1, %3) <{shard_dims = array<i64: -1>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #tt.shard_type<identity>}> : (tensor<128x96xbf16>, tensor<128x96xbf16>) -> tensor<128x96xbf16>
    %5 = "ttir.matmul"(%2, %4, %0) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    return %5 : tensor<64x96xbf16>
  }
}
*/

/*
static void build(::mlir::OpBuilder &odsBuilder, 
::mlir::OperationState &odsState, 
::mlir::Type result, 
::mlir::Value input, 
::mlir::Value output, 
::mlir::tt::MeshShardTypeAttr shard_type, 
::mlir::tt::MeshShardDirectionAttr shard_direction, 
::mlir::DenseI64ArrayAttr shard_shape, 
::mlir::DenseI64ArrayAttr shard_dims);

ttmlir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::MeshShardOp>(
  rewriter, srcOp, outputType, inputOperand,
  meshSharding.getShardType(),
  mlir::tt::MeshShardDirection::ShardToFull,
  meshSharding.getShardShape(), meshSharding.getShardDims());


let arguments = (ins AnyRankedTensor:$input,
  AnyRankedTensor:$output,
  TT_MeshShardTypeAttr:$shard_type,
  TT_MeshShardDirectionAttr:$shard_direction,
  DenseI64ArrayAttr:$shard_shape,
  DenseI64ArrayAttr:$shard_dims)
*/

// insert some dummy op for now
/*
uint32_t someValue = 5;
Type elementType = IntegerType::get(context, 32);
std::vector<APInt> values = { APInt(32, someValue) };
DenseElementsAttr denseAttr = DenseElementsAttr::get(RankedTensorType::get({1}, elementType), values);
Type tensorType = RankedTensorType::get({1}, elementType);
builder.create<ttir::ConstantOp>(loc, tensorType, denseAttr);
*/

/*
class OperandCountAnalysis {
public:
  OperandCountAnalysis(Operation *op, AnalysisManager &manager) :op(op) {
    operandCount = op->getNumOperands();
  }

  size_t getOperandCount() const { return operandCount; }

private:
  Operation *op;
  size_t operandCount;
}

class OperandCountAnalysisManager : public AnalysisManager {
public:
  OperandCountAnalysis &getOperandCountAnalysis(Operation *op) {
    if (operandCountCache.find(op) == operandCountCache.end()) {
      operandCountCache[op] = std::make_unique<OperandCountAnalysis>(op, *this);
    }

    return *operandCountCache[op];
  }

private:
  llvm::DenseMap<Operation *, std::unique_ptr<OperandCountAnalysis>> operandCountCache;
}
*/



