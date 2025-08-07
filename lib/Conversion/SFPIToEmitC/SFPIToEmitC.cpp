// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/SFPIToEmitC/SFPIToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/SFPI/IR/SFPI.h"
#include "ttmlir/Dialect/SFPI/IR/SFPIOps.h"

#include <string>

using namespace mlir;
using namespace mlir::tt;
using namespace mlir::tt::sfpi;

namespace mlir::tt {

#define GEN_PASS_DEF_CONVERTSFPITOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

/// Converts SFPI types to EmitC types for GCC builtin translation
class SFPIToEmitCTypeConverter : public TypeConverter {
public:
  SFPIToEmitCTypeConverter() {
    // Convert SFPI 4x8 vectors to __rvtt_vec_t
    addConversion([](Type type) -> std::optional<Type> {
      if (auto vectorType = dyn_cast<VectorType>(type)) {
        if (vectorType.getShape().size() == 2 &&
            vectorType.getShape()[0] == 4 && vectorType.getShape()[1] == 8 &&
            (vectorType.getElementType().isF32() ||
             vectorType.getElementType().isInteger(32))) {
          // Convert to __rvtt_vec_t (represented as opaque type)
          auto context = type.getContext();
          return emitc::OpaqueType::get(context, "sfpi::__rvtt_vec_t");
        }
      }
      return type; // Pass through other types unchanged
    });

    addConversion([](emitc::OpaqueType type) { return type; });
    addConversion([](IntegerType type) { return type; });
    addConversion([](FloatType type) { return type; });
    addConversion([](IndexType type) { return type; });

    // Add materialization patterns for unrealized conversion casts
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) {
        return nullptr;
      }
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) {
        return nullptr;
      }
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

using ValueOrString = std::variant<Value, std::string>;

//===----------------------------------------------------------------------===//
// Pattern Rewriters
//===----------------------------------------------------------------------===//

/// Base class for SFPI to EmitC conversion patterns
template <typename SFPIOpType, typename BuiltinOperands>
class SFPIToEmitCOpConversionPattern : public OpConversionPattern<SFPIOpType> {
public:
  using OpConversionPattern<SFPIOpType>::OpConversionPattern;

protected:
  StringRef getOpName(SFPIOpType op) const {
    StringRef name = op.getOperationName();
    assert(name.starts_with("sfpi."));
    return name.drop_front(5);
  }

  LogicalResult
  matchAndRewrite(SFPIOpType op, typename SFPIOpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = nullptr;
    if (op->getNumResults() > 0) {
      resultType =
          this->getTypeConverter()->convertType(op->getResult(0).getType());
      if (!resultType) {
        return failure();
      }
    }

    SmallVector<Value> operands;
    for (auto valueOrStr : BuiltinOperands::getBuiltinOperands(adaptor)) {
      if (auto *maybeValue = std::get_if<Value>(&valueOrStr); maybeValue) {
        operands.push_back(*maybeValue);
      } else if (auto *maybeStr = std::get_if<std::string>(&valueOrStr);
                 maybeStr) {
        auto literalAttribute = rewriter.create<emitc::LiteralOp>(
            op.getLoc(), rewriter.getI32Type(), *maybeStr);
        operands.push_back(literalAttribute->getResult(0));
      } else {
        llvm_unreachable("Unsupported builtin operand variant");
      }
    }

    if (resultType) {
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          op, resultType,
          rewriter.getStringAttr(Twine("__builtin_rvtt_sfp", getOpName(op))),
          nullptr, nullptr, operands);
    } else {
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          op, TypeRange{},
          rewriter.getStringAttr(Twine("__builtin_rvtt_sfp", getOpName(op))),
          nullptr, nullptr, operands);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Builtin Operand Getters
//===----------------------------------------------------------------------===//

static std::string addPrefix(llvm::StringRef prefix, llvm::StringRef input) {
  llvm::SmallVector<llvm::StringRef, 8> fields;
  input.split(fields, '|');
  std::string result;
  result.reserve(input.size() + (prefix.size() * fields.size()));
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i > 0) {
      result += "|";
    }
    result += prefix.str();
    result += fields[i].str();
  }
  return result;
}

struct AddOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename AddOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLhs(),
        adaptor.getRhs(),
        addPrefix("sfpi::SFPIADD_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct MulOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename MulOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLhs(),
        adaptor.getRhs(),
        addPrefix("sfpi::SFPMAD_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct MadOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename MadOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getA(),
        adaptor.getB(),
        adaptor.getC(),
        addPrefix("sfpi::SFPMAD_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct MovOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename MovOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        addPrefix("sfpi::SFPMOV_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct LoadOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename LoadOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        addPrefix("sfpi::SFPLOAD_MOD0_", stringifyEnum(adaptor.getMod0())),
        addPrefix("sfpi::SFPLOAD_ADDR_MODE_", stringifyEnum(adaptor.getMode())),
        adaptor.getAddr(),
    };
  }
};

struct StoreOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename StoreOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getValue(),
        addPrefix("sfpi::SFPSTORE_MOD0_", stringifyEnum(adaptor.getMod0())),
        addPrefix("sfpi::SFPSTORE_ADDR_MODE_",
                  stringifyEnum(adaptor.getMode())),
        adaptor.getAddr(),
    };
  }
};

struct XLoadIOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XLoadIOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        addPrefix("sfpi::SFPXLOADI_MOD0_", stringifyEnum(adaptor.getMod0())),
        adaptor.getImm(),
    };
  }
};

struct DivP2OpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename DivP2Op::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getExp(), adaptor.getSrc(),
        "0", // mod1 parameter not present in MLIR op
    };
  }
};

struct XiAddIOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XiAddIOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        adaptor.getImm(),
        addPrefix("sfpi::SFPXIADD_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct XiAddVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XiAddVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getDst(),
        adaptor.getSrc(),
        addPrefix("sfpi::SFPXIADD_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct AndOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename AndOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLhs(),
        adaptor.getRhs(),
    };
  }
};

struct OrOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename OrOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLhs(),
        adaptor.getRhs(),
    };
  }
};

struct XorOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XorOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLhs(),
        adaptor.getRhs(),
    };
  }
};

struct NotOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename NotOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
    };
  }
};

struct SetExpIOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename SetExpIOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getImm(),
        adaptor.getSrc(),
    };
  }
};

struct SetExpVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename SetExpVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getImm(),
        adaptor.getSrc(),
    };
  }
};

struct SetManIOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename SetManIOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getImm(),
        adaptor.getSrc(),
        "0",
    };
  }
};

struct SetManVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename SetManVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        adaptor.getMan(),
    };
  }
};

struct SetSgnIOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename SetSgnIOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getImm(),
        adaptor.getSrc(),
    };
  }
};

struct SetSgnVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename SetSgnVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        adaptor.getSgn(),
    };
  }
};

struct ExExpOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename ExExpOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        addPrefix("sfpi::SFPEXEXP_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct ExManOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename ExManOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        addPrefix("sfpi::SFPEXMAN_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct AbsOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename AbsOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        addPrefix("sfpi::SFPABS_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct LzOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename LzOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        addPrefix("sfpi::SFPLZ_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct XfCmpSOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XfCmpSOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLhs(),
        adaptor.getRhs(),
        addPrefix("sfpi::SFPXSCMP_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct XfCmpVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XfCmpVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLhs(),
        adaptor.getRhs(),
        addPrefix("sfpi::SFPXCMP_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct XiCmpSOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XiCmpSOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLhs(),
        adaptor.getRhs(),
        addPrefix("sfpi::SFPXSCMP_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct XiCmpVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XiCmpVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLhs(),
        adaptor.getRhs(),
        addPrefix("sfpi::SFPXCMP_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct CastOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename CastOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        addPrefix("sfpi::SFPCAST_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct ShftIOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename ShftIOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getDst(),
        adaptor.getImm(),
    };
  }
};

struct ShftVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename ShftVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getDst(),
        adaptor.getSrc(),
    };
  }
};

struct Shft2IOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename Shft2IOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getDst(),
        adaptor.getImm(),
    };
  }
};

struct Shft2VOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename Shft2VOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getDst(),
        adaptor.getSrc(),
    };
  }
};

struct Shft2GOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename Shft2GOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getL0(),
        adaptor.getL1(),
        adaptor.getL2(),
        adaptor.getL3(),
        addPrefix("sfpi::SFPSHFT2_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct Shft2GEOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename Shft2GEOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(), adaptor.getL0(), adaptor.getL1(),
        adaptor.getL2(),  adaptor.getL3(),
    };
  }
};

struct Shft2EOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename Shft2EOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        addPrefix("sfpi::SFPSHFT2_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct StochRndIOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename StochRndIOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        addPrefix("sfpi::SFPSTOCHRND_RND_", stringifyEnum(adaptor.getMode())),
        adaptor.getImm8(),
        adaptor.getSrcc(),
        addPrefix("sfpi::SFPSTOCHRND_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct StochRndVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename StochRndVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        addPrefix("sfpi::SFPSTOCHRND_RND_", stringifyEnum(adaptor.getMode())),
        adaptor.getImm8(),
        adaptor.getSrcc(),
        addPrefix("sfpi::SFPSTOCHRND_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct LutOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename LutOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getL0(),
        adaptor.getL1(),
        adaptor.getL2(),
        adaptor.getL3(),
        addPrefix("sfpi::SFPLUT_MOD0_", stringifyEnum(adaptor.getMod0())),
    };
  }
};

struct LutFp32_3ROpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename LutFp32_3ROp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getL0(),
        adaptor.getL1(),
        adaptor.getL2(),
        adaptor.getL3(),
        addPrefix("sfpi::SFPLUTFP32_MOD0_", stringifyEnum(adaptor.getMod0())),
    };
  }
};

struct LutFp32_6ROpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename LutFp32_6ROp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getL0(),
        adaptor.getL1(),
        adaptor.getL2(),
        adaptor.getL4(),
        adaptor.getL5(),
        adaptor.getL6(),
        adaptor.getL3(),
        addPrefix("sfpi::SFPLUTFP32_MOD0_", stringifyEnum(adaptor.getMod0())),
    };
  }
};

struct SwapOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename SwapOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getDst(),
        adaptor.getSrc(),
        addPrefix("sfpi::SFPSWAP_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct TranspOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename TranspOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getL0(),
        adaptor.getL1(),
        adaptor.getL2(),
        adaptor.getL3(),
    };
  }
};

struct SetCCIOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename SetCCIOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getImm(),
        addPrefix("sfpi::SFPSETCC_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct SetCCVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename SetCCVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getSrc(),
        addPrefix("sfpi::SFPSETCC_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct EnCCOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename EnCCOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getImm(),
        addPrefix("sfpi::SFPSETCC_MOD1_", stringifyEnum(adaptor.getMod1())),
    };
  }
};

struct PushCOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename PushCOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{};
  }
};

struct PopCOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename PopCOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{};
  }
};

struct CompCOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename CompCOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{};
  }
};

struct XBoolOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XBoolOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getT(),
        adaptor.getA(),
        adaptor.getB(),
    };
  }
};

struct XCondBOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XCondBOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getS(),
        adaptor.getI(),
    };
  }
};

struct XCondIOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XCondIOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getI(),
    };
  }
};

struct XVifOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename XVifOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{};
  }
};

struct AssignLRegOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename AssignLRegOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getLr(),
    };
  }
};

struct AssignLvOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename AssignLvOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getV(),
        adaptor.getIn(),
    };
  }
};

struct PreserveLRegOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename PreserveLRegOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getX(),
        adaptor.getN(),
    };
  }
};

struct ConfigVOpBuiltinOperands {
  static SmallVector<ValueOrString>
  getBuiltinOperands(typename ConfigVOp::Adaptor adaptor) {
    return std::initializer_list<ValueOrString>{
        adaptor.getValue(),
        addPrefix("sfpi::SFPCONFIG_DEST_", stringifyEnum(adaptor.getDest())),
    };
  }
};

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

struct ConvertSFPIToEmitCPass
    : public ::mlir::tt::impl::ConvertSFPIToEmitCBase<ConvertSFPIToEmitCPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    SFPIToEmitCTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    // Target is legal if it doesn't contain SFPI operations
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<sfpi::SFPIDialect>();

    // Function dialect handling
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    // Populate conversion patterns
    populateSFPIToEmitCConversionPatterns(patterns, typeConverter);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//

void mlir::tt::populateSFPIToEmitCConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<
      SFPIToEmitCOpConversionPattern<LoadOp, LoadOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<StoreOp, StoreOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XLoadIOp, XLoadIOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<MovOp, MovOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<AddOp, AddOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<MulOp, MulOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<MadOp, MadOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<DivP2Op, DivP2OpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XiAddIOp, XiAddIOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XiAddVOp, XiAddVOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<AndOp, AndOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<OrOp, OrOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XorOp, XorOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<NotOp, NotOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<SetExpIOp, SetExpIOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<SetExpVOp, SetExpVOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<SetManIOp, SetManIOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<SetManVOp, SetManVOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<SetSgnIOp, SetSgnIOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<SetSgnVOp, SetSgnVOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<ExExpOp, ExExpOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<ExManOp, ExManOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<AbsOp, AbsOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<LzOp, LzOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XfCmpSOp, XfCmpSOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XfCmpVOp, XfCmpVOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XiCmpSOp, XiCmpSOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XiCmpVOp, XiCmpVOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<CastOp, CastOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<ShftIOp, ShftIOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<ShftVOp, ShftVOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<Shft2IOp, Shft2IOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<Shft2VOp, Shft2VOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<Shft2GOp, Shft2GOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<Shft2GEOp, Shft2GEOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<Shft2EOp, Shft2EOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<StochRndIOp, StochRndIOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<StochRndVOp, StochRndVOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<LutOp, LutOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<LutFp32_3ROp, LutFp32_3ROpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<LutFp32_6ROp, LutFp32_6ROpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<SwapOp, SwapOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<TranspOp, TranspOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<SetCCIOp, SetCCIOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<SetCCVOp, SetCCVOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<EnCCOp, EnCCOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<PushCOp, PushCOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<PopCOp, PopCOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<CompCOp, CompCOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XBoolOp, XBoolOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XCondBOp, XCondBOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XCondIOp, XCondIOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<XVifOp, XVifOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<AssignLRegOp, AssignLRegOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<AssignLvOp, AssignLvOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<PreserveLRegOp,
                                     PreserveLRegOpBuiltinOperands>,
      SFPIToEmitCOpConversionPattern<ConfigVOp, ConfigVOpBuiltinOperands>>(
      typeConverter, patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tt::createConvertSFPIToEmitCPass() {
  return std::make_unique<ConvertSFPIToEmitCPass>();
}
