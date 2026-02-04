// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

// Helper function to extract scalar attribute from constant-like ops
// (FullOp, ZerosOp, OnesOp, ConstantOp with splat).
std::optional<Attribute> getScalarFromConstantOp(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op) {
    return std::nullopt;
  }

  auto tensorType = mlir::cast<RankedTensorType>(value.getType());
  Type elemType = tensorType.getElementType();

  if (isa<mlir::tt::ttir::ZerosOp>(op)) {
    if (elemType.isIntOrIndex()) {
      return IntegerAttr::get(elemType, 0);
    }
    return FloatAttr::get(elemType, 0.0);
  }

  if (isa<mlir::tt::ttir::OnesOp>(op)) {
    if (elemType.isIntOrIndex()) {
      return IntegerAttr::get(elemType, 1);
    }
    return FloatAttr::get(elemType, 1.0);
  }

  if (auto fullOp = dyn_cast<mlir::tt::ttir::FullOp>(op)) {
    return fullOp.getFillValue();
  }

  if (auto constantOp = dyn_cast<mlir::tt::ttir::ConstantOp>(op)) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue())) {
      if (denseAttr.isSplat()) {
        return denseAttr.getSplatValue<Attribute>();
      }
    }
  }

  return std::nullopt;
}

// Interface-based pattern for converting binary ops to scalar form when one
// operand is a constant splat. Applies to any op implementing
// BinaryTensorToScalarOpConversionInterface.
class BinaryToScalarCanonicalizationPattern
    : public mlir::OpInterfaceRewritePattern<
          mlir::tt::ttir::BinaryTensorToScalarOpConversionInterface> {
public:
  using mlir::OpInterfaceRewritePattern<
      mlir::tt::ttir::BinaryTensorToScalarOpConversionInterface>::
      OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttir::BinaryTensorToScalarOpConversionInterface op,
                  mlir::PatternRewriter &rewriter) const override {
    // Get operands from the underlying operation
    Operation *operation = op.getOperation();
    if (operation->getNumOperands() < 2) {
      return mlir::failure();
    }

    Value lhs = operation->getOperand(0);
    Value rhs = operation->getOperand(1);

    // Try RHS as scalar first
    if (auto scalar = getScalarFromConstantOp(rhs)) {
      Operation *scalarOp =
          op.createScalarOp(rewriter, lhs, *scalar, /*scalarOnLhs=*/false);
      if (scalarOp) {
        rewriter.replaceOp(operation, scalarOp->getResults());
        return mlir::success();
      }
    }

    // Try LHS as scalar
    if (auto scalar = getScalarFromConstantOp(lhs)) {
      Operation *scalarOp =
          op.createScalarOp(rewriter, rhs, *scalar, /*scalarOnLhs=*/true);
      if (scalarOp) {
        rewriter.replaceOp(operation, scalarOp->getResults());
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

} // namespace
using namespace mlir::tt::ttir;

namespace mlir::tt::ttir {
static void printDimensionList(::mlir::AsmPrinter &printer,
                               ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

static ::mlir::ParseResult
parseDimensionList(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}

template <typename... Args>
static void printVargDimensionList(mlir::AsmPrinter &printer, Args &&...dims) {
  printDimensionList(printer,
                     llvm::SmallVector<int64_t>({std::forward<Args>(dims)...}));
}

template <typename... Args>
static mlir::ParseResult parseVargDimensionList(mlir::AsmParser &odsParser,
                                                Args &...dims) {
  llvm::SmallVector<int64_t> dimensions;
  mlir::ParseResult result = parseDimensionList(odsParser, dimensions);
  if (succeeded(result)) {
    llvm::SmallVector<std::tuple_element_t<0, std::tuple<Args...>> *> copy(
        {&dims...});
    assert(dimensions.size() == sizeof...(dims));
    for (size_t i = 0; i < dimensions.size(); ++i) {
      *copy[i] = dimensions[i];
    }
  }
  return result;
}
} // namespace mlir::tt::ttir

#include "ttmlir/Dialect/TTIR/IR/TTIROpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.cpp.inc"

// This DialectInlinerInterface is nearly identical to the one found in
// mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp. We need
// to define one for the TTIRDialect as well since the IR uses
// FuncDialect for function definitions/calls, and TTIR for ops.
// We need to legalize inlining for all TTIR ops.
struct TTIRInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Everything can be inlined
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // This should only ever be a func::ReturnOp
    auto returnOp = cast<func::ReturnOp>(op);

    // Replace usages of the functions result with the result operands of the
    // return op
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }
};

#include "ttmlir/Dialect/TTIR/IR/TTIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TTIR dialect.
//===----------------------------------------------------------------------===//

void TTIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTIR/IR/TTIROps.cpp.inc"
      >();
  addInterfaces<TTIRInlinerInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.cpp.inc"
      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// TTIR constant materializer.
//===----------------------------------------------------------------------===//

::mlir::Operation *TTIRDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  if (auto elementsAttr = mlir::dyn_cast<mlir::ElementsAttr>(value)) {
    return builder.create<ttir::ConstantOp>(loc, type, elementsAttr);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TTIR dialect canonicalization patterns.
//===----------------------------------------------------------------------===//

void TTIRDialect::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns) const {
  // Register interface-based canonicalization pattern for binary ops that
  // support conversion to scalar form.
  patterns.add<BinaryToScalarCanonicalizationPattern>(getContext());
}
