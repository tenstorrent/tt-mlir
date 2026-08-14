// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

// Ensure enum helpers (FieldParser, etc.) are visible before attrs
// The declarations live in D2MOps.h via D2MOpsEnums.h.inc; only include cpp
// here.
#include "ttmlir/Dialect/D2M/IR/D2MOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.cpp.inc"

using namespace mlir;
using namespace mlir::tt::d2m;
namespace ttcore = mlir::tt::ttcore;

// Custom assembly format for D2M_ThreadAttr.
//
// Format:  `<` threadType (`,` kernelSymbol)?
//               (`,` `dm_core` `=` dmCoreIndex)? `>`
//
// The two optional groups both start with `,`, so the declarative tablegen
// format cannot disambiguate them: it always tries to parse the kernel symbol
// after the first comma and fails on `#d2m.thread<datamovement, dm_core = 0>`.
// We peek for the `dm_core` keyword to pick the correct branch.
mlir::Attribute ThreadAttr::parse(::mlir::AsmParser &parser, ::mlir::Type) {
  if (parser.parseLess()) {
    return {};
  }

  ::mlir::FailureOr<ThreadType> threadType =
      ::mlir::FieldParser<ThreadType>::parse(parser);
  if (::mlir::failed(threadType)) {
    return {};
  }

  SymbolRefAttr kernelSymbol;
  int32_t dmCoreIndex = -1;

  // First optional: either `, @kernel` or `, dm_core = N`.
  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseOptionalKeyword("dm_core").succeeded()) {
      if (parser.parseEqual() || parser.parseInteger(dmCoreIndex)) {
        return {};
      }
    } else {
      if (parser.parseAttribute(kernelSymbol)) {
        return {};
      }
      // Second optional: only valid if a kernel symbol was given above.
      if (parser.parseOptionalComma().succeeded()) {
        if (parser.parseKeyword("dm_core") || parser.parseEqual() ||
            parser.parseInteger(dmCoreIndex)) {
          return {};
        }
      }
    }
  }

  if (parser.parseGreater()) {
    return {};
  }

  return ThreadAttr::get(parser.getContext(), *threadType, kernelSymbol,
                         dmCoreIndex);
}

void ThreadAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getThreadType());
  if (getKernelSymbol()) {
    printer << ", ";
    printer.printAttribute(getKernelSymbol());
  }
  if (getDmCoreIndex() != -1) {
    printer << ", dm_core = " << getDmCoreIndex();
  }
  printer << ">";
}

namespace {

int64_t coreCount(ttcore::CoreRangeAttr range) {
  ttcore::CoreCoordAttr start = range.getStartCoord();
  ttcore::CoreCoordAttr end = range.getEndCoord();
  return (end.getY() - start.getY() + 1) * (end.getX() - start.getX() + 1);
}

void appendCores(ttcore::CoreRangeAttr range,
                 SmallVectorImpl<std::pair<int64_t, int64_t>> &cores) {
  ttcore::CoreCoordAttr start = range.getStartCoord();
  ttcore::CoreCoordAttr end = range.getEndCoord();
  for (int64_t y = start.getY(); y <= end.getY(); ++y) {
    for (int64_t x = start.getX(); x <= end.getX(); ++x) {
      cores.emplace_back(y, x);
    }
  }
}

bool hasDuplicateCores(ArrayRef<std::pair<int64_t, int64_t>> cores,
                       function_ref<InFlightDiagnostic()> emitError,
                       StringRef what) {
  DenseSet<std::pair<int64_t, int64_t>> seen;
  for (auto core : cores) {
    if (!seen.insert(core).second) {
      emitError() << "duplicate " << what << " core (" << core.first << ", "
                  << core.second << ")";
      return true;
    }
  }
  return false;
}

LogicalResult
verifyExplicitPairs(ArrayRef<SenderReceiversAttr> pairs,
                    function_ref<InFlightDiagnostic()> emitError) {
  if (pairs.empty()) {
    return emitError() << "explicit mapping must contain at least one pair";
  }

  SmallVector<std::pair<int64_t, int64_t>> senders;
  SmallVector<std::pair<int64_t, int64_t>> receivers;
  for (SenderReceiversAttr pair : pairs) {
    ttcore::CoreCoordAttr sender = pair.getSender();
    senders.emplace_back(sender.getY(), sender.getX());
    for (ttcore::CoreRangeAttr recv : pair.getReceivers()) {
      appendCores(recv, receivers);
    }
  }
  if (hasDuplicateCores(senders, emitError, "sender")) {
    return failure();
  }
  if (hasDuplicateCores(receivers, emitError, "receiver")) {
    return failure();
  }

  DenseSet<std::pair<int64_t, int64_t>> receiverSet(receivers.begin(),
                                                    receivers.end());
  for (auto sender : senders) {
    if (receiverSet.contains(sender)) {
      return emitError() << "sender core (" << sender.first << ", "
                         << sender.second << ") overlaps a receiver core";
    }
  }
  return success();
}

} // namespace

::mlir::LogicalResult SenderReceiversAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ttcore::CoreCoordAttr /*sender*/,
    ::llvm::ArrayRef<ttcore::CoreRangeAttr> receivers) {
  if (receivers.empty()) {
    return emitError() << "sender_receivers must have at least one receiver "
                          "core range";
  }
  return success();
}

::mlir::LogicalResult GlobalCBMappingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    GlobalCBMappingKind kind, ttcore::CoreRangeAttr sender,
    ttcore::CoreRangeAttr receiver,
    ::llvm::ArrayRef<SenderReceiversAttr> pairs) {
  switch (kind) {
  case GlobalCBMappingKind::Zip: {
    if (!sender || !receiver) {
      return emitError() << "zip mapping requires sender and receiver "
                            "core ranges";
    }
    if (!pairs.empty()) {
      return emitError() << "zip mapping cannot include explicit pairs";
    }
    if (coreCount(sender) != coreCount(receiver)) {
      return emitError() << "zip mapping requires equal sender and receiver "
                            "core counts, got "
                         << coreCount(sender) << " and " << coreCount(receiver);
    }
    if (sender.intersects(receiver)) {
      return emitError() << "zip mapping sender and receiver core ranges "
                            "must be disjoint";
    }
    return success();
  }
  case GlobalCBMappingKind::RowFanout: {
    if (!sender || !receiver) {
      return emitError() << "row_fanout mapping requires sender and "
                            "receiver core ranges";
    }
    if (!pairs.empty()) {
      return emitError() << "row_fanout mapping cannot include explicit pairs";
    }
    int64_t senderWidth =
        sender.getEndCoord().getX() - sender.getStartCoord().getX() + 1;
    int64_t senderHeight =
        sender.getEndCoord().getY() - sender.getStartCoord().getY() + 1;
    int64_t receiverHeight =
        receiver.getEndCoord().getY() - receiver.getStartCoord().getY() + 1;
    if (senderWidth != 1) {
      return emitError() << "row_fanout mapping requires sender width 1, got "
                         << senderWidth;
    }
    if (senderHeight != receiverHeight) {
      return emitError()
             << "row_fanout mapping requires equal sender and receiver "
                "heights, got "
             << senderHeight << " and " << receiverHeight;
    }
    if (sender.intersects(receiver)) {
      return emitError() << "row_fanout mapping sender and receiver core "
                            "ranges must be disjoint";
    }
    return success();
  }
  case GlobalCBMappingKind::Explicit:
    if (sender || receiver) {
      return emitError() << "explicit mapping cannot include sender/receiver "
                            "core ranges";
    }
    return verifyExplicitPairs(pairs, emitError);
  }
  return emitError() << "unknown global_cb mapping kind";
}

mlir::Attribute GlobalCBMappingAttr::parse(::mlir::AsmParser &parser,
                                           ::mlir::Type) {
  if (parser.parseLess()) {
    return {};
  }

  FailureOr<GlobalCBMappingKind> kind =
      FieldParser<GlobalCBMappingKind>::parse(parser);
  if (failed(kind)) {
    return {};
  }

  if (parser.parseComma()) {
    return {};
  }

  if (*kind == GlobalCBMappingKind::Explicit) {
    SmallVector<SenderReceiversAttr> pairs;
    auto parsePair = [&]() -> ParseResult {
      SenderReceiversAttr pair;
      if (parser.parseAttribute(pair)) {
        return failure();
      }
      pairs.push_back(pair);
      return success();
    };
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       parsePair) ||
        parser.parseGreater()) {
      return {};
    }
    return GlobalCBMappingAttr::getChecked(
        [&] { return parser.emitError(parser.getCurrentLocation()); },
        parser.getContext(), *kind, /*sender=*/nullptr, /*receiver=*/nullptr,
        pairs);
  }

  ttcore::CoreRangeAttr sender;
  ttcore::CoreRangeAttr receiver;
  if (parser.parseKeyword("sender") || parser.parseEqual() ||
      parser.parseAttribute(sender) || parser.parseComma() ||
      parser.parseKeyword("receiver") || parser.parseEqual() ||
      parser.parseAttribute(receiver) || parser.parseGreater()) {
    return {};
  }
  return GlobalCBMappingAttr::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), *kind, sender, receiver, /*pairs=*/{});
}

void GlobalCBMappingAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getKind());
  if (getKind() == GlobalCBMappingKind::Explicit) {
    printer << ", [";
    llvm::interleaveComma(getPairs(), printer, [&](SenderReceiversAttr pair) {
      printer.printAttribute(pair);
    });
    printer << "]>";
    return;
  }
  printer << ", sender = ";
  printer.printAttribute(getSender());
  printer << ", receiver = ";
  printer.printAttribute(getReceiver());
  printer << ">";
}

::mlir::LogicalResult GlobalCBMappingAttr::expand(
    ::llvm::SmallVectorImpl<SenderReceiversAttr> &expanded) const {
  expanded.clear();
  MLIRContext *ctx = getContext();
  switch (getKind()) {
  case GlobalCBMappingKind::Explicit:
    llvm::append_range(expanded, getPairs());
    return success();
  case GlobalCBMappingKind::Zip: {
    SmallVector<std::pair<int64_t, int64_t>> senders;
    SmallVector<std::pair<int64_t, int64_t>> receivers;
    appendCores(getSender(), senders);
    appendCores(getReceiver(), receivers);
    if (senders.size() != receivers.size()) {
      return failure();
    }
    for (auto [senderYX, recvYX] : llvm::zip(senders, receivers)) {
      auto sender =
          ttcore::CoreCoordAttr::get(ctx, senderYX.first, senderYX.second);
      auto recvCoord =
          ttcore::CoreCoordAttr::get(ctx, recvYX.first, recvYX.second);
      auto recvRange = ttcore::CoreRangeAttr::get(ctx, recvCoord, recvCoord);
      expanded.push_back(SenderReceiversAttr::get(ctx, sender, {recvRange}));
    }
    return success();
  }
  case GlobalCBMappingKind::RowFanout: {
    SmallVector<std::pair<int64_t, int64_t>> senders;
    appendCores(getSender(), senders);
    ttcore::CoreRangeAttr receiver = getReceiver();
    int64_t recvStartX = receiver.getStartCoord().getX();
    int64_t recvEndX = receiver.getEndCoord().getX();
    for (auto senderYX : senders) {
      auto sender =
          ttcore::CoreCoordAttr::get(ctx, senderYX.first, senderYX.second);
      auto rowStart =
          ttcore::CoreCoordAttr::get(ctx, senderYX.first, recvStartX);
      auto rowEnd = ttcore::CoreCoordAttr::get(ctx, senderYX.first, recvEndX);
      auto recvRange = ttcore::CoreRangeAttr::get(ctx, rowStart, rowEnd);
      expanded.push_back(SenderReceiversAttr::get(ctx, sender, {recvRange}));
    }
    return success();
  }
  }
  return failure();
}

#include "ttmlir/Dialect/D2M/IR/D2MOpsDialect.cpp.inc"

struct D2MDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    //
    // If this is a GenericOp, protect it from hoisting constants outside of
    // its region body. e.g. do not hoist %const0 outside of the following op:
    //
    // %1 = "d2m.generic"(...) <{...}> ({
    // ^bb0(...):
    //   %const0 = arith.constant 0 : index
    // }) : (...) -> ...
    //
    // As opposed to the default canonicalization behavior, which would hoist it
    // it like this:
    //
    // %const0 = arith.constant 0 : index
    // %1 = "d2m.generic"(...) <{...}> ({
    // ^bb0(...):
    // }) : (...) -> ...
    //
    return isa<GenericOp>(region->getParentOp());
  }
};

void D2MDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/D2M/IR/D2MOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.cpp.inc"
      >();
  addInterfaces<D2MDialectFoldInterface>();
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.cpp.inc"
      >();
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
  registerTypes();
}
