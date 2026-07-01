// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <optional>

namespace mlir::tt::d2m {

namespace {

constexpr StringLiteral kPhysicalCBPortMapAttrName = "d2m.physical_cb_ports";

struct CBUseWindow {
  Operation *firstUser = nullptr;
  Operation *lastUser = nullptr;

  void addUse(Operation *user) {
    if (!firstUser || user->isBeforeInBlock(firstUser)) {
      firstUser = user;
    }
    if (!lastUser || lastUser->isBeforeInBlock(user)) {
      lastUser = user;
    }
  }

  bool valid() const { return firstUser && lastUser; }
};

struct CBPortLifetime {
  OpOperand *operand = nullptr;
  Type type;
  llvm::DenseMap<Block *, CBUseWindow> useWindows;
  SmallVector<GetArgOp> getArgs;
  SmallVector<GetCBOp> getCBs;

  int64_t getOperandIndex() const { return operand->getOperandNumber(); }
};

struct CBReuseGroup {
  CBPortLifetime *representative = nullptr;
  SmallVector<CBPortLifetime *> members;
};

enum class CBUseWindowOrder {
  Unknown,
  FirstBeforeSecond,
  SecondBeforeFirst,
  Overlap,
};

static Type getCBBackedMemRefType(Type type) {
  if (auto cbType = mlir::dyn_cast<CBType>(type)) {
    return cbType.getUnderlyingAs<MemRefType>();
  }

  auto memrefType = mlir::dyn_cast<MemRefType>(type);
  if (!memrefType) {
    return {};
  }

  if (mlir::isa<ttcore::DeviceLayoutInterface>(memrefType.getLayout())) {
    return {};
  }

  if (ttcore::getMemorySpace(memrefType) == ttcore::MemorySpace::RegisterDst) {
    return {};
  }

  if (mlir::isa<StridedLayoutAttr>(memrefType.getLayout())) {
    return {};
  }

  return memrefType;
}

static CBUseWindowOrder getRelativeOrder(const CBUseWindow &lhs,
                                         const CBUseWindow &rhs) {
  if (!lhs.valid() || !rhs.valid()) {
    return CBUseWindowOrder::Unknown;
  }
  if (lhs.lastUser->isBeforeInBlock(rhs.firstUser)) {
    return CBUseWindowOrder::FirstBeforeSecond;
  }
  if (rhs.lastUser->isBeforeInBlock(lhs.firstUser)) {
    return CBUseWindowOrder::SecondBeforeFirst;
  }
  return CBUseWindowOrder::Overlap;
}

static bool canShareCBPort(const CBPortLifetime &lhs,
                           const CBPortLifetime &rhs) {
  if (lhs.type != rhs.type || lhs.useWindows.size() != rhs.useWindows.size()) {
    return false;
  }

  CBUseWindowOrder requiredOrder = CBUseWindowOrder::Unknown;
  for (const auto &[block, lhsWindow] : lhs.useWindows) {
    auto it = rhs.useWindows.find(block);
    if (it == rhs.useWindows.end()) {
      return false;
    }

    CBUseWindowOrder order = getRelativeOrder(lhsWindow, it->second);
    if (order == CBUseWindowOrder::Overlap) {
      return false;
    }
    if (order == CBUseWindowOrder::Unknown) {
      continue;
    }
    if (requiredOrder != CBUseWindowOrder::Unknown && requiredOrder != order) {
      return false;
    }
    requiredOrder = order;
  }

  return true;
}

static Operation *getEnclosingOperationInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block) {
    op = op->getParentOp();
  }
  return op;
}

static void addResultUseWindow(Value value, Block *block,
                               CBPortLifetime &lifetime) {
  for (Operation *user : value.getUsers()) {
    if (Operation *orderedUser = getEnclosingOperationInBlock(user, block)) {
      lifetime.useWindows[block].addUse(orderedUser);
    }
  }
}

static OpOperand *getGenericOperand(GenericOp generic, int64_t operandIdx) {
  if (operandIdx < 0 ||
      operandIdx >= static_cast<int64_t>(generic->getNumOperands())) {
    return nullptr;
  }
  return &generic->getOpOperand(static_cast<unsigned>(operandIdx));
}

static std::optional<std::reference_wrapper<CBPortLifetime>>
getOrCreateLifetime(llvm::MapVector<OpOperand *, CBPortLifetime> &lifetimes,
                    GenericOp generic, int64_t operandIdx, Type type) {
  OpOperand *operand = getGenericOperand(generic, operandIdx);
  if (!operand) {
    return std::nullopt;
  }

  CBPortLifetime &lifetime = lifetimes[operand];
  lifetime.operand = operand;
  if (!lifetime.type) {
    lifetime.type = type;
  }
  if (lifetime.type != type) {
    return std::nullopt;
  }
  return lifetime;
}

} // namespace

StringRef getPhysicalCBPortMapAttrName() { return kPhysicalCBPortMapAttrName; }

void setPhysicalCBPortMap(Operation *op, ArrayRef<int64_t> logicalToPhysical) {
  op->setAttr(getPhysicalCBPortMapAttrName(),
              DenseI64ArrayAttr::get(op->getContext(), logicalToPhysical));
}

DenseI64ArrayAttr getPhysicalCBPortMap(Operation *op) {
  if (!op) {
    return nullptr;
  }
  return op->getAttrOfType<DenseI64ArrayAttr>(getPhysicalCBPortMapAttrName());
}

int64_t getPhysicalCBPort(Operation *op, int64_t logicalOperandIdx) {
  DenseI64ArrayAttr portMap = getPhysicalCBPortMap(op);
  if (!portMap || logicalOperandIdx < 0 ||
      logicalOperandIdx >= static_cast<int64_t>(portMap.asArrayRef().size())) {
    return logicalOperandIdx;
  }
  int64_t physicalPort = portMap.asArrayRef()[logicalOperandIdx];
  return physicalPort < 0 ? logicalOperandIdx : physicalPort;
}

Value getOrCreateCB(RewriterBase &rewriter, GenericOp generic, Block *block,
                    unsigned cbOperandIndex) {
  // If CB already exists, return it
  Value cb = nullptr;
  block->walk([&](GetCBOp getCBOp) {
    if (getCBOp.getCbOperandIdx() == cbOperandIndex) {
      cb = getCBOp.getResult();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (cb) {
    return cb;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(block);
  Value localBuffer = generic.getOperand(cbOperandIndex);
  cb = rewriter
           .create<GetCBOp>(
               generic.getLoc(),
               CBType::get(rewriter.getContext(),
                           mlir::cast<ShapedType>(localBuffer.getType())),
               cbOperandIndex,
               ResolutionStageAttr::get(rewriter.getContext(),
                                        ResolutionStage::Compile))
           .getResult();
  return cb;
}

void reuseDisjointCBPorts(RewriterBase &rewriter, GenericOp generic,
                          int64_t maxPhysicalCBPorts) {
  int64_t firstAdditionalArgIdx =
      generic.getInputs().size() + generic.getOutputs().size();
  llvm::MapVector<OpOperand *, CBPortLifetime> lifetimes;
  rewriter.modifyOpInPlace(
      generic, [&]() { generic->removeAttr(getPhysicalCBPortMapAttrName()); });

  for (Region &region : generic->getRegions()) {
    if (region.empty()) {
      continue;
    }

    Block *block = &region.front();

    block->walk([&](GetArgOp getArgOp) {
      int64_t operandIdx = getArgOp.getOperandIndex();
      Type cbBackedType = getCBBackedMemRefType(getArgOp.getResult().getType());
      if (!cbBackedType) {
        return WalkResult::advance();
      }

      auto lifetime =
          getOrCreateLifetime(lifetimes, generic, operandIdx, cbBackedType);
      if (!lifetime) {
        return WalkResult::advance();
      }
      lifetime->get().getArgs.push_back(getArgOp);
      addResultUseWindow(getArgOp.getResult(), block, lifetime->get());
      return WalkResult::advance();
    });

    block->walk([&](GetCBOp getCBOp) {
      int64_t operandIdx = getCBOp.getCbOperandIdx();
      Type cbBackedType = getCBBackedMemRefType(getCBOp.getResult().getType());
      if (!cbBackedType) {
        return WalkResult::advance();
      }

      auto lifetime =
          getOrCreateLifetime(lifetimes, generic, operandIdx, cbBackedType);
      if (!lifetime) {
        return WalkResult::advance();
      }
      lifetime->get().getCBs.push_back(getCBOp);
      addResultUseWindow(getCBOp.getResult(), block, lifetime->get());
      return WalkResult::advance();
    });
  }

  SmallVector<CBReuseGroup> groups;
  SmallVector<CBPortLifetime *> activeAdditionalLifetimes;
  llvm::DenseMap<CBPortLifetime *, CBPortLifetime *> representativeFor;
  int64_t fixedPortCount = 0;

  for (auto &entry : lifetimes) {
    CBPortLifetime &lifetime = entry.second;
    representativeFor[&lifetime] = &lifetime;
    if (lifetime.useWindows.empty()) {
      continue;
    }

    if (lifetime.getOperandIndex() < firstAdditionalArgIdx) {
      ++fixedPortCount;
      continue;
    }

    groups.push_back({&lifetime, {&lifetime}});
    activeAdditionalLifetimes.push_back(&lifetime);
  }

  int64_t physicalPortCount = fixedPortCount + activeAdditionalLifetimes.size();
  for (CBPortLifetime *lifetime : llvm::reverse(activeAdditionalLifetimes)) {
    if (physicalPortCount <= maxPhysicalCBPorts) {
      break;
    }

    CBReuseGroup *ownGroup = nullptr;
    for (CBReuseGroup &group : groups) {
      if (group.representative == lifetime) {
        ownGroup = &group;
        break;
      }
    }
    if (ownGroup && ownGroup->members.size() > 1) {
      continue;
    }

    CBReuseGroup *selectedGroup = nullptr;
    for (CBReuseGroup &group : groups) {
      CBPortLifetime *representative = group.representative;
      if (representative->getOperandIndex() >= lifetime->getOperandIndex() ||
          representativeFor.lookup(representative) != representative) {
        continue;
      }

      bool canShare = true;
      for (CBPortLifetime *member : group.members) {
        if (!canShareCBPort(*lifetime, *member)) {
          canShare = false;
          break;
        }
      }
      if (canShare) {
        selectedGroup = &group;
        break;
      }
    }

    if (!selectedGroup) {
      continue;
    }

    selectedGroup->members.push_back(lifetime);
    representativeFor[lifetime] = selectedGroup->representative;
    --physicalPortCount;
  }

  SmallVector<int64_t> logicalToPhysical;
  logicalToPhysical.reserve(generic->getNumOperands());
  for (int64_t operandIdx = 0,
               operandCount = static_cast<int64_t>(generic->getNumOperands());
       operandIdx < operandCount; ++operandIdx) {
    logicalToPhysical.push_back(operandIdx);
  }

  bool hasReuse = false;
  for (auto &entry : lifetimes) {
    CBPortLifetime &lifetime = entry.second;
    CBPortLifetime *representative = representativeFor.lookup(&lifetime);
    if (!representative || representative == &lifetime) {
      continue;
    }

    hasReuse = true;
    logicalToPhysical[lifetime.getOperandIndex()] =
        representative->getOperandIndex();
  }

  if (hasReuse) {
    rewriter.modifyOpInPlace(
        generic, [&]() { setPhysicalCBPortMap(generic, logicalToPhysical); });
  }
}

} // namespace mlir::tt::d2m
