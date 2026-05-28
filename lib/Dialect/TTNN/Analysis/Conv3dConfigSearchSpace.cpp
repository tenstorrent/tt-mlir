// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv3dConfigSearchSpace.h"

#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {

namespace {

// Currently selected value for each Conv3dConfigAttr field. baseConfig
// pre-populates fields not in the search; active search fields overwrite
// their slot each iteration before the attr is built.
struct ConfigSnapshot {
  std::optional<ttcore::DataType> weightsDtype;
  std::optional<uint32_t> tOutBlock;
  std::optional<uint32_t> wOutBlock;
  std::optional<uint32_t> hOutBlock;
  std::optional<uint32_t> cOutBlock;
  std::optional<uint32_t> cInBlock;
  std::optional<ttcore::GridAttr> computeWithStorageGridSize;
};

enum class Field {
  WeightsDtype,
  TOutBlock,
  WOutBlock,
  HOutBlock,
  COutBlock,
  CInBlock,
  ComputeWithStorageGridSize,
};

// Each active field has a vector of candidate values and a cursor into it.
// Cursors are advanced like a multi-digit odometer (most significant = first
// active field).
struct ActiveField {
  Field field;
  size_t index = 0;
  size_t size = 0;

  // Returns true if the cursor wrapped around to 0 (carry to a more
  // significant neighbour).
  bool advance() {
    if (++index >= size) {
      index = 0;
      return true;
    }
    return false;
  }
};

void applyCursorToSnapshot(const ActiveField &af,
                           const Conv3dConfigSearchSpace &space,
                           ConfigSnapshot &snapshot) {
  size_t i = af.index;
  switch (af.field) {
  case Field::WeightsDtype:
    snapshot.weightsDtype = space.weightsDtype[i];
    break;
  case Field::TOutBlock:
    snapshot.tOutBlock = space.tOutBlock[i];
    break;
  case Field::WOutBlock:
    snapshot.wOutBlock = space.wOutBlock[i];
    break;
  case Field::HOutBlock:
    snapshot.hOutBlock = space.hOutBlock[i];
    break;
  case Field::COutBlock:
    snapshot.cOutBlock = space.cOutBlock[i];
    break;
  case Field::CInBlock:
    snapshot.cInBlock = space.cInBlock[i];
    break;
  case Field::ComputeWithStorageGridSize:
    snapshot.computeWithStorageGridSize = space.computeWithStorageGridSize[i];
    break;
  }
}

Conv3dConfigAttr buildAttrFromSnapshot(::mlir::MLIRContext *ctx,
                                       const ConfigSnapshot &snapshot) {
  return Conv3dConfigAttr::get(ctx, snapshot.weightsDtype, snapshot.tOutBlock,
                               snapshot.wOutBlock, snapshot.hOutBlock,
                               snapshot.cOutBlock, snapshot.cInBlock,
                               snapshot.computeWithStorageGridSize);
}

} // namespace

bool forEachConv3dConfig(
    Conv3dOp *op, Conv3dConfigAttr baseConfig,
    const Conv3dConfigSearchSpace &space,
    llvm::function_ref<bool(const Conv3dConfigAttr &)> filterOut,
    llvm::function_ref<void(Conv3dConfigAttr)> callback) {
  assert(baseConfig &&
         "forEachConv3dConfig requires a non-null baseConfig so it can "
         "source an MLIRContext.");
  (void)op;

  ::mlir::MLIRContext *ctx = baseConfig.getContext();

  // Seed snapshot from baseConfig: any field already pinned on baseConfig is
  // copied through verbatim and excluded from the search.
  ConfigSnapshot snapshot;
  snapshot.weightsDtype = baseConfig.getWeightsDtype();
  snapshot.tOutBlock = baseConfig.getTOutBlock();
  snapshot.wOutBlock = baseConfig.getWOutBlock();
  snapshot.hOutBlock = baseConfig.getHOutBlock();
  snapshot.cOutBlock = baseConfig.getCOutBlock();
  snapshot.cInBlock = baseConfig.getCInBlock();
  snapshot.computeWithStorageGridSize =
      baseConfig.getComputeWithStorageGridSize();

  llvm::SmallVector<ActiveField> activeFields;
  auto addActive = [&](Field f, size_t n) {
    activeFields.push_back(ActiveField{f, /*index=*/0, /*size=*/n});
  };

  if (space.isWeightsDtypeSetForSearch() &&
      !snapshot.weightsDtype.has_value()) {
    addActive(Field::WeightsDtype, space.weightsDtype.size());
  }
  if (space.isTOutBlockSetForSearch() && !snapshot.tOutBlock.has_value()) {
    addActive(Field::TOutBlock, space.tOutBlock.size());
  }
  if (space.isWOutBlockSetForSearch() && !snapshot.wOutBlock.has_value()) {
    addActive(Field::WOutBlock, space.wOutBlock.size());
  }
  if (space.isHOutBlockSetForSearch() && !snapshot.hOutBlock.has_value()) {
    addActive(Field::HOutBlock, space.hOutBlock.size());
  }
  if (space.isCOutBlockSetForSearch() && !snapshot.cOutBlock.has_value()) {
    addActive(Field::COutBlock, space.cOutBlock.size());
  }
  if (space.isCInBlockSetForSearch() && !snapshot.cInBlock.has_value()) {
    addActive(Field::CInBlock, space.cInBlock.size());
  }
  if (space.isComputeWithStorageGridSizeSetForSearch() &&
      !snapshot.computeWithStorageGridSize.has_value()) {
    addActive(Field::ComputeWithStorageGridSize,
              space.computeWithStorageGridSize.size());
  }

  if (activeFields.empty()) {
    return false;
  }

  // Initialize the snapshot's active-field slots to the first candidate
  // value so the first emitted attr is well-formed.
  for (const ActiveField &af : activeFields) {
    applyCursorToSnapshot(af, space, snapshot);
  }

  while (true) {
    Conv3dConfigAttr generated = buildAttrFromSnapshot(ctx, snapshot);
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Next conv3d config: {}",
                 generated);

    if (!filterOut || !filterOut(generated)) {
      callback(generated);
    } else {
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Filtered out {}",
                   generated);
    }

    // Odometer-advance: try to bump the least significant active field
    // first. If it wraps to 0, carry over to the more significant neighbour.
    int currentFieldToAdvance = static_cast<int>(activeFields.size()) - 1;
    while (currentFieldToAdvance >= 0) {
      ActiveField &af = activeFields[currentFieldToAdvance];
      bool wrapped = af.advance();
      applyCursorToSnapshot(af, space, snapshot);
      if (wrapped) {
        currentFieldToAdvance--;
      } else {
        break;
      }
    }

    if (currentFieldToAdvance < 0) {
      break;
    }
  }

  return true;
}

} // namespace mlir::tt::ttnn
