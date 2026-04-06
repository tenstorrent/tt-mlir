// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv3dConfigSearchSpace.h"

#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {

// --- Conv3dConfigSearchSpace ------------------------------------------------

Conv3dConfigSearchSpace::Conv3dConfigSearchSpace(ttnn::Conv3dOp conv3dOp,
                                                 Conv3dConfigAttr baseConfig) {
  constexpr uint32_t tileWidth = 32;
  constexpr uint32_t l1Alignment = 16;

  uint32_t inChannels = conv3dOp.getInChannels();
  uint32_t outChannels = conv3dOp.getOutChannels();
  uint32_t paddedOutChannels =
      ((outChannels + tileWidth - 1) / tileWidth) * tileWidth;

  // C_in_block: 0 means "full C_in" (fastest), then explicit divisors large
  // to small.  Must divide C_in and be L1-aligned.
  if (!baseConfig.hasCInBlock()) {
    cInBlock.push_back(0);
    for (uint32_t val = inChannels; val >= tileWidth; val -= tileWidth) {
      if (inChannels % val == 0 && val % l1Alignment == 0) {
        cInBlock.push_back(val);
      }
    }
  }

  // C_out_block: 0 means "full padded_C_out" (fastest), then explicit
  // divisors large to small.  Must be tile-aligned and divide padded_C_out.
  if (!baseConfig.hasCOutBlock()) {
    cOutBlock.push_back(0);
    for (uint32_t val = paddedOutChannels; val >= tileWidth; val -= tileWidth) {
      if (paddedOutChannels % val == 0) {
        cOutBlock.push_back(val);
      }
    }
  }

  // Spatial blocks: default is 1×1×1 (minimum L1).  Only search if the base
  // config hasn't pinned the value.  Include descending values so larger
  // (faster) blocks are tried first.
  auto addSpatialBlock = [](llvm::SmallVector<uint32_t> &out, uint32_t maxVal,
                            bool hasValue) {
    if (!hasValue && maxVal > 1) {
      for (uint32_t v = maxVal; v >= 1; --v) {
        out.push_back(v);
      }
    }
  };

  addSpatialBlock(tOutBlock, conv3dOp.getInputDepth(),
                  baseConfig.hasTOutBlock());
  addSpatialBlock(hOutBlock, conv3dOp.getInputHeight(),
                  baseConfig.hasHOutBlock());
  addSpatialBlock(wOutBlock, conv3dOp.getInputWidth(),
                  baseConfig.hasWOutBlock());
}

// --- Conv3dConfigGenerator --------------------------------------------------

Conv3dConfigGenerator::Conv3dConfigGenerator(
    ttnn::Conv3dOp *op, Conv3dConfigAttr baseConfig,
    const Conv3dConfigSearchSpace &space,
    std::function<bool(const Conv3dConfigAttr &)> filterOutFn)
    : op(op), baseConfig(baseConfig), searchSpace(space),
      filterOutFn(filterOutFn) {

  auto addField =
      [&](const llvm::SmallVector<uint32_t> &values, bool isSet, bool hasValue,
          std::function<Conv3dConfigAttr(Conv3dConfigAttr, uint32_t)> &&apply) {
        if (isSet && !hasValue) {
          activeSearchFields.emplace_back(CyclingCursor(values),
                                          std::move(apply));
        }
      };

  addField(searchSpace.cInBlock, searchSpace.isCInBlockSetForSearch(),
           baseConfig.hasCInBlock(),
           [](Conv3dConfigAttr a, uint32_t v) { return a.withCInBlock(v); });

  addField(searchSpace.cOutBlock, searchSpace.isCOutBlockSetForSearch(),
           baseConfig.hasCOutBlock(),
           [](Conv3dConfigAttr a, uint32_t v) { return a.withCOutBlock(v); });

  addField(searchSpace.tOutBlock, searchSpace.isTOutBlockSetForSearch(),
           baseConfig.hasTOutBlock(),
           [](Conv3dConfigAttr a, uint32_t v) { return a.withTOutBlock(v); });

  addField(searchSpace.wOutBlock, searchSpace.isWOutBlockSetForSearch(),
           baseConfig.hasWOutBlock(),
           [](Conv3dConfigAttr a, uint32_t v) { return a.withWOutBlock(v); });

  addField(searchSpace.hOutBlock, searchSpace.isHOutBlockSetForSearch(),
           baseConfig.hasHOutBlock(),
           [](Conv3dConfigAttr a, uint32_t v) { return a.withHOutBlock(v); });

  isDone = activeSearchFields.empty();
}

Conv3dConfigAttr Conv3dConfigGenerator::generateCurrent() const {
  Conv3dConfigAttr config = baseConfig;
  for (const ActiveFieldEntry &field : activeSearchFields) {
    config = field.applyValue(config, *field.cursor);
  }
  return config;
}

void Conv3dConfigGenerator::advanceOdometer() {
  // Odometer-style: advance last field; on wrap, carry to next.
  for (int pos = activeSearchFields.size() - 1; pos >= 0; --pos) {
    if (!activeSearchFields[pos].cursor.advance()) {
      return;
    }
  }
  // All fields wrapped — search exhausted.
  isDone = true;
}

// --- iterator implementation ------------------------------------------------

Conv3dConfigAttr
Conv3dConfigGenerator::advanceToNextValid(Conv3dConfigGenerator *gen) {
  while (!gen->isDone) {
    Conv3dConfigAttr config = gen->generateCurrent();
    gen->advanceOdometer();
    if (!gen->filterOutFn(config)) {
      TTMLIR_TRACE(ttmlir::LogComponent::ValidationFallback,
                   "Next conv3d config: {}", config);
      return config;
    }
    TTMLIR_TRACE(ttmlir::LogComponent::ValidationFallback, "Filtered out {}",
                 config);
  }
  return nullptr;
}

Conv3dConfigGenerator::iterator::iterator(Conv3dConfigGenerator *gen)
    : gen(gen) {
  current = advanceToNextValid(gen);
}

Conv3dConfigGenerator::iterator &Conv3dConfigGenerator::iterator::operator++() {
  assert(gen && current && "incrementing past-the-end iterator");
  current = advanceToNextValid(gen);
  return *this;
}

} // namespace mlir::tt::ttnn
