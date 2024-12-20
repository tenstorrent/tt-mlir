// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_CORERANGESET_H
#define TTMLIR_DIALECT_TT_UTILS_CORERANGESET_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Utils.h"
#include <tuple>

namespace mlir::tt {

using locsize2d = std::tuple<std::array<int32_t, 2>,  //  {{locX, locY},
                             std::array<int32_t, 2>>; // {sizeX, sizeY}}

inline std::vector<locsize2d> toCoreRangeSet(GridAttr virtualGrid,
                                             GridAttr deviceGrid) {
  llvm::SmallVector<std::int64_t> tensorGridShape(virtualGrid.getShape());
  std::vector<locsize2d> coreRangeSet;
  AffineMap mapping = deviceGrid.getMapping();
  ::ttmlir::utils::sample(
      tensorGridShape, [&](ArrayRef<std::int64_t> virtualCoreCoord) {
        llvm::SmallVector<std::int64_t> coreCoord =
            mapping.compose(virtualCoreCoord);
        assert(coreCoord.size() == PhysGridResultIdx::NumIndices &&
               "expected a 2D core");
        assert(coreCoord[PhysGridResultIdx::DeviceIdx] == 0 &&
               "expected single device");

        if (!coreRangeSet.empty() &&
            ((std::get<0>(coreRangeSet.back())[1] ==
              coreCoord[PhysGridResultIdx::CoreCoordY]) &&
             (std::get<0>(coreRangeSet.back())[0] +
              std::get<1>(coreRangeSet.back())[0]) ==
                 coreCoord[PhysGridResultIdx::CoreCoordX])) {
          const auto &[loc, size] = coreRangeSet.back();
          coreRangeSet.back() = {loc, {size[0] + 1, size[1]}};
        } else {
          coreRangeSet.push_back(
              {{static_cast<int32_t>(coreCoord[PhysGridResultIdx::CoreCoordX]),
                static_cast<int32_t>(coreCoord[PhysGridResultIdx::CoreCoordY])},
               {1, 1}});
        }
        if (coreRangeSet.size() > 1) {
          const auto &[locPrev, sizePrev] =
              coreRangeSet[coreRangeSet.size() - 2];
          const auto &[loc, size] = coreRangeSet.back();
          if ((locPrev[0] == loc[0]) && (sizePrev[0] == size[0]) &&
              ((locPrev[1] + sizePrev[1]) == loc[1])) {
            assert(size[1] == 1);
            coreRangeSet[coreRangeSet.size() - 2] = {
                locPrev, {sizePrev[0], sizePrev[1] + 1}};
            coreRangeSet.pop_back();
          }
        }
      });
  return coreRangeSet;
}

} // namespace mlir::tt

#endif // TTMLIR_DIALECT_TT_UTILS_CORERANGESET_H