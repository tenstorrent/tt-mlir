// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/AffineMap.h"

#include <cstdint>
#include <vector>

namespace mlir::ttmlir::python {
void populateTTIRModule(nb::module_ &m) {
  m.def(
      "rearrange_inv_pattern_map",
      [](MlirContext context, std::string pattern, std::vector<int64_t> shape) {
        mlir::FailureOr<AffineMap> failureOrMap = tt::ttir::RearrangeOp::getInvPatternMap(unwrap(context), pattern, shape);
        if (failed(failureOrMap)) {
          throw std::runtime_error("rearrange_pattern_affine_map: failed to parse pattern \"" + pattern + "\".");
        }
        return wrap(*failureOrMap);
      });

  m.def(
      "affine_map_compose",
      [](MlirAffineMap map, std::vector<int64_t> index) {
        auto sample = unwrap(map).compose(index);
        return std::vector<int64_t>(sample.begin(), sample.end());
      });
}
} // namespace mlir::ttmlir::python
