// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/Roofline.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::roofline {

std::optional<HwSpec> getHwSpec(ttcore::Arch arch) {
  HwSpec spec;
  // HiFi2 = 32 cycles / 32x32x32 tile-mul; same on WH and BH per
  // tech_reports/matrix_engine/matrix_engine.md.
  spec.cyclesPerTileMatmul = 32;
  switch (arch) {
  case ttcore::Arch::WormholeB0:
    spec.dramBandwidthBytesPerSec = 288ULL * 1000ULL * 1000ULL * 1000ULL;
    spec.aiclkHz = 1000ULL * 1000ULL * 1000ULL; // 1.0 GHz
    return spec;
  case ttcore::Arch::Blackhole:
    spec.dramBandwidthBytesPerSec = 512ULL * 1000ULL * 1000ULL * 1000ULL;
    spec.aiclkHz = 1350ULL * 1000ULL * 1000ULL; // 1.35 GHz
    return spec;
  default:
    return std::nullopt;
  }
}

uint64_t getNumTileMatmuls(Value lhs, Value result, bool transposeA) {
  auto lhsType = mlir::dyn_cast<RankedTensorType>(lhs.getType());
  auto resultType = mlir::dyn_cast<RankedTensorType>(result.getType());
  if (!lhsType || !resultType) {
    return 0;
  }
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();

  // Both operands must be at least rank 2 to extract M/K/N.
  if (lhsShape.size() < 2 || resultShape.size() < 2) {
    return 0;
  }

  int64_t M = resultShape[resultShape.size() - 2];
  // The contraction dim K is the last dim of A, unless A is transposed, in
  // which case it is the second-to-last dim.
  int64_t K = transposeA ? lhsShape[lhsShape.size() - 2]
                         : lhsShape[lhsShape.size() - 1];
  int64_t N = resultShape[resultShape.size() - 1];
  int64_t batch = 1;
  for (size_t i = 0; i < resultShape.size() - 2; i++) {
    batch *= resultShape[i];
  }

  const int64_t tileH = ttcore::TileType::getDefaultShape()[0];
  const int64_t tileW = ttcore::TileType::getDefaultShape()[1];
  int64_t tilesM = (M + tileH - 1) / tileH;
  int64_t tilesK = (K + tileW - 1) / tileW;
  int64_t tilesN = (N + tileW - 1) / tileW;

  return static_cast<uint64_t>(batch) * tilesM * tilesK * tilesN;
}

} // namespace mlir::tt::ttnn::roofline
