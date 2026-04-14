// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "op_tensor_trace.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <unordered_set>
#include <vector>

#include <sys/stat.h>
#include <cerrno>

namespace tt::runtime::ttnn {
namespace {

// Use TTNN log type so TTMLIR_RUNTIME_LOGGER_TYPES=RuntimeTTNN captures only these
// lines (not other runtime channels on LogAlways).
constexpr ::tt::runtime::logger::LogType kOpTraceLog =
    ::tt::runtime::logger::LogRuntimeTTNN;

bool envFlag(const char *v) {
  if (!v || !v[0]) {
    return false;
  }
  std::string s(v);
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

bool bufferAllZeros(const void *p, size_t n) {
  const auto *b = static_cast<const uint8_t *>(p);
  for (size_t i = 0; i < n; ++i) {
    if (b[i] != 0) {
      return false;
    }
  }
  return true;
}

std::optional<bool> hostTensorAllZero(const ::tt::runtime::Tensor &hostTensor) {
  // Use TTNN tensor sizing (same as memcpy); TensorDesc can lag or report
  // itemsize/volume inconsistently for some host/mesh views after toHost.
  const ::ttnn::Tensor &tt =
      utils::getTTNNTensorFromRuntimeTensor(hostTensor);
  const uint32_t elem = tt.element_size();
  const uint32_t physVol = tt.physical_volume();
  const size_t n =
      static_cast<size_t>(physVol) * static_cast<size_t>(elem);
  const uint32_t logVol = tt.logical_volume();

  if (n == 0) {
    if (logVol > 0U && elem > 0U) {
      return std::nullopt;
    }
    return true;
  }
  std::vector<std::byte> buf(n);
  ::tt::runtime::ttnn::memcpy(buf.data(), hostTensor);
  return bufferAllZeros(buf.data(), n);
}

CallbackContext makeCb(ProgramContext *pc) {
  return CallbackContext(::tt::runtime::utils::unsafeBorrowShared(pc),
                       ::tt::runtime::DeviceRuntime::TTNN);
}

OpContext makeOp(const ::tt::target::ttnn::Operation *op) {
  return OpContext(::tt::runtime::utils::unsafeBorrowShared(
                       const_cast<::tt::target::ttnn::Operation *>(op)),
                   ::tt::runtime::DeviceRuntime::TTNN);
}

std::string refBrief(const ::tt::runtime::TensorRef &ref) {
  const auto &tr = ref.as<::tt::target::ttnn::TensorRef>(
      ::tt::runtime::DeviceRuntime::TTNN);
  std::ostringstream o;
  o << "global_id=" << tr.global_id();
  return o.str();
}

// One-line tensor summary for logs: "64x8x17x32 dt=2 b=12345" (no brackets).
std::string tensorBrief(const ::tt::runtime::Tensor &t) {
  const ::ttnn::Tensor &tt = utils::getTTNNTensorFromRuntimeTensor(t);
  ::tt::runtime::TensorDesc d = getTensorDesc(t);
  const size_t payloadBytes =
      static_cast<size_t>(tt.physical_volume()) * static_cast<size_t>(tt.element_size());
  std::ostringstream o;
  for (size_t i = 0; i < d.shape.size(); ++i) {
    if (i) {
      o << "x";
    }
    o << d.shape[i];
  }
  o << " dt=" << static_cast<int>(d.dataType) << " b=" << payloadBytes;
  if (payloadBytes == 0 && tt.logical_volume() > 0U && tt.element_size() > 0U) {
    o << " UNREADABLE_HOST(log_vol=" << tt.logical_volume() << ")";
  }
  return o.str();
}

// Mesh tensors yield multiple host shards from toHost; retrieveTensorFromPool
// LOG_FATALs in that case — aggregate all_zero across shards here.
std::optional<bool> poolRefAllShardsZero(const CallbackContext &cb,
                                         const ::tt::runtime::TensorRef &ref) {
  const auto &programContext =
      cb.as<ProgramContext>(::tt::runtime::DeviceRuntime::TTNN);
  const ProgramTensorPool &tensorPool = programContext.getTensorPool();
  const auto *tensorRefPtr =
      &ref.as<::tt::target::ttnn::TensorRef>(::tt::runtime::DeviceRuntime::TTNN);
  if (!tensorRefPtr || !tensorPool.contains(tensorRefPtr)) {
    return std::nullopt;
  }
  ::tt::runtime::Tensor outTensor = utils::createRuntimeTensorFromTTNN(
      tensorPool.getTTNNTensorAndValidate(tensorRefPtr));
  std::vector<::tt::runtime::Tensor> hostTensors =
      ::tt::runtime::ttnn::toHost(outTensor, /*untilize=*/true);
  if (hostTensors.empty()) {
    return std::nullopt;
  }
  bool anyFail = false;
  bool anyNonZero = false;
  for (const ::tt::runtime::Tensor &h : hostTensors) {
    std::optional<bool> oz = hostTensorAllZero(h);
    if (!oz.has_value()) {
      anyFail = true;
    } else if (!*oz) {
      anyNonZero = true;
    }
  }
  if (anyNonZero) {
    return false;
  }
  if (anyFail) {
    return std::nullopt;
  }
  return true;
}

// MLIR debug strings embed the readable op as `"ttnn.foo"` (e.g. multiply, add).
std::string extractMlirTtnnOpName(const char *debug) {
  if (!debug) {
    return {};
  }
  std::string s(debug);
  const std::string needle = "\"ttnn.";
  size_t pos = 0;
  while ((pos = s.find(needle, pos)) != std::string::npos) {
    pos += needle.size();
    size_t end = pos;
    while (end < s.size() && (std::isalnum(static_cast<unsigned char>(s[end])) ||
                              s[end] == '_')) {
      ++end;
    }
    if (end > pos) {
      return "ttnn." + s.substr(pos, end - pos);
    }
  }
  return {};
}

// Last inner `loc("name")` in nested loc(...) strings.
std::string extractShortLoc(const char *loc) {
  if (!loc) {
    return {};
  }
  std::string s(loc);
  const std::string key = "loc(\"";
  std::string last;
  size_t p = 0;
  while ((p = s.find(key, p)) != std::string::npos) {
    p += key.size();
    size_t q = s.find('"', p);
    if (q != std::string::npos) {
      last = s.substr(p, q - p);
      p = q + 1;
    } else {
      break;
    }
  }
  return last;
}

std::string formatDeviceIds(const std::vector<std::uint32_t> &devs) {
  if (devs.empty()) {
    return {};
  }
  if (devs.size() <= 16) {
    std::ostringstream o;
    for (size_t i = 0; i < devs.size(); ++i) {
      if (i) {
        o << ",";
      }
      o << devs[i];
    }
    return o.str();
  }
  return std::to_string(devs.size()) + "_devices";
}

bool opTensorTraceMeshDetailEnabled() {
  return envFlag(std::getenv("TT_RUNTIME_OP_TENSOR_TRACE_MESH_DETAIL"));
}

// Large per-device / full-tensor row previews (logTensorPreview). Off by default;
// does not follow TT_RUNTIME_OP_TENSOR_TRACE_VERBOSE.
bool opTensorTraceMeshTensorPreviewsEnabled() {
  return envFlag(std::getenv("TT_RUNTIME_OP_TENSOR_TRACE_MESH_PREVIEWS"));
}

bool opTensorTraceFastMode() {
  return envFlag(std::getenv("TT_RUNTIME_OP_TENSOR_TRACE_FAST"));
}

// Ops where per-device values often differ (sharding, collectives).
bool opTensorTraceWantsPerDeviceHeavy(const ::tt::target::ttnn::Operation *op) {
  if (!op) {
    return false;
  }
  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::MeshShardOp:
  case ::tt::target::ttnn::OpType::MeshPartitionOp:
  case ::tt::target::ttnn::OpType::AllGatherOp:
  case ::tt::target::ttnn::OpType::AllReduceOp:
  case ::tt::target::ttnn::OpType::ReduceScatterOp:
  case ::tt::target::ttnn::OpType::DistributeTensorOp:
  case ::tt::target::ttnn::OpType::AllToAllCombineOp:
  case ::tt::target::ttnn::OpType::AllToAllDispatchOp:
  case ::tt::target::ttnn::OpType::PointToPointOp:
  case ::tt::target::ttnn::OpType::LayerNormPreAllGatherOp:
  case ::tt::target::ttnn::OpType::AggregateTensorOp:
    return true;
  default:
    return false;
  }
}

uint64_t fnv1a64Bytes(const uint8_t *p, size_t n) {
  uint64_t h = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= kPrime;
  }
  return h;
}

float bf16BitsToFloat(uint16_t u) {
  uint32_t bits = static_cast<uint32_t>(u) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

void unravelRowMajor(size_t flat, const std::vector<uint32_t> &shape,
                     std::vector<uint32_t> &coord) {
  coord.resize(shape.size());
  for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
    coord[static_cast<size_t>(dim)] =
        static_cast<uint32_t>(flat % shape[static_cast<size_t>(dim)]);
    flat /= shape[static_cast<size_t>(dim)];
  }
}

size_t linearElemOffset(const std::vector<uint32_t> &coord,
                        const ::tt::runtime::TensorDesc &d) {
  size_t o = 0;
  for (size_t i = 0; i < coord.size(); ++i) {
    o += static_cast<size_t>(coord[i]) * static_cast<size_t>(d.stride[i]);
  }
  return o;
}

struct HostTensorNumericDigest {
  uint64_t fnv = 0;
  double absSum = std::numeric_limits<double>::quiet_NaN();
};

HostTensorNumericDigest hostTensorNumericDigest(
    const ::tt::runtime::Tensor &hostTensor) {
  HostTensorNumericDigest out;
  ::tt::runtime::TensorDesc d = getTensorDesc(hostTensor);
  const ::ttnn::Tensor &tt =
      utils::getTTNNTensorFromRuntimeTensor(hostTensor);
  const size_t nbytes =
      static_cast<size_t>(tt.physical_volume()) *
      static_cast<size_t>(tt.element_size());
  std::vector<uint8_t> buf(nbytes);
  ::tt::runtime::ttnn::memcpy(buf.data(), hostTensor);
  out.fnv = fnv1a64Bytes(buf.data(), buf.size());
  if (d.volume() == 0) {
    out.absSum = 0.0;
    return out;
  }
  if (d.itemsize == 2) {
    double as = 0.0;
    for (size_t li = 0; li < d.volume(); ++li) {
      std::vector<uint32_t> coord;
      unravelRowMajor(li, d.shape, coord);
      const size_t off = linearElemOffset(coord, d);
      uint16_t u = 0;
      std::memcpy(&u, buf.data() + off * d.itemsize, 2);
      const double v = static_cast<double>(bf16BitsToFloat(u));
      as += std::abs(v);
    }
    out.absSum = as;
  } else if (d.itemsize == 4) {
    double as = 0.0;
    for (size_t li = 0; li < d.volume(); ++li) {
      std::vector<uint32_t> coord;
      unravelRowMajor(li, d.shape, coord);
      const size_t off = linearElemOffset(coord, d);
      float f = 0.0f;
      std::memcpy(&f, buf.data() + off * d.itemsize, sizeof(f));
      as += std::abs(static_cast<double>(f));
    }
    out.absSum = as;
  }
  return out;
}

std::pair<uint64_t, double> hostTensorFnvAndAbsSum(
    const ::tt::runtime::Tensor &hostTensor) {
  const HostTensorNumericDigest d = hostTensorNumericDigest(hostTensor);
  return {d.fnv, d.absSum};
}

double sliceAbsSumRowMajor(const ::tt::runtime::TensorDesc &d,
                           const uint8_t *buf, unsigned shardDim,
                           size_t startElem, size_t extentAlongDim) {
  if (d.volume() == 0 || extentAlongDim == 0 ||
      shardDim >= d.shape.size()) {
    return 0.0;
  }
  const size_t endExclusive = startElem + extentAlongDim;
  if (d.itemsize == 2) {
    double s = 0.0;
    for (size_t li = 0; li < d.volume(); ++li) {
      std::vector<uint32_t> coord;
      unravelRowMajor(li, d.shape, coord);
      const size_t di = coord[shardDim];
      if (di < startElem || di >= endExclusive) {
        continue;
      }
      const size_t off = linearElemOffset(coord, d);
      uint16_t u = 0;
      std::memcpy(&u, buf + off * d.itemsize, 2);
      s += std::abs(static_cast<double>(bf16BitsToFloat(u)));
    }
    return s;
  }
  if (d.itemsize == 4) {
    double s = 0.0;
    for (size_t li = 0; li < d.volume(); ++li) {
      std::vector<uint32_t> coord;
      unravelRowMajor(li, d.shape, coord);
      const size_t di = coord[shardDim];
      if (di < startElem || di >= endExclusive) {
        continue;
      }
      const size_t off = linearElemOffset(coord, d);
      float f = 0.0f;
      std::memcpy(&f, buf + off * d.itemsize, sizeof(f));
      s += std::abs(static_cast<double>(f));
    }
    return s;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

std::vector<::tt::runtime::Tensor>
hostShardsFromPoolRef(ProgramContext *programContext,
                      const ::tt::target::ttnn::TensorRef *tensorRefPtr);

std::unordered_map<std::uint32_t, ::tt::runtime::Tensor>
perDeviceHostTensorsFromPoolRef(
    ProgramContext *programContext,
    const ::tt::runtime::TensorRef &ref) {
  std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> m;
  if (!programContext) {
    return m;
  }
  const auto *tensorRefPtr =
      &ref.as<::tt::target::ttnn::TensorRef>(::tt::runtime::DeviceRuntime::TTNN);
  std::vector<::tt::runtime::Tensor> shards =
      hostShardsFromPoolRef(programContext, tensorRefPtr);
  for (size_t i = 0; i < shards.size(); ++i) {
    m[static_cast<std::uint32_t>(i)] = std::move(shards[i]);
  }
  return m;
}

void logGroupedPerDeviceSums(
    const std::string &pfx, const char *label,
    const std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> &perDev) {
  if (perDev.empty()) {
    return;
  }
  constexpr unsigned kWidth = 8;
  constexpr unsigned kRows = 4;
  std::uint32_t maxDev = 0;
  for (const auto &kv : perDev) {
    maxDev = std::max(maxDev, kv.first);
  }
  double totalAbs = 0.0;
  for (unsigned row = 0; row < kRows; ++row) {
    const unsigned base = row * kWidth;
    std::ostringstream line;
    line << std::fixed << std::setprecision(5);
    line << "OPSUM | " << label << " | devices " << base << "-"
         << (base + kWidth - 1) << " | ";
    double rowAbs = 0.0;
    for (unsigned j = 0; j < kWidth; ++j) {
      const std::uint32_t d = base + j;
      if (j) {
        line << " ";
      }
      auto it = perDev.find(d);
      if (it == perDev.end()) {
        line << "d" << d << "=-";
        continue;
      }
      const HostTensorNumericDigest dig = hostTensorNumericDigest(it->second);
      if (!std::isnan(dig.absSum)) {
        rowAbs += dig.absSum;
      }
      line << "d" << d << ":|Σ|=" << dig.absSum;
    }
    line << " | row_|Σ|=" << rowAbs;
    LOG_INFO(kOpTraceLog, pfx, line.str());
    totalAbs += rowAbs;
  }
  std::ostringstream tot;
  tot << std::fixed << std::setprecision(6);
  tot << "OPSUM | " << label << " | ALL[0-" << (kRows * kWidth - 1)
      << "] total_|Σ|=" << totalAbs << " max_dev_id=" << maxDev;
  LOG_INFO(kOpTraceLog, pfx, tot.str());
}

void logAllGatherSumInvariant(
    const std::string &pfx, ProgramContext *programContext,
    const ::tt::target::ttnn::Operation *op, OpContext &oc, CallbackContext &cb,
    const std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> &outMap) {
  if (!op || op->type_type() != ::tt::target::ttnn::OpType::AllGatherOp) {
    return;
  }
  std::vector<::tt::runtime::TensorRef> inRefs = getOpInputRefs(oc, cb);
  if (inRefs.empty()) {
    return;
  }
  std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> inMap =
      perDeviceHostTensorsFromPoolRef(programContext, inRefs[0]);
  if (inMap.empty()) {
    LOG_WARNING(kOpTraceLog, pfx,
                "OPSUM | all_gather | WARN | no per-device host input read");
    return;
  }
  logGroupedPerDeviceSums(pfx, "IN0 (all_gather shard)", inMap);
  double sumAllShardAbs = 0.0;
  for (const auto &kv : inMap) {
    const HostTensorNumericDigest d = hostTensorNumericDigest(kv.second);
    if (!std::isnan(d.absSum)) {
      sumAllShardAbs += d.absSum;
    }
  }
  {
    std::ostringstream os;
    os << std::fixed << std::setprecision(8);
    os << "OPSUM | all_gather | disjoint_input_|Σ| (sum of per-device IN0 |Σ|)="
       << sumAllShardAbs;
    LOG_INFO(kOpTraceLog, pfx, os.str());
  }
  unsigned passCount = 0;
  unsigned failCount = 0;
  const double tol =
      1e-1 * std::max(1.0, sumAllShardAbs) +
      1.0;
  for (const auto &kv : outMap) {
    const HostTensorNumericDigest od = hostTensorNumericDigest(kv.second);
    const bool ok =
        !std::isnan(od.absSum) && !std::isnan(sumAllShardAbs) &&
        std::abs(od.absSum - sumAllShardAbs) <= tol;
    if (ok) {
      ++passCount;
    } else {
      ++failCount;
    }
    std::ostringstream dl;
    dl << std::fixed << std::setprecision(8);
    dl << "OPSUM | all_gather | dev=" << kv.first << " out_|Σ|=" << od.absSum
       << " vs global_|Σ|=" << sumAllShardAbs << " | "
       << (ok ? "PASS" : "FAIL");
    LOG_INFO(kOpTraceLog, pfx, dl.str());
  }
  LOG_INFO(kOpTraceLog, pfx,
           "OPSUM | all_gather | summary | pass_devs=", passCount,
           " fail_devs=", failCount,
           " (each device out_|Σ| should match sum of input-shard |Σ|)");
}

std::string escapeJsonString(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '"') { out += "\\\""; }
    else if (c == '\\') { out += "\\\\"; }
    else if (c == '\n') { out += "\\n"; }
    else { out += c; }
  }
  return out;
}

std::string shapeToString(const std::vector<uint32_t> &sh) {
  std::ostringstream o;
  for (size_t i = 0; i < sh.size(); ++i) {
    if (i) {
      o << "x";
    }
    o << sh[i];
  }
  return o.str();
}

void logTensorPreview(const std::string &pfx, const char *label,
                      const ::tt::runtime::Tensor &hostTensor,
                      unsigned maxRows, unsigned maxCols) {
  ::tt::runtime::TensorDesc d = getTensorDesc(hostTensor);
  const ::ttnn::Tensor &tt =
      utils::getTTNNTensorFromRuntimeTensor(hostTensor);
  const size_t nbytes =
      static_cast<size_t>(tt.physical_volume()) *
      static_cast<size_t>(tt.element_size());
  std::vector<uint8_t> buf(nbytes);
  ::tt::runtime::ttnn::memcpy(buf.data(), hostTensor);
  LOG_INFO(kOpTraceLog, pfx, "MESH | ", label, " | shape=", shapeToString(d.shape),
           " | dt=", static_cast<int>(d.dataType), " | itemsize=", d.itemsize);
  if (d.shape.empty() || d.volume() == 0) {
    return;
  }
  if (d.itemsize != 2 && d.itemsize != 4) {
    LOG_INFO(kOpTraceLog, pfx, "MESH | ", label, " | preview=skipped_non_fp16_fp32");
    return;
  }
  auto fmtElem = [&](size_t linearIdx) -> std::string {
    std::vector<uint32_t> coord;
    unravelRowMajor(linearIdx, d.shape, coord);
    const size_t off = linearElemOffset(coord, d);
    if (d.itemsize == 2) {
      uint16_t u = 0;
      std::memcpy(&u, buf.data() + off * d.itemsize, 2);
      std::ostringstream os;
      os << std::setprecision(4) << bf16BitsToFloat(u);
      return os.str();
    }
    float f = 0.0f;
    std::memcpy(&f, buf.data() + off * d.itemsize, sizeof(f));
    std::ostringstream os;
    os << std::setprecision(5) << f;
    return os.str();
  };
  if (d.shape.size() == 1) {
    const size_t n = std::min<size_t>(d.shape[0], maxCols);
    std::ostringstream row;
    row << "[";
    for (size_t i = 0; i < n; ++i) {
      if (i) {
        row << ", ";
      }
      row << fmtElem(i);
    }
    if (d.shape[0] > n) {
      row << ", ...";
    }
    row << "]";
    LOG_INFO(kOpTraceLog, pfx, "MESH | ", label, " | ", row.str());
    return;
  }
  if (d.shape.size() == 2) {
    const unsigned R = std::min<unsigned>(d.shape[0], maxRows);
    const unsigned C = std::min<unsigned>(d.shape[1], maxCols);
    for (unsigned r = 0; r < R; ++r) {
      std::ostringstream row;
      row << "r" << r << " [";
      for (unsigned c = 0; c < C; ++c) {
        if (c) {
          row << ", ";
        }
        row << fmtElem(static_cast<size_t>(r) * d.shape[1] + c);
      }
      if (d.shape[1] > C) {
        row << ", ...";
      }
      row << "]";
      LOG_INFO(kOpTraceLog, pfx, "MESH | ", label, " | ", row.str());
    }
    if (d.shape[0] > R) {
      LOG_INFO(kOpTraceLog, pfx, "MESH | ", label, " | ... (truncated rows)");
    }
    return;
  }
  LOG_INFO(kOpTraceLog, pfx, "MESH | ", label,
           " | preview=rank>2_use_corner_vol=", d.volume());
  const unsigned nshow = std::min<unsigned>(
      static_cast<unsigned>(d.volume()), maxRows * maxCols);
  std::ostringstream os;
  os << "flat[0:" << nshow << ") [";
  for (unsigned i = 0; i < nshow; ++i) {
    if (i) {
      os << ", ";
    }
    os << fmtElem(i);
  }
  os << "]";
  LOG_INFO(kOpTraceLog, pfx, "MESH | ", label, " | ", os.str());
}

std::vector<::tt::runtime::Tensor>
hostShardsFromPoolRef(ProgramContext *programContext,
                      const ::tt::target::ttnn::TensorRef *tensorRefPtr) {
  std::vector<::tt::runtime::Tensor> out;
  if (!programContext || !tensorRefPtr) {
    return out;
  }
  ProgramTensorPool &tensorPool = programContext->getTensorPool();
  if (!tensorPool.contains(tensorRefPtr)) {
    return out;
  }
  ::tt::runtime::Tensor rt = utils::createRuntimeTensorFromTTNN(
      tensorPool.getTTNNTensorAndValidate(tensorRefPtr));
  return ::tt::runtime::ttnn::toHost(rt, /*untilize=*/true);
}

bool tensorMatchesShardOffsets(const ::tt::runtime::TensorDesc &inD,
                               const uint8_t *inBase,
                               const ::tt::runtime::TensorDesc &outD,
                               const uint8_t *outBase,
                               const std::vector<unsigned> &shardDims,
                               const std::vector<size_t> &sliceIndexPerDim) {
  for (size_t fo = 0; fo < outD.volume(); ++fo) {
    std::vector<uint32_t> oc;
    unravelRowMajor(fo, outD.shape, oc);
    std::vector<uint32_t> ic = oc;
    for (size_t k = 0; k < shardDims.size(); ++k) {
      const unsigned d = shardDims[k];
      ic[d] = static_cast<uint32_t>(
          static_cast<size_t>(oc[d]) +
          sliceIndexPerDim[k] * static_cast<size_t>(outD.shape[d]));
    }
    const size_t inOff = linearElemOffset(ic, inD);
    const size_t outOff = linearElemOffset(oc, outD);
    if (std::memcmp(inBase + inOff * inD.itemsize,
                    outBase + outOff * outD.itemsize,
                    inD.itemsize) != 0) {
      return false;
    }
  }
  return true;
}

void logFbI64VecAsMeshAttr(const std::string &pfx, const char *name,
                           const flatbuffers::Vector<int64_t> *v) {
  std::ostringstream o;
  o << name << "=[";
  if (v) {
    for (flatbuffers::uoffset_t i = 0; i < v->size(); ++i) {
      if (i) {
        o << ",";
      }
      o << (*v)[i];
    }
  }
  o << "]";
  LOG_INFO(kOpTraceLog, pfx, "MESH | ", o.str());
}

void logMeshShardFullToShardDetail(
    const ::tt::target::ttnn::MeshShardOp *ms, ProgramContext *programContext,
    const std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> &outMap,
    const std::string &pfx) {
  if (!ms || ms->shard_direction() !=
                 ::tt::target::MeshShardDirection::FullToShardShape) {
    return;
  }
  logFbI64VecAsMeshAttr(pfx, "shard_shape", ms->shard_shape());
  logFbI64VecAsMeshAttr(pfx, "shard_dims", ms->shard_dims());

  std::vector<::tt::runtime::Tensor> inHosts =
      hostShardsFromPoolRef(programContext, ms->in());
  if (inHosts.empty()) {
    LOG_WARNING(kOpTraceLog, pfx, "MESH | WARN | mesh_shard no host input read");
    return;
  }
  size_t best = 0;
  size_t bestVol = 0;
  for (size_t i = 0; i < inHosts.size(); ++i) {
    const size_t v = getTensorDesc(inHosts[i]).volume();
    if (v > bestVol) {
      bestVol = v;
      best = i;
    }
  }
  ::tt::runtime::Tensor &globalIn = inHosts[best];
  ::tt::runtime::TensorDesc inDesc = getTensorDesc(globalIn);
  const ::ttnn::Tensor &inTt =
      utils::getTTNNTensorFromRuntimeTensor(globalIn);
  std::vector<uint8_t> inBuf(static_cast<size_t>(inTt.physical_volume()) *
                             static_cast<size_t>(inTt.element_size()));
  ::tt::runtime::ttnn::memcpy(inBuf.data(), globalIn);
  const auto inFnvAbs = hostTensorFnvAndAbsSum(globalIn);
  if (opTensorTraceMeshTensorPreviewsEnabled()) {
    logTensorPreview(pfx, "full_input (global)", globalIn, 8, 16);
  }
  LOG_INFO(kOpTraceLog, pfx, "MESH | full_input | host_copies=", inHosts.size(),
           " | using_copy=", best, " | fnv1a64=", inFnvAbs.first,
           " | abs_sum=", inFnvAbs.second);

  if (outMap.empty()) {
    return;
  }
  const ::tt::runtime::Tensor &sampleOut = outMap.begin()->second;
  ::tt::runtime::TensorDesc outDesc = getTensorDesc(sampleOut);
  std::vector<unsigned> shardDims;
  for (unsigned d = 0; d < inDesc.shape.size(); ++d) {
    if (inDesc.shape[d] != outDesc.shape[d]) {
      if (outDesc.shape[d] == 0 ||
          inDesc.shape[d] % outDesc.shape[d] != 0) {
        LOG_WARNING(kOpTraceLog, pfx, "MESH | WARN | shape_mismatch_dim=", d,
                    " in=", inDesc.shape[d], " out=", outDesc.shape[d]);
        continue;
      }
      shardDims.push_back(d);
    }
  }
  if (shardDims.empty()) {
    LOG_INFO(kOpTraceLog, pfx,
             "MESH | no differing dims vs first output; printing per-device outs");
  }
  size_t numUniqueSlices = 1;
  for (unsigned d : shardDims) {
    numUniqueSlices *= static_cast<size_t>(inDesc.shape[d] / outDesc.shape[d]);
  }

  std::map<std::vector<uint8_t>, std::vector<uint32_t>> fingerprintToDevs;
  double sumAllDeviceOuts = 0.0;
  bool sumOk = true;
  for (const auto &kv : outMap) {
    const auto outFs = hostTensorFnvAndAbsSum(kv.second);
    sumAllDeviceOuts += outFs.second;
    if (std::isnan(outFs.second)) {
      sumOk = false;
    }
    const ::ttnn::Tensor &ott =
        utils::getTTNNTensorFromRuntimeTensor(kv.second);
    std::vector<uint8_t> obuf(static_cast<size_t>(ott.physical_volume()) *
                              static_cast<size_t>(ott.element_size()));
    ::tt::runtime::ttnn::memcpy(obuf.data(), kv.second);
    fingerprintToDevs[obuf].push_back(kv.first);
  }

  LOG_INFO(kOpTraceLog, pfx, "MESH | partition | inferred_shard_dims=",
           shardDims.size(), " | num_unique_slices~=", numUniqueSlices,
           " | devices=", outMap.size(),
           " | distinct_output_buffers=", fingerprintToDevs.size());

  std::vector<uint8_t> asciiBar(std::min<size_t>(inDesc.volume(), 256), '.');
  if (shardDims.size() == 1) {
    const unsigned sd = shardDims[0];
    const size_t S = static_cast<size_t>(inDesc.shape[sd] / outDesc.shape[sd]);
    const size_t chunk = outDesc.shape[sd];
    if (sd == 0 && inDesc.shape.size() == 1 &&
        inDesc.shape[0] <= asciiBar.size()) {
      for (size_t s = 0; s < S; ++s) {
        const char mark =
            static_cast<char>('A' + static_cast<int>(s % 26));
        for (size_t j = 0; j < chunk; ++j) {
          asciiBar[s * chunk + j] = static_cast<uint8_t>(mark);
        }
      }
      std::string bar;
      bar.reserve(asciiBar.size());
      for (uint8_t b : asciiBar) {
        bar += static_cast<char>(b);
      }
      LOG_INFO(kOpTraceLog, pfx,
               "MESH | input_dim0_highlight | legend=A.. per slice | ", bar);
    } else {
      LOG_INFO(kOpTraceLog, pfx,
               "MESH | input_highlight | dim=", sd, " | num_slices=", S,
               " | slice_len=", chunk,
               " | (1D bar only when rank==1 and dim0 sharded and len<=",
               asciiBar.size(), ")");
    }
  }

  std::unordered_set<size_t> seenSingleSlices;
  std::unordered_set<size_t> seenMultiLinear;
  double sumUniqueOnce = 0.0;
  for (const auto &kv : outMap) {
    const ::tt::runtime::Tensor &devOut = kv.second;
    ::tt::runtime::TensorDesc od = getTensorDesc(devOut);
    const ::ttnn::Tensor &dtt = utils::getTTNNTensorFromRuntimeTensor(devOut);
    std::vector<uint8_t> dbuf(static_cast<size_t>(dtt.physical_volume()) *
                              static_cast<size_t>(dtt.element_size()));
    ::tt::runtime::ttnn::memcpy(dbuf.data(), devOut);
    const auto outFnvAbs = hostTensorFnvAndAbsSum(devOut);

    if (opTensorTraceMeshTensorPreviewsEnabled()) {
      logTensorPreview(pfx, ("device_out dev=" + std::to_string(kv.first)).c_str(),
                       devOut, 6, 12);
    }
    LOG_INFO(kOpTraceLog, pfx, "MESH | dev=", kv.first,
             " | out_fnv1a64=", outFnvAbs.first,
             " | out_abs_sum=", outFnvAbs.second);

    if (shardDims.empty()) {
      continue;
    }
    bool found = false;
    if (shardDims.size() == 1) {
      const unsigned sd = shardDims[0];
      const size_t S = inDesc.shape[sd] / outDesc.shape[sd];
      for (size_t s = 0; s < S; ++s) {
        if (tensorMatchesShardOffsets(inDesc, inBuf.data(), od, dbuf.data(),
                                      {sd}, {s})) {
          LOG_INFO(kOpTraceLog, pfx, "MESH | dev=", kv.first,
                   " | EXPECT_SLICE dim=", sd, " index=", s, " | range [",
                   s * od.shape[sd], ":", (s + 1) * od.shape[sd], ") | ",
                   "cmp_bytes=PASS");
          if (seenSingleSlices.insert(s).second) {
            sumUniqueOnce += outFnvAbs.second;
          }
          found = true;
          break;
        }
      }
    } else {
      std::vector<size_t> factors;
      factors.reserve(shardDims.size());
      size_t total = 1;
      for (unsigned sd : shardDims) {
        const size_t f = inDesc.shape[sd] / od.shape[sd];
        factors.push_back(f);
        total *= f;
      }
      for (size_t t = 0; t < total; ++t) {
        size_t x = t;
        std::vector<size_t> sliceIdx(shardDims.size());
        for (size_t k = 0; k < shardDims.size(); ++k) {
          sliceIdx[k] = x % factors[k];
          x /= factors[k];
        }
        if (tensorMatchesShardOffsets(inDesc, inBuf.data(), od, dbuf.data(),
                                      shardDims, sliceIdx)) {
          std::ostringstream desc;
          desc << "multi_idx(t=" << t << ")";
          for (size_t k = 0; k < shardDims.size(); ++k) {
            desc << " d" << shardDims[k] << "=" << sliceIdx[k];
          }
          LOG_INFO(kOpTraceLog, pfx, "MESH | dev=", kv.first, " | EXPECT_SLICE ",
                   desc.str(), " | cmp_bytes=PASS");
          if (seenMultiLinear.insert(t).second) {
            sumUniqueOnce += outFnvAbs.second;
          }
          found = true;
          break;
        }
      }
    }
    if (!found) {
      LOG_WARNING(kOpTraceLog, pfx, "MESH | dev=", kv.first,
                  " | EXPECT_SLICE | cmp_bytes=NO_MATCH");
    }
  }

  if (!shardDims.empty() && !std::isnan(inFnvAbs.second) && sumOk) {
    const double rep =
        outMap.empty() ? 1.0
                       : static_cast<double>(outMap.size()) /
                             static_cast<double>(numUniqueSlices);
    const double sumTol =
        1e-1 * static_cast<double>(std::max<size_t>(1, inDesc.volume())) +
        1e-2 * inFnvAbs.second + 1.0;
    const bool sumUniquePass =
        std::abs(sumUniqueOnce - inFnvAbs.second) <= sumTol;
    const bool sumAllPass =
        std::abs(sumAllDeviceOuts - inFnvAbs.second * rep) <=
        sumTol * std::max(1.0, rep);
    LOG_INFO(kOpTraceLog, pfx, "MESH | CHECK | abs_sum(input)=", inFnvAbs.second,
             " | abs_sum(unique_matched_out)=", sumUniqueOnce,
             " | unique_vs_input=", (sumUniquePass ? "PASS" : "FAIL"),
             " | abs_sum(all_device_outs)=", sumAllDeviceOuts,
             " | repl_factor~=", rep,
             " | all_vs_input*repl=", (sumAllPass ? "PASS" : "FAIL"));
  }
}

void logMeshPartitionDetail(
    const ::tt::target::ttnn::MeshPartitionOp *mp,
    ProgramContext *programContext,
    const std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> &outMap,
    const std::string &pfx) {
  if (!mp) {
    return;
  }
  const int32_t dim = mp->dim();
  const std::optional<uint32_t> clusterAxis = mp->cluster_axis();

  std::vector<::tt::runtime::Tensor> inHosts =
      hostShardsFromPoolRef(programContext, mp->in());
  if (inHosts.empty() || outMap.empty()) {
    LOG_WARNING(kOpTraceLog, pfx, "MESH | mesh_partition | missing host IO");
    return;
  }
  size_t best = 0;
  size_t bestVol = 0;
  for (size_t i = 0; i < inHosts.size(); ++i) {
    const size_t v = getTensorDesc(inHosts[i]).volume();
    if (v > bestVol) {
      bestVol = v;
      best = i;
    }
  }
  ::tt::runtime::Tensor &globalIn = inHosts[best];
  ::tt::runtime::TensorDesc inDesc = getTensorDesc(globalIn);
  const ::ttnn::Tensor &inTt =
      utils::getTTNNTensorFromRuntimeTensor(globalIn);
  std::vector<uint8_t> inBuf(static_cast<size_t>(inTt.physical_volume()) *
                             static_cast<size_t>(inTt.element_size()));
  ::tt::runtime::ttnn::memcpy(inBuf.data(), globalIn);
  const HostTensorNumericDigest inDig = hostTensorNumericDigest(globalIn);

  if (opTensorTraceMeshTensorPreviewsEnabled()) {
    logTensorPreview(pfx, "mesh_partition full_input", globalIn, 8, 16);
  }
  {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "MESH | mesh_partition | dim=" << dim << " | cluster_axis="
       << (clusterAxis.has_value() ? std::to_string(*clusterAxis) : "null")
       << " | global_shape=" << shapeToString(inDesc.shape)
       << " | input_fnv=" << inDig.fnv << " input_|Σ|=" << inDig.absSum;
    LOG_INFO(kOpTraceLog, pfx, os.str());
  }

  const ::tt::runtime::Tensor &sampleOut = outMap.begin()->second;
  ::tt::runtime::TensorDesc outDesc = getTensorDesc(sampleOut);
  const uint32_t ndev = static_cast<uint32_t>(outMap.size());
  if (dim < 0 || static_cast<size_t>(dim) >= inDesc.shape.size()) {
    LOG_WARNING(kOpTraceLog, pfx, "MESH | mesh_partition | bad dim");
    return;
  }
  const size_t inDim = static_cast<size_t>(dim);
  const size_t chunk = outDesc.shape[inDim];
  const size_t numSlices =
      (chunk > 0 && inDesc.shape[inDim] % chunk == 0)
          ? static_cast<size_t>(inDesc.shape[inDim] / chunk)
          : 0;

  {
    std::ostringstream plan;
    plan << "MESH | mesh_partition PLAN | partition_dim=" << inDim
         << " | local_shard_extent=" << chunk
         << " | num_disjoint_slices=" << numSlices
         << " | mesh_ranks=" << ndev
         << " | linear_reference: slice s uses global indices dim " << inDim
         << " in [" << chunk << "*s, " << chunk << "*(s+1))"
         << " (compare dev_id==s when layout is 1:1)";
    LOG_INFO(kOpTraceLog, pfx, plan.str());
  }

  std::vector<double> sliceAbs;
  sliceAbs.reserve(numSlices);
  double sumSliceTable = 0.0;
  for (size_t s = 0; s < numSlices; ++s) {
    const double e = sliceAbsSumRowMajor(
        inDesc, inBuf.data(), static_cast<unsigned>(inDim), s * chunk, chunk);
    sliceAbs.push_back(e);
    sumSliceTable += e;
    std::ostringstream sl;
    sl << std::fixed << std::setprecision(6);
    sl << "MESH | mesh_partition EXPECT | slice=" << s << " | dim" << inDim
       << "_range=[" << (s * chunk) << ":" << ((s + 1) * chunk)
       << ") | slice_|Σ|=" << e;
    LOG_INFO(kOpTraceLog, pfx, sl.str());
  }

  const double tableTol =
      1e-2 * std::max(1.0, inDig.absSum) +
      1e-3 * static_cast<double>(std::max<size_t>(1, inDesc.volume()));
  const bool tableVsInput =
      !std::isnan(inDig.absSum) && !std::isnan(sumSliceTable) &&
      std::abs(sumSliceTable - inDig.absSum) <= tableTol;

  {
    std::ostringstream os;
    os << std::fixed << std::setprecision(8);
    os << "MESH | mesh_partition CHECK | input_|Σ|=" << inDig.absSum
       << " | |Σ|(slice_table)=" << sumSliceTable
       << " | slice_table_vs_input=" << (tableVsInput ? "PASS" : "FAIL");
    LOG_INFO(kOpTraceLog, pfx, os.str());
  }

  double sumOutAll = 0.0;
  for (const auto &kv : outMap) {
    const HostTensorNumericDigest od = hostTensorNumericDigest(kv.second);
    if (!std::isnan(od.absSum)) {
      sumOutAll += od.absSum;
    }
    if (opTensorTraceMeshTensorPreviewsEnabled()) {
      logTensorPreview(
          pfx, ("mesh_partition out dev=" + std::to_string(kv.first)).c_str(),
          kv.second, 6, 12);
    }
    const ::ttnn::Tensor &ott =
        utils::getTTNNTensorFromRuntimeTensor(kv.second);
    std::vector<uint8_t> dbuf(static_cast<size_t>(ott.physical_volume()) *
                              static_cast<size_t>(ott.element_size()));
    ::tt::runtime::ttnn::memcpy(dbuf.data(), kv.second);
    bool ok = false;
    size_t matchedSlice = std::numeric_limits<size_t>::max();
    if (numSlices > 0) {
      for (size_t s = 0; s < numSlices; ++s) {
        if (tensorMatchesShardOffsets(inDesc, inBuf.data(), outDesc, dbuf.data(),
                                      {static_cast<unsigned>(inDim)}, {s})) {
          ok = true;
          matchedSlice = s;
          break;
        }
      }
    }
    std::string rangeStr = "unknown";
    if (matchedSlice != std::numeric_limits<size_t>::max()) {
      rangeStr = "[" + std::to_string(matchedSlice * chunk) + ":" +
                 std::to_string((matchedSlice + 1) * chunk) + ")";
    }

    double linearExpect = std::numeric_limits<double>::quiet_NaN();
    if (kv.first < numSlices && kv.first < sliceAbs.size()) {
      linearExpect = sliceAbs[kv.first];
    }
    const bool linearOk =
        !std::isnan(linearExpect) && !std::isnan(od.absSum) &&
        std::abs(od.absSum - linearExpect) <= tableTol;

    {
      std::ostringstream dl;
      dl << std::fixed << std::setprecision(6);
      dl << "MESH | mesh_partition dev=" << kv.first
         << " | bytes_slice=" << (ok ? "PASS" : "FAIL")
         << " | matched_slice_idx="
         << (matchedSlice != std::numeric_limits<size_t>::max()
                 ? std::to_string(matchedSlice)
                 : std::string("none"))
         << " | range_dim" << inDim << "=" << rangeStr;
      if (!std::isnan(linearExpect)) {
        dl << " | linear_model expect_|Σ|(slice " << kv.first << ")=" << linearExpect
           << " vs out_|Σ|=" << od.absSum << " " << (linearOk ? "PASS" : "FAIL");
      } else {
        dl << " | linear_model expect_|Σ|=n/a (dev>=" << numSlices
           << ") out_|Σ|=" << od.absSum;
      }
      dl << " | fnv=" << od.fnv;
      LOG_INFO(kOpTraceLog, pfx, dl.str());
    }
  }

  if (!std::isnan(inDig.absSum) && !std::isnan(sumOutAll) && ndev > 0 &&
      numSlices > 0) {
    const double repl =
        static_cast<double>(ndev) / static_cast<double>(numSlices);
    const bool replInt = std::abs(repl - std::round(repl)) < 1e-6;
    const double pred =
        replInt ? inDig.absSum * std::round(repl) : std::numeric_limits<double>::quiet_NaN();
    const bool replPass =
        replInt && !std::isnan(pred) &&
        std::abs(sumOutAll - pred) <= tableTol * std::max(1.0, std::round(repl));
    std::ostringstream os;
    os << std::fixed << std::setprecision(8);
    os << "MESH | mesh_partition CHECK | sum_all_dev_out_|Σ|=" << sumOutAll
       << " | input_|Σ|=" << inDig.absSum << " | ranks/slices=" << ndev << "/"
       << numSlices << " | repl~=" << repl;
    if (replInt) {
      os << " | if_full_replica_each_slice expect_out_total_|Σ|~=" << pred << " "
         << (replPass ? "PASS" : "FAIL");
    }
    LOG_INFO(kOpTraceLog, pfx, os.str());
  }
}

std::string getFullTensorDumpPath() {
  const char *logFile = std::getenv("TTMLIR_RUNTIME_LOGGER_FILE");
  if (!logFile || !logFile[0]) {
    return "";
  }
  std::string path(logFile);
  auto dotPos = path.rfind(".log");
  if (dotPos != std::string::npos) {
    path.replace(dotPos, 4, "_full_tensors.log");
  } else {
    path += "_full_tensors";
  }
  return path;
}

void dumpFullTensorValues(
    const std::string &pfx, const char *label,
    const std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> &perDev) {
  std::string dumpPath = getFullTensorDumpPath();
  if (dumpPath.empty() || perDev.empty()) {
    return;
  }
  std::ofstream ofs(dumpPath, std::ios::app);
  if (!ofs) {
    LOG_WARNING(kOpTraceLog, pfx, "DUMP | failed to open ", dumpPath);
    return;
  }

  ofs << "\n=== " << pfx << label << " ===\n";
  for (const auto &kv : perDev) {
    ::tt::runtime::TensorDesc d = getTensorDesc(kv.second);
    const ::ttnn::Tensor &tt =
        utils::getTTNNTensorFromRuntimeTensor(kv.second);
    const size_t nbytes =
        static_cast<size_t>(tt.physical_volume()) *
        static_cast<size_t>(tt.element_size());
    if (nbytes == 0) {
      ofs << "device=" << kv.first << " (empty)\n";
      continue;
    }
    std::vector<uint8_t> buf(nbytes);
    ::tt::runtime::ttnn::memcpy(buf.data(), kv.second);

    ofs << "device=" << kv.first << " shape=" << shapeToString(d.shape)
        << " dt=" << static_cast<int>(d.dataType) << "\n";

    if (d.itemsize != 2 && d.itemsize != 4) {
      ofs << "  (unsupported itemsize=" << d.itemsize << ")\n";
      continue;
    }

    auto fmtVal = [&](size_t linearIdx) {
      std::vector<uint32_t> coord;
      unravelRowMajor(linearIdx, d.shape, coord);
      const size_t off = linearElemOffset(coord, d);
      if (d.itemsize == 2) {
        uint16_t u = 0;
        std::memcpy(&u, buf.data() + off * d.itemsize, 2);
        ofs << std::setprecision(6) << bf16BitsToFloat(u);
      } else {
        float f = 0.0f;
        std::memcpy(&f, buf.data() + off * d.itemsize, sizeof(f));
        ofs << std::setprecision(6) << f;
      }
    };

    if (d.shape.size() == 2) {
      for (uint32_t r = 0; r < d.shape[0]; ++r) {
        ofs << "  [";
        for (uint32_t c = 0; c < d.shape[1]; ++c) {
          if (c) {
            ofs << ", ";
          }
          fmtVal(static_cast<size_t>(r) * d.shape[1] + c);
        }
        ofs << "]\n";
      }
    } else {
      ofs << "  [";
      for (size_t li = 0; li < d.volume(); ++li) {
        if (li) {
          ofs << ", ";
        }
        fmtVal(li);
      }
      ofs << "]\n";
    }
  }
  ofs << "\n";
  LOG_INFO(kOpTraceLog, pfx, "DUMP | full tensor ", label, " → ", dumpPath);
}

std::string getTopKDumpDir() {
  const char *dir = std::getenv("TT_RUNTIME_OP_TENSOR_TRACE_TOPK_DUMP_DIR");
  if (!dir || !dir[0]) {
    return "";
  }
  return std::string(dir);
}

bool ensureDirExists(const std::string &path) {
  struct stat st {};
  if (stat(path.c_str(), &st) == 0) {
    return S_ISDIR(st.st_mode);
  }
  auto pos = path.rfind('/');
  if (pos != std::string::npos && pos > 0) {
    ensureDirExists(path.substr(0, pos));
  }
  return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

bool writeNpyFile(const std::string &path,
                  const std::vector<uint32_t> &shape,
                  const std::string &descr,
                  const void *data, size_t dataBytes) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    return false;
  }
  std::ostringstream hdr;
  hdr << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) {
      hdr << ", ";
    }
    hdr << shape[i];
  }
  if (shape.size() == 1) {
    hdr << ",";
  }
  hdr << "), }";
  std::string hdrStr = hdr.str();

  constexpr size_t kPrelude = 10;
  size_t totalHdrLen = hdrStr.size() + 1;
  size_t padding = (64 - (kPrelude + totalHdrLen) % 64) % 64;
  totalHdrLen += padding;

  const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
  ofs.write(reinterpret_cast<const char *>(magic), 8);

  auto hlen = static_cast<uint16_t>(totalHdrLen);
  ofs.write(reinterpret_cast<const char *>(&hlen), 2);

  ofs.write(hdrStr.data(), static_cast<std::streamsize>(hdrStr.size()));
  for (size_t i = 0; i < padding; ++i) {
    ofs.put(' ');
  }
  ofs.put('\n');

  ofs.write(static_cast<const char *>(data),
            static_cast<std::streamsize>(dataBytes));
  return ofs.good();
}

bool writeHostTensorAsNpy(const std::string &path,
                          const ::tt::runtime::Tensor &hostTensor) {
  ::tt::runtime::TensorDesc d = getTensorDesc(hostTensor);
  const ::ttnn::Tensor &tt =
      utils::getTTNNTensorFromRuntimeTensor(hostTensor);
  const size_t nbytes =
      static_cast<size_t>(tt.physical_volume()) *
      static_cast<size_t>(tt.element_size());
  if (nbytes == 0 || d.volume() == 0) {
    return false;
  }

  std::vector<uint8_t> rawBuf(nbytes);
  ::tt::runtime::ttnn::memcpy(rawBuf.data(), hostTensor);

  bool isIntType = false;
  switch (d.dataType) {
  case ::tt::target::DataType::Int32:
  case ::tt::target::DataType::UInt32:
  case ::tt::target::DataType::UInt16:
  case ::tt::target::DataType::UInt8:
    isIntType = true;
    break;
  default:
    break;
  }

  const std::string descr = isIntType ? "<i4" : "<f4";
  std::vector<uint8_t> npyData(d.volume() * 4);

  for (size_t li = 0; li < d.volume(); ++li) {
    std::vector<uint32_t> coord;
    unravelRowMajor(li, d.shape, coord);
    const size_t off = linearElemOffset(coord, d);

    if (isIntType) {
      int32_t val = 0;
      if (d.itemsize == 4) {
        std::memcpy(&val, rawBuf.data() + off * d.itemsize, 4);
      } else if (d.itemsize == 2) {
        uint16_t u = 0;
        std::memcpy(&u, rawBuf.data() + off * d.itemsize, 2);
        val = static_cast<int32_t>(u);
      } else if (d.itemsize == 1) {
        val = static_cast<int32_t>(rawBuf[off]);
      }
      std::memcpy(npyData.data() + li * 4, &val, 4);
    } else {
      float f = 0.0f;
      if (d.itemsize == 2) {
        uint16_t u = 0;
        std::memcpy(&u, rawBuf.data() + off * d.itemsize, 2);
        f = bf16BitsToFloat(u);
      } else if (d.itemsize == 4) {
        std::memcpy(&f, rawBuf.data() + off * d.itemsize, 4);
      }
      std::memcpy(npyData.data() + li * 4, &f, 4);
    }
  }

  return writeNpyFile(path, d.shape, descr, npyData.data(), npyData.size());
}

void dumpTopKTensorsAsNpy(const ::tt::target::ttnn::Operation *op,
                          ProgramContext *programContext,
                          const std::string &pfx, size_t opSeq) {
  std::string dumpDir = getTopKDumpDir();
  if (dumpDir.empty()) {
    return;
  }

  const auto *topKOp = op->type_as_TopKOp();
  if (!topKOp) {
    return;
  }

  if (!ensureDirExists(dumpDir)) {
    LOG_WARNING(kOpTraceLog, pfx,
                "TOPK_DUMP | failed to create dir: ", dumpDir);
    return;
  }

  const std::string base =
      dumpDir + "/topk_" + std::to_string(opSeq);

  const auto *inRef = topKOp->input_tensor();
  if (inRef) {
    std::vector<::tt::runtime::Tensor> inHosts =
        hostShardsFromPoolRef(programContext, inRef);
    for (size_t i = 0; i < inHosts.size(); ++i) {
      std::string p =
          base + "_input_dev" + std::to_string(i) + ".npy";
      if (writeHostTensorAsNpy(p, inHosts[i])) {
        LOG_INFO(kOpTraceLog, pfx, "TOPK_DUMP | wrote input dev=", i,
                 " → ", p);
      }
    }
  }

  if (!topKOp->outputs()) {
    return;
  }
  const char *outputNames[] = {"values", "indices"};
  for (size_t outIdx = 0; outIdx < topKOp->outputs()->size(); ++outIdx) {
    const auto *outRef = topKOp->outputs()->Get(
        static_cast<flatbuffers::uoffset_t>(outIdx));
    if (!outRef) {
      continue;
    }
    std::vector<::tt::runtime::Tensor> hosts =
        hostShardsFromPoolRef(programContext, outRef);
    const char *name = outIdx < 2 ? outputNames[outIdx] : "output";
    for (size_t i = 0; i < hosts.size(); ++i) {
      std::string p = base + "_" + name + "_dev" + std::to_string(i) + ".npy";
      if (writeHostTensorAsNpy(p, hosts[i])) {
        LOG_INFO(kOpTraceLog, pfx, "TOPK_DUMP | wrote ", name, " dev=", i,
                 " → ", p);
      }
    }
  }

  std::string loc = extractShortLoc(
      op->loc_info() ? op->loc_info()->c_str() : nullptr);
  std::string mlirOp = extractMlirTtnnOpName(
      op->debug_info() ? op->debug_info()->c_str() : nullptr);

  std::string metaPath = base + "_meta.json";
  std::ofstream meta(metaPath);
  if (meta) {
    meta << "{\n  \"op_seq\": " << opSeq
         << ",\n  \"k\": " << topKOp->k()
         << ",\n  \"dim\": " << topKOp->dim()
         << ",\n  \"largest\": " << (topKOp->largest() ? "true" : "false")
         << ",\n  \"sorted\": " << (topKOp->sorted() ? "true" : "false")
         << ",\n  \"loc\": \"" << escapeJsonString(loc) << "\""
         << ",\n  \"mlir_op\": \"" << escapeJsonString(mlirOp) << "\"";
    if (inRef) {
      meta << ",\n  \"in_global_id\": " << inRef->global_id();
    }
    meta << "\n}\n";
    LOG_INFO(kOpTraceLog, pfx, "TOPK_DUMP | wrote meta → ", metaPath);
  }
}

void dumpTypecastTensorsAsNpy(const ::tt::target::ttnn::Operation *op,
                              ProgramContext *programContext,
                              const std::string &pfx, size_t opSeq) {
  std::string dumpDir = getTopKDumpDir();
  if (dumpDir.empty()) {
    return;
  }

  const auto *typecastOp = op->type_as_TypecastOp();
  if (!typecastOp) {
    return;
  }

  if (!ensureDirExists(dumpDir)) {
    LOG_WARNING(kOpTraceLog, pfx,
                "TYPECAST_DUMP | failed to create dir: ", dumpDir);
    return;
  }

  const std::string base =
      dumpDir + "/typecast_" + std::to_string(opSeq);

  const auto *inRef = typecastOp->in();
  if (inRef) {
    std::vector<::tt::runtime::Tensor> inHosts =
        hostShardsFromPoolRef(programContext, inRef);
    for (size_t i = 0; i < inHosts.size(); ++i) {
      std::string p =
          base + "_input_dev" + std::to_string(i) + ".npy";
      if (writeHostTensorAsNpy(p, inHosts[i])) {
        LOG_INFO(kOpTraceLog, pfx, "TYPECAST_DUMP | wrote input dev=", i,
                 " → ", p);
      }
    }
  }

  const auto *outRef = typecastOp->out();
  if (outRef) {
    std::vector<::tt::runtime::Tensor> outHosts =
        hostShardsFromPoolRef(programContext, outRef);
    for (size_t i = 0; i < outHosts.size(); ++i) {
      std::string p =
          base + "_output_dev" + std::to_string(i) + ".npy";
      if (writeHostTensorAsNpy(p, outHosts[i])) {
        LOG_INFO(kOpTraceLog, pfx, "TYPECAST_DUMP | wrote output dev=", i,
                 " → ", p);
      }
    }
  }

  std::string loc = extractShortLoc(
      op->loc_info() ? op->loc_info()->c_str() : nullptr);
  std::string mlirOp = extractMlirTtnnOpName(
      op->debug_info() ? op->debug_info()->c_str() : nullptr);

  std::string metaPath = base + "_meta.json";
  std::ofstream meta(metaPath);
  if (meta) {
    meta << "{\n  \"op_seq\": " << opSeq
         << ",\n  \"target_dtype\": " << static_cast<int>(typecastOp->dtype())
         << ",\n  \"loc\": \"" << escapeJsonString(loc) << "\""
         << ",\n  \"mlir_op\": \"" << escapeJsonString(mlirOp) << "\"";
    if (inRef) {
      meta << ",\n  \"in_global_id\": " << inRef->global_id();
    }
    if (outRef) {
      meta << ",\n  \"out_global_id\": " << outRef->global_id();
    }
    meta << "\n}\n";
    LOG_INFO(kOpTraceLog, pfx, "TYPECAST_DUMP | wrote meta → ", metaPath);
  }
}

void logTopKOutputs(const ::tt::target::ttnn::Operation *op,
                    ProgramContext *programContext,
                    const std::string &pfx) {
  const auto *topKOp = op->type_as_TopKOp();
  if (!topKOp || !topKOp->outputs()) {
    LOG_WARNING(kOpTraceLog, pfx, "OUT | TopKOp | no outputs in flatbuffer");
    return;
  }

  const size_t numOutputs = topKOp->outputs()->size();
  const char *outputNames[] = {"values", "indices"};

  for (size_t outIdx = 0; outIdx < numOutputs; ++outIdx) {
    const auto *outRef = topKOp->outputs()->Get(
        static_cast<flatbuffers::uoffset_t>(outIdx));
    if (!outRef) {
      continue;
    }

    std::vector<::tt::runtime::Tensor> hosts =
        hostShardsFromPoolRef(programContext, outRef);

    std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> outMap;
    for (size_t i = 0; i < hosts.size(); ++i) {
      outMap[static_cast<std::uint32_t>(i)] = std::move(hosts[i]);
    }

    const char *name = outIdx < 2 ? outputNames[outIdx] : "?";

    if (outMap.empty()) {
      LOG_WARNING(kOpTraceLog, pfx, "OUT[", outIdx, "] | TopK ", name,
                  " | empty_after_host_read");
      continue;
    }

    std::vector<std::uint32_t> zeroDev, nonzeroDev;
    std::string briefStr;
    for (const auto &kv : outMap) {
      std::optional<bool> oz = hostTensorAllZero(kv.second);
      briefStr = tensorBrief(kv.second);
      if (!oz.has_value() || !*oz) {
        nonzeroDev.push_back(kv.first);
      } else {
        zeroDev.push_back(kv.first);
      }
    }

    {
      std::ostringstream outLine;
      outLine << "OUT[" << outIdx << "] " << name;
      if (!nonzeroDev.empty()) {
        outLine << " | ok | " << nonzeroDev.size() << "/" << outMap.size()
                << " | " << briefStr << " | dev="
                << formatDeviceIds(nonzeroDev);
      }
      if (!zeroDev.empty()) {
        outLine << " | ZERO=" << zeroDev.size();
      }
      outLine << " | global_id=" << outRef->global_id();
      LOG_INFO(kOpTraceLog, pfx, outLine.str());
    }

    std::string sumLabel =
        std::string("OUT[") + std::to_string(outIdx) + "] " + name;
    logGroupedPerDeviceSums(pfx, sumLabel.c_str(), outMap);
    dumpFullTensorValues(pfx, sumLabel.c_str(), outMap);
  }
}

} // namespace

bool opTensorTraceEnvEnabled() {
  static const bool on = envFlag(std::getenv("TT_RUNTIME_OP_TENSOR_TRACE"));
  return on;
}

std::vector<std::optional<bool>> opTensorTraceCaptureInputZeroState(
    const ::tt::target::ttnn::Operation *op, ProgramContext *programContext) {
  std::vector<std::optional<bool>> out;
  if (!opTensorTraceEnvEnabled() || !op || !programContext) {
    return out;
  }
  CallbackContext cb = makeCb(programContext);
  OpContext oc = makeOp(op);
  std::vector<::tt::runtime::TensorRef> refs = getOpInputRefs(oc, cb);
  out.reserve(refs.size());
  const bool skipHost =
      opTensorTraceFastMode() && !opTensorTraceWantsPerDeviceHeavy(op);
  if (skipHost) {
    for (size_t i = 0; i < refs.size(); ++i) {
      out.push_back(std::nullopt);
    }
    return out;
  }
  for (const ::tt::runtime::TensorRef &ref : refs) {
    out.push_back(poolRefAllShardsZero(cb, ref));
  }
  return out;
}

void opTensorTraceLogCompletedOp(
    const ::tt::target::ttnn::Operation *op, ProgramContext *programContext,
    const std::vector<std::optional<bool>> &inputAllZero) {
  if (!opTensorTraceEnvEnabled() || !op || !programContext) {
    return;
  }

  CallbackContext cb = makeCb(programContext);
  OpContext oc = makeOp(op);

  static thread_local size_t opSeq = 0;
  const size_t seq = ++opSeq;
  const std::string pfx = std::string("#") + std::to_string(seq) + " | ";

  const bool traceFastSkipHost =
      opTensorTraceFastMode() && !opTensorTraceWantsPerDeviceHeavy(op);

  const char *enumName = ::tt::target::ttnn::EnumNameOpType(op->type_type());
  std::string mlirOp =
      extractMlirTtnnOpName(op->debug_info() ? op->debug_info()->c_str() : nullptr);
  std::string shortLoc =
      extractShortLoc(op->loc_info() ? op->loc_info()->c_str() : nullptr);

  {
    std::ostringstream opLine;
    opLine << pfx << "OP | ";
    if (!mlirOp.empty()) {
      opLine << mlirOp;
      if (enumName) {
        opLine << " | " << enumName;
      }
    } else if (enumName) {
      opLine << enumName;
    } else {
      opLine << "?";
    }
    if (!shortLoc.empty()) {
      opLine << " | loc=" << shortLoc;
    }
    LOG_INFO(kOpTraceLog, opLine.str());
  }

  if (envFlag(std::getenv("TT_RUNTIME_OP_TENSOR_TRACE_VERBOSE"))) {
    if (op->debug_info()) {
      std::string d(op->debug_info()->c_str());
      constexpr size_t kMax = 400;
      if (d.size() > kMax) {
        d.resize(kMax);
        d += "...";
      }
      LOG_INFO(kOpTraceLog, pfx, "MLIR | ", d);
    }
    if (op->loc_info()) {
      LOG_INFO(kOpTraceLog, pfx, "LOC | ", op->loc_info()->c_str());
    }
  }

  std::vector<::tt::runtime::TensorRef> inRefs = getOpInputRefs(oc, cb);
  bool anyInputNonZeroKnown = false;
  for (size_t i = 0; i < inputAllZero.size(); ++i) {
    std::string refPart =
        (i < inRefs.size()) ? refBrief(inRefs[i]) : std::string("ref=?");
    if (!inputAllZero[i].has_value()) {
      LOG_INFO(kOpTraceLog, pfx, "IN | ", i, " | ", refPart,
               " | z=? | ",
               traceFastSkipHost ? "trace_fast_skip" : "err=host_or_pool");
      continue;
    }
    if (*inputAllZero[i]) {
      LOG_INFO(kOpTraceLog, pfx, "IN | ", i, " | ", refPart, " | z=yes");
    } else {
      LOG_INFO(kOpTraceLog, pfx, "IN | ", i, " | ", refPart,
               " | z=no | NONZERO_INPUT");
      anyInputNonZeroKnown = true;
    }
  }
  if (inRefs.size() != inputAllZero.size()) {
    LOG_WARNING(kOpTraceLog, pfx, "WARN | in_ref_mismatch | pool=",
                inRefs.size(), " | captured=", inputAllZero.size());
  }

  static thread_local std::unordered_set<uint32_t> topkOutputGlobalIdsTrace;

  if (op->type_type() == ::tt::target::ttnn::OpType::TopKOp) {
    if (!traceFastSkipHost) {
      logTopKOutputs(op, programContext, pfx);
    }
    dumpTopKTensorsAsNpy(op, programContext, pfx, seq);
    const auto *topKOp = op->type_as_TopKOp();
    if (topKOp && topKOp->outputs()) {
      for (flatbuffers::uoffset_t i = 0; i < topKOp->outputs()->size(); ++i) {
        const auto *outRef = topKOp->outputs()->Get(i);
        if (outRef) {
          topkOutputGlobalIdsTrace.insert(outRef->global_id());
        }
      }
    }
    return;
  }

  if (op->type_type() == ::tt::target::ttnn::OpType::TypecastOp) {
    const auto *typecastOp = op->type_as_TypecastOp();
    if (typecastOp && typecastOp->in() &&
        topkOutputGlobalIdsTrace.count(typecastOp->in()->global_id())) {
      static thread_local size_t typecastSeqInTrace = 0;
      dumpTypecastTensorsAsNpy(op, programContext, pfx, ++typecastSeqInTrace);
    }
  }

  if (traceFastSkipHost) {
    LOG_INFO(kOpTraceLog, pfx,
             "OUT | trace_fast | per_device_host_skipped "
             "(set TT_RUNTIME_OP_TENSOR_TRACE_FAST=0 for full host trace)");
    return;
  }

  std::optional<::tt::runtime::TensorRef> outRef = getOpOutputRef(oc, cb);
  if (!outRef) {
    LOG_INFO(kOpTraceLog, pfx, "OUT | none | ",
             (enumName ? enumName : "?"));
    return;
  }

  std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> outMap =
      getOpOutputTensor(oc, cb);
  if (outMap.empty()) {
    LOG_WARNING(kOpTraceLog, pfx, "WARN | OUT | empty_after_host_read");
    return;
  }

  std::vector<std::uint32_t> zeroDev, nonzeroDev, unknownDev;
  std::string briefZero, briefNonzero, briefUnk;
  for (const auto &kv : outMap) {
    std::optional<bool> oz = hostTensorAllZero(kv.second);
    std::string brief = tensorBrief(kv.second);
    if (!oz.has_value()) {
      unknownDev.push_back(kv.first);
      briefUnk = std::move(brief);
      continue;
    }
    if (*oz) {
      zeroDev.push_back(kv.first);
      briefZero = std::move(brief);
    } else {
      nonzeroDev.push_back(kv.first);
      briefNonzero = std::move(brief);
    }
  }

  const bool everyOutZero =
      !zeroDev.empty() && nonzeroDev.empty() && unknownDev.empty();

  if (!unknownDev.empty()) {
    LOG_INFO(kOpTraceLog, pfx, "OUT | unk | ", unknownDev.size(), "/",
             outMap.size(), " | ", briefUnk, " | dev=", formatDeviceIds(unknownDev));
  }
  if (!nonzeroDev.empty()) {
    LOG_INFO(kOpTraceLog, pfx, "OUT | ok | ", nonzeroDev.size(), "/",
             outMap.size(), " | ", briefNonzero, " | dev=",
             formatDeviceIds(nonzeroDev));
  }
  if (!zeroDev.empty()) {
    LOG_WARNING(kOpTraceLog, pfx, "OUT | ZERO | ", zeroDev.size(), "/",
                outMap.size(), " | ", briefZero, " | dev=",
                formatDeviceIds(zeroDev));
  }

  logGroupedPerDeviceSums(pfx, "OUT", outMap);

  if (op->type_type() == ::tt::target::ttnn::OpType::SoftmaxOp) {
    dumpFullTensorValues(pfx, "OUT softmax", outMap);
  }

  logAllGatherSumInvariant(pfx, programContext, op, oc, cb, outMap);

  if (opTensorTraceMeshDetailEnabled()) {
    if (op->type_type() == ::tt::target::ttnn::OpType::MeshShardOp) {
      logMeshShardFullToShardDetail(op->type_as_MeshShardOp(), programContext,
                                    outMap, pfx);
    } else if (op->type_type() == ::tt::target::ttnn::OpType::MeshPartitionOp) {
      logMeshPartitionDetail(op->type_as_MeshPartitionOp(), programContext,
                             outMap, pfx);
    }
  }

  if (everyOutZero && anyInputNonZeroKnown) {
    LOG_WARNING(kOpTraceLog, pfx,
                "WARN | SUSPICIOUS | nonzero_inputs_zero_outputs");
  }
}

bool opTensorTraceTopKDumpEnabled() {
  static const bool on = !getTopKDumpDir().empty();
  return on;
}

bool opTensorTraceOpDumpEnabled() {
  return opTensorTraceTopKDumpEnabled();
}

void dumpGenericOpTensorsAsNpy(const ::tt::target::ttnn::Operation *op,
                               ProgramContext *programContext,
                               const std::string &pfx, size_t opSeq,
                               const std::vector<uint32_t> &inGlobalIds,
                               uint32_t outGlobalId) {
  std::string dumpDir = getTopKDumpDir();
  if (dumpDir.empty()) {
    return;
  }
  if (!ensureDirExists(dumpDir)) {
    return;
  }

  const char *enumName =
      ::tt::target::ttnn::EnumNameOpType(op->type_type());
  std::string opName = enumName ? std::string(enumName) : "unknown";
  std::string mlirOp = extractMlirTtnnOpName(
      op->debug_info() ? op->debug_info()->c_str() : nullptr);
  std::string loc = extractShortLoc(
      op->loc_info() ? op->loc_info()->c_str() : nullptr);

  std::string safeName = mlirOp;
  for (char &c : safeName) {
    if (c == '.') { c = '_'; }
  }
  if (safeName.empty()) {
    safeName = opName;
  }

  const std::string base =
      dumpDir + "/" + safeName + "_" + std::to_string(opSeq);

  CallbackContext cb = makeCb(programContext);
  OpContext oc = makeOp(op);

  std::vector<::tt::runtime::TensorRef> inRefs = getOpInputRefs(oc, cb);
  for (size_t i = 0; i < inRefs.size(); ++i) {
    const auto &ref = inRefs[i];
    const auto *refPtr =
        &ref.as<::tt::target::ttnn::TensorRef>(::tt::runtime::DeviceRuntime::TTNN);
    if (!refPtr) {
      continue;
    }
    std::vector<::tt::runtime::Tensor> hosts =
        hostShardsFromPoolRef(programContext, refPtr);
    for (size_t d = 0; d < hosts.size(); ++d) {
      std::string p = base + "_in" + std::to_string(i) +
                       "_dev" + std::to_string(d) + ".npy";
      if (writeHostTensorAsNpy(p, hosts[d])) {
        LOG_INFO(kOpTraceLog, pfx, "OP_DUMP | wrote in", i, " dev=", d,
                 " → ", p);
      }
    }
  }

  if (op->type_type() == ::tt::target::ttnn::OpType::TopKOp) {
    const auto *topKOp = op->type_as_TopKOp();
    if (topKOp && topKOp->outputs()) {
      const char *outputNames[] = {"values", "indices"};
      for (size_t outIdx = 0; outIdx < topKOp->outputs()->size(); ++outIdx) {
        const auto *outRef = topKOp->outputs()->Get(
            static_cast<flatbuffers::uoffset_t>(outIdx));
        if (!outRef) {
          continue;
        }
        std::vector<::tt::runtime::Tensor> hosts =
            hostShardsFromPoolRef(programContext, outRef);
        const char *name = outIdx < 2 ? outputNames[outIdx] : "output";
        for (size_t d = 0; d < hosts.size(); ++d) {
          std::string p = base + "_" + name + "_dev" + std::to_string(d) + ".npy";
          writeHostTensorAsNpy(p, hosts[d]);
        }
      }
    }
  } else {
    std::unordered_map<std::uint32_t, ::tt::runtime::Tensor> outMap =
        getOpOutputTensor(oc, cb);
    for (const auto &kv : outMap) {
      std::string p = base + "_out_dev" + std::to_string(kv.first) + ".npy";
      if (writeHostTensorAsNpy(p, kv.second)) {
        LOG_INFO(kOpTraceLog, pfx, "OP_DUMP | wrote out dev=", kv.first,
                 " → ", p);
      }
    }
  }

  std::string metaPath = base + "_meta.json";
  std::ofstream meta(metaPath);
  if (meta) {
    meta << "{\n  \"op_seq\": " << opSeq
         << ",\n  \"op_type\": \"" << escapeJsonString(opName) << "\""
         << ",\n  \"mlir_op\": \"" << escapeJsonString(mlirOp) << "\""
         << ",\n  \"loc\": \"" << escapeJsonString(loc) << "\""
         << ",\n  \"in_global_ids\": [";
    for (size_t i = 0; i < inGlobalIds.size(); ++i) {
      if (i) { meta << ", "; }
      meta << inGlobalIds[i];
    }
    meta << "]"
         << ",\n  \"out_global_id\": " << outGlobalId;

    meta << ",\n  \"params\": {";
    switch (op->type_type()) {
    case ::tt::target::ttnn::OpType::TopKOp: {
      const auto *topKOp = op->type_as_TopKOp();
      if (topKOp) {
        meta << "\"dim\": " << topKOp->dim()
             << ", \"k\": " << topKOp->k()
             << ", \"largest\": " << (topKOp->largest() ? "true" : "false")
             << ", \"sorted\": " << (topKOp->sorted() ? "true" : "false");
        if (topKOp->outputs()) {
          meta << ", \"out_global_ids\": [";
          for (flatbuffers::uoffset_t i = 0; i < topKOp->outputs()->size(); ++i) {
            if (i) { meta << ", "; }
            const auto *r = topKOp->outputs()->Get(i);
            meta << (r ? r->global_id() : 0);
          }
          meta << "]";
        }
      }
      break;
    }
    case ::tt::target::ttnn::OpType::TypecastOp: {
      const auto *tOp = op->type_as_TypecastOp();
      if (tOp) {
        const char *dtName = ::tt::target::EnumNameDataType(tOp->dtype());
        meta << "\"dtype\": \"" << (dtName ? dtName : "?") << "\"";
      }
      break;
    }
    case ::tt::target::ttnn::OpType::ReshapeOp: {
      const auto *rOp = op->type_as_ReshapeOp();
      if (rOp && rOp->shape()) {
        meta << "\"shape\": [";
        for (flatbuffers::uoffset_t i = 0; i < rOp->shape()->size(); ++i) {
          if (i) { meta << ", "; }
          meta << rOp->shape()->Get(i);
        }
        meta << "]";
      }
      break;
    }
    case ::tt::target::ttnn::OpType::ConcatOp: {
      const auto *cOp = op->type_as_ConcatOp();
      if (cOp) {
        meta << "\"dim\": " << cOp->dim();
      }
      break;
    }
    case ::tt::target::ttnn::OpType::SoftmaxOp: {
      const auto *sOp = op->type_as_SoftmaxOp();
      if (sOp) {
        meta << "\"dim\": " << sOp->dimension();
      }
      break;
    }
    case ::tt::target::ttnn::OpType::AllGatherOp: {
      const auto *aOp = op->type_as_AllGatherOp();
      if (aOp) {
        meta << "\"all_gather_dim\": " << aOp->all_gather_dim()
             << ", \"cluster_axis\": " << aOp->cluster_axis();
      }
      break;
    }
    case ::tt::target::ttnn::OpType::SliceOp: {
      const auto *sOp = op->type_as_SliceOp();
      if (sOp) {
        auto writeI64Vec = [&](const char *name,
                               const flatbuffers::Vector<int64_t> *v) {
          meta << "\"" << name << "\": [";
          if (v) {
            for (flatbuffers::uoffset_t i = 0; i < v->size(); ++i) {
              if (i) { meta << ", "; }
              meta << v->Get(i);
            }
          }
          meta << "]";
        };
        if (sOp->params_type() ==
            ::tt::target::ttnn::SliceOpParams::SliceStaticOpParams) {
          const auto *sp = sOp->params_as_SliceStaticOpParams();
          if (sp) {
            writeI64Vec("begins", sp->begins());
            meta << ", ";
            writeI64Vec("ends", sp->ends());
            meta << ", ";
          }
        }
        writeI64Vec("step", sOp->step());
      }
      break;
    }
    case ::tt::target::ttnn::OpType::EltwiseBinaryOp: {
      const auto *eOp = op->type_as_EltwiseBinaryOp();
      if (eOp) {
        const char *btNames[] = {
            "Add", "Multiply", "LogicalRightShift", "Subtract",
            "Equal", "NotEqual", "GreaterEqual", "GreaterThan",
            "LessEqual", "LessThan", "Divide", "LogicalAnd",
            "LogicalOr", "LogicalXor"};
        auto bt = static_cast<unsigned>(eOp->type());
        const char *btName = bt < 14 ? btNames[bt] : "?";
        meta << "\"binary_type\": \"" << btName << "\"";
      }
      break;
    }
    case ::tt::target::ttnn::OpType::ToLayoutOp: {
      const auto *tOp = op->type_as_ToLayoutOp();
      if (tOp) {
        const char *layoutNames[] = {"row_major", "tile", "invalid"};
        auto lt = static_cast<unsigned>(tOp->layout());
        meta << "\"layout\": \"" << (lt < 3 ? layoutNames[lt] : "?") << "\"";
      }
      break;
    }
    case ::tt::target::ttnn::OpType::ScatterOp: {
      const auto *sOp = op->type_as_ScatterOp();
      if (sOp) {
        meta << "\"dim\": " << sOp->dim();
      }
      break;
    }
    default:
      break;
    }
    meta << "}";

    meta << "\n}\n";
  }
}

void opTensorTraceOpDump(const ::tt::target::ttnn::Operation *op,
                         ProgramContext *programContext) {
  if (!op || !programContext) {
    return;
  }
  static thread_local size_t dumpSeq = 0;
  static thread_local std::unordered_set<uint32_t> trackedGlobalIds;

  if (op->type_type() == ::tt::target::ttnn::OpType::DeallocateOp) {
    return;
  }

  CallbackContext cb = makeCb(programContext);
  OpContext oc = makeOp(op);

  bool isTopK = (op->type_type() == ::tt::target::ttnn::OpType::TopKOp);

  std::vector<::tt::runtime::TensorRef> inRefs = getOpInputRefs(oc, cb);
  std::vector<uint32_t> inGlobalIds;
  bool anyInputTracked = false;
  for (const auto &ref : inRefs) {
    const auto *refPtr =
        &ref.as<::tt::target::ttnn::TensorRef>(::tt::runtime::DeviceRuntime::TTNN);
    uint32_t gid = refPtr ? refPtr->global_id() : 0;
    inGlobalIds.push_back(gid);
    if (trackedGlobalIds.count(gid)) {
      anyInputTracked = true;
    }
  }

  if (!isTopK && !anyInputTracked) {
    return;
  }

  uint32_t outGlobalId = 0;
  if (isTopK) {
    const auto *topKOp = op->type_as_TopKOp();
    if (topKOp && topKOp->outputs()) {
      for (flatbuffers::uoffset_t i = 0; i < topKOp->outputs()->size(); ++i) {
        const auto *outRef = topKOp->outputs()->Get(i);
        if (outRef) {
          trackedGlobalIds.insert(outRef->global_id());
          if (i == 0) {
            outGlobalId = outRef->global_id();
          }
        }
      }
    }
  } else {
    std::optional<::tt::runtime::TensorRef> outRef = getOpOutputRef(oc, cb);
    if (outRef) {
      const auto *refPtr =
          &outRef->as<::tt::target::ttnn::TensorRef>(::tt::runtime::DeviceRuntime::TTNN);
      if (refPtr) {
        outGlobalId = refPtr->global_id();
        if (op->type_type() != ::tt::target::ttnn::OpType::ScatterOp) {
          trackedGlobalIds.insert(outGlobalId);
        }
      }
    }
  }

  const size_t seq = ++dumpSeq;
  const std::string pfx =
      std::string("#op_dump_") + std::to_string(seq) + " | ";
  dumpGenericOpTensorsAsNpy(op, programContext, pfx, seq,
                            inGlobalIds, outGlobalId);
}

} // namespace tt::runtime::ttnn
