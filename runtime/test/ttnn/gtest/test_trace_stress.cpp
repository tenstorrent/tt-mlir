// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/test/ttnn/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/debug.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

// --- bf16 utilities ---

uint16_t floatToBf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float bf16ToFloat(uint16_t bf) {
  uint32_t bits = static_cast<uint32_t>(bf) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Pearson correlation coefficient between two float vectors
float computePCC(const std::vector<float> &a, const std::vector<float> &b) {
  EXPECT_EQ(a.size(), b.size());
  size_t n = a.size();
  if (n == 0) {
    return 1.0f;
  }

  double sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
  for (size_t i = 0; i < n; i++) {
    sumA += a[i];
    sumB += b[i];
    sumAB += static_cast<double>(a[i]) * b[i];
    sumA2 += static_cast<double>(a[i]) * a[i];
    sumB2 += static_cast<double>(b[i]) * b[i];
  }

  double num = n * sumAB - sumA * sumB;
  double den =
      std::sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
  if (den < 1e-12) {
    return 1.0f;
  }
  return static_cast<float>(num / den);
}

// Convert raw bf16 bytes to float vector
std::vector<float> bf16BytesToFloat(const std::vector<std::byte> &data) {
  std::vector<float> result;
  result.reserve(data.size() / 2);
  for (size_t i = 0; i + 1 < data.size(); i += 2) {
    uint16_t bf;
    std::memcpy(&bf, &data[i], sizeof(bf));
    result.push_back(bf16ToFloat(bf));
  }
  return result;
}

// --- Operation types ---

enum class Op {
  NEW_TRACED,
  RUN_TRACED,
  NEW_REGULAR,
  RUN_REGULAR,
  ALLOC_TENSOR,
  DROP_GRAPH
};

// --- Test state ---

struct GraphState {
  tt::runtime::Binary binary;
  std::vector<tt::runtime::Tensor> inputs;
  std::vector<float> goldenOutput; // expected output as floats
};

struct AllocatedTensor {
  tt::runtime::Tensor deviceTensor;
  std::vector<std::byte> expectedData;
};

class StressState {
public:
  StressState(tt::runtime::Device device, const std::string &tracedPath,
              const std::string &regularPath, uint32_t seed)
      : device_(device), tracedPath_(tracedPath), regularPath_(regularPath),
        rng_(seed) {}

  Op pickOp(const std::set<Op> &allowed) {
    std::vector<Op> candidates(allowed.begin(), allowed.end());

    // Filter out invalid ops based on current state
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [this](Op op) {
                         if (op == Op::RUN_TRACED && tracedGraphs_.empty()) {
                           return true;
                         }
                         if (op == Op::RUN_REGULAR && regularGraphs_.empty()) {
                           return true;
                         }
                         if (op == Op::DROP_GRAPH && tracedGraphs_.empty() &&
                             regularGraphs_.empty()) {
                           return true;
                         }
                         if (op == Op::ALLOC_TENSOR && tracedGraphs_.empty()) {
                           return true;
                         }
                         return false;
                       }),
        candidates.end());

    if (candidates.empty()) {
      // Fallback
      if (allowed.count(Op::NEW_TRACED)) {
        return Op::NEW_TRACED;
      }
      return Op::NEW_REGULAR;
    }

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    return candidates[dist(rng_)];
  }

  std::string execute(Op op) {
    std::string name =
        opName(op) + "_" + std::to_string(counter_);
    counter_++;

    switch (op) {
    case Op::NEW_TRACED:
      executeNewGraph(name, tracedPath_, tracedGraphs_, /*isMatmul=*/true);
      break;
    case Op::RUN_TRACED:
      name = executeRunGraph(tracedGraphs_, /*isMatmul=*/true);
      break;
    case Op::NEW_REGULAR:
      executeNewGraph(name, regularPath_, regularGraphs_, /*isMatmul=*/false);
      break;
    case Op::RUN_REGULAR:
      name = executeRunGraph(regularGraphs_, /*isMatmul=*/false);
      break;
    case Op::ALLOC_TENSOR:
      executeAllocTensor(name);
      break;
    case Op::DROP_GRAPH:
      name = executeDropGraph();
      break;
    }

    return "[" + std::to_string(counter_) + "] " + opName(op) + "(" + name +
           ")";
  }

  void verifyLiveTensors() {
    for (auto &[name, alloc] : liveTensors_) {
      auto hostTensors =
          tt::runtime::toHost(alloc.deviceTensor, /*untilize=*/true,
                              /*blocking=*/true);
      ASSERT_FALSE(hostTensors.empty())
          << "Failed to read back tensor " << name;
      auto readBack = tt::runtime::getTensorDataBuffer(hostTensors[0]);
      ASSERT_EQ(readBack.size(), alloc.expectedData.size())
          << "Size mismatch for tensor " << name;
      ASSERT_EQ(
          std::memcmp(readBack.data(), alloc.expectedData.data(),
                      readBack.size()),
          0)
          << "Tensor " << name << " corrupted at step " << counter_;
    }
  }


private:
  static constexpr uint32_t kDim = 256;
  static constexpr uint32_t kNumElements = kDim * kDim;

  tt::runtime::Device device_;
  std::string tracedPath_;
  std::string regularPath_;
  std::mt19937 rng_;
  std::map<std::string, GraphState> tracedGraphs_;
  std::map<std::string, GraphState> regularGraphs_;
  std::map<std::string, AllocatedTensor> liveTensors_;
  uint64_t counter_ = 0;

  static std::string opName(Op op) {
    switch (op) {
    case Op::NEW_TRACED:
      return "new_traced";
    case Op::RUN_TRACED:
      return "run_traced";
    case Op::NEW_REGULAR:
      return "new_regular";
    case Op::RUN_REGULAR:
      return "run_regular";
    case Op::ALLOC_TENSOR:
      return "alloc_tensor";
    case Op::DROP_GRAPH:
      return "drop_graph";
    }
    return "unknown";
  }

  // Generate random bf16 data
  std::vector<uint16_t> randomBf16Data() {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<uint16_t> data(kNumElements);
    for (auto &val : data) {
      val = floatToBf16(dist(rng_));
    }
    return data;
  }

  // Compute matmul golden (256x256 @ 256x256) in float
  std::vector<float> computeMatmulGolden(const std::vector<uint16_t> &a,
                                          const std::vector<uint16_t> &b) {
    std::vector<float> result(kNumElements, 0.0f);
    for (uint32_t i = 0; i < kDim; i++) {
      for (uint32_t k = 0; k < kDim; k++) {
        float aVal = bf16ToFloat(a[i * kDim + k]);
        for (uint32_t j = 0; j < kDim; j++) {
          result[i * kDim + j] += aVal * bf16ToFloat(b[k * kDim + j]);
        }
      }
    }
    return result;
  }

  // Compute add golden (256x256 + 256x256) in float
  std::vector<float> computeAddGolden(const std::vector<uint16_t> &a,
                                       const std::vector<uint16_t> &b) {
    std::vector<float> result(kNumElements);
    for (size_t i = 0; i < kNumElements; i++) {
      result[i] = bf16ToFloat(a[i]) + bf16ToFloat(b[i]);
    }
    return result;
  }

  tt::runtime::Tensor createInputTensor(const std::vector<uint16_t> &data) {
    std::vector<uint32_t> shape = {kDim, kDim};
    std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
    return tt::runtime::createOwnedHostTensor(
        data.data(), shape, stride, sizeof(uint16_t),
        tt::target::DataType::BFloat16);
  }

  void executeNewGraph(const std::string &name, const std::string &binaryPath,
                       std::map<std::string, GraphState> &graphMap,
                       bool isMatmul) {
    tt::runtime::Binary binary =
        tt::runtime::Binary::loadFromPath(binaryPath.c_str());

    auto inputDataA = randomBf16Data();
    auto inputDataB = randomBf16Data();

    auto hostA = createInputTensor(inputDataA);
    auto hostB = createInputTensor(inputDataB);

    tt::runtime::Layout layoutA =
        tt::runtime::getLayout(binary, 0, 0);
    tt::runtime::Layout layoutB =
        tt::runtime::getLayout(binary, 0, 1);

    auto deviceA = tt::runtime::toLayout(hostA, device_, layoutA, true);
    auto deviceB = tt::runtime::toLayout(hostB, device_, layoutB, true);

    std::vector<tt::runtime::Tensor> inputs = {deviceA, deviceB};
    auto outputs = tt::runtime::submit(device_, binary, 0, inputs);
    ASSERT_FALSE(outputs.empty()) << "submit returned no outputs for " << name;

    // Verify output
    auto golden = isMatmul ? computeMatmulGolden(inputDataA, inputDataB)
                           : computeAddGolden(inputDataA, inputDataB);
    verifyOutput(outputs[0], golden, name);

    graphMap.emplace(name, GraphState{
        .binary = binary,
        .inputs = std::move(inputs),
        .goldenOutput = std::move(golden),
    });
  }

  std::string executeRunGraph(std::map<std::string, GraphState> &graphMap,
                              bool isMatmul) {
    auto it = graphMap.begin();
    std::uniform_int_distribution<size_t> dist(0, graphMap.size() - 1);
    std::advance(it, dist(rng_));
    const std::string &name = it->first;
    auto &state = it->second;

    auto outputs =
        tt::runtime::submit(device_, state.binary, 0, state.inputs);
    EXPECT_FALSE(outputs.empty()) << "submit returned no outputs for " << name;
    if (!outputs.empty()) {
      verifyOutput(outputs[0], state.goldenOutput, name);
    }
    return name;
  }

  void executeAllocTensor(const std::string &name) {
    auto data = randomBf16Data();
    auto hostTensor = createInputTensor(data);

    // Use the first traced graph's layout to place on device
    auto &firstGraph = tracedGraphs_.begin()->second;
    tt::runtime::Layout layout =
        tt::runtime::getLayout(firstGraph.binary, 0, 0);
    auto deviceTensor =
        tt::runtime::toLayout(hostTensor, device_, layout, true);

    // Read back immediately to get the actual on-device representation
    auto hostReadBack =
        tt::runtime::toHost(deviceTensor, /*untilize=*/true, /*blocking=*/true);
    auto expectedData = tt::runtime::getTensorDataBuffer(hostReadBack[0]);

    liveTensors_[name] = AllocatedTensor{
        .deviceTensor = deviceTensor,
        .expectedData = std::move(expectedData),
    };
  }

  std::string executeDropGraph() {
    // Merge both maps to pick from
    std::vector<std::pair<std::string, bool>> candidates;
    for (auto &[name, _] : tracedGraphs_) {
      candidates.push_back({name, true});
    }
    for (auto &[name, _] : regularGraphs_) {
      candidates.push_back({name, false});
    }
    if (candidates.empty()) {
      return "none";
    }

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    auto [name, isTraced] = candidates[dist(rng_)];

    if (isTraced) {
      tracedGraphs_.erase(name);
    } else {
      regularGraphs_.erase(name);
    }
    return name;
  }

  void verifyOutput(const tt::runtime::Tensor &output,
                    const std::vector<float> &golden,
                    const std::string &name) {
    auto hostTensors =
        tt::runtime::toHost(output, /*untilize=*/true, /*blocking=*/true);
    ASSERT_FALSE(hostTensors.empty())
        << "Failed to read back output for " << name;
    auto outputBytes = tt::runtime::getTensorDataBuffer(hostTensors[0]);
    auto outputFloats = bf16BytesToFloat(outputBytes);

    float pcc = computePCC(outputFloats, golden);
    ASSERT_GE(pcc, 0.99f) << "Output PCC check failed for " << name
                           << ": pcc=" << pcc << " at step " << counter_;
  }
};

// --- Test parameters ---

struct StressParams {
  std::string name;
  std::set<Op> allowedOps;
  uint32_t seed;
  uint32_t numOps;
  bool expectAllocCorruption;
};

std::ostream &operator<<(std::ostream &os, const StressParams &p) {
  return os << p.name;
}

// --- Test fixture ---

class TraceStressTest : public ::testing::TestWithParam<StressParams> {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTNN);
    tt::runtime::MeshDeviceOptions opts;
    opts.meshShape = {1, 1};
    opts.enableProgramCache = true;
    device_.emplace(tt::runtime::openMeshDevice(opts));
    tt::runtime::clearProgramCache(*device_);
  }

  void TearDown() override {
    if (device_) {
      tt::runtime::closeMeshDevice(*device_);
    }
    tt::runtime::debug::Stats::get().clear();
  }

  std::optional<tt::runtime::Device> device_;
};

// --- Test body ---

TEST_P(TraceStressTest, RandomOps) {
  const auto &params = GetParam();

  if (params.expectAllocCorruption) {
    GTEST_SKIP() << "toLayout staleness detection not yet implemented";
  }

  const char *home = std::getenv("TT_MLIR_HOME");
  ASSERT_NE(home, nullptr)
      << "TT_MLIR_HOME environment variable must be set";
  std::string base = std::string(home) +
                     "/build/test/ttmlir/Runtime/TTNN/n150/trace/Output/";
  std::string tracedPath = base + "single_matmul.mlir.tmp.ttnn";
  std::string regularPath = base + "single_add_no_trace.mlir.tmp.ttnn";

  StressState state(*device_, tracedPath, regularPath, params.seed);

  for (uint32_t i = 0; i < params.numOps; i++) {
    Op op = state.pickOp(params.allowedOps);
    state.execute(op);
    state.verifyLiveTensors();
  }
}

INSTANTIATE_TEST_SUITE_P(
    TraceStress, TraceStressTest,
    ::testing::Values(
        StressParams{"traced_only",
                     {Op::NEW_TRACED, Op::RUN_TRACED},
                     42, 200, false},
        StressParams{"traced_with_drops",
                     {Op::NEW_TRACED, Op::RUN_TRACED, Op::DROP_GRAPH},
                     42, 200, false},
        StressParams{"traced_and_regular",
                     {Op::NEW_TRACED, Op::RUN_TRACED, Op::NEW_REGULAR,
                      Op::RUN_REGULAR},
                     42, 200, false},
        StressParams{"traced_regular_drops",
                     {Op::NEW_TRACED, Op::RUN_TRACED, Op::NEW_REGULAR,
                      Op::RUN_REGULAR, Op::DROP_GRAPH},
                     42, 200, false},
        StressParams{"traced_and_alloc",
                     {Op::NEW_TRACED, Op::RUN_TRACED, Op::ALLOC_TENSOR},
                     42, 200, true},
        StressParams{"kitchen_sink",
                     {Op::NEW_TRACED, Op::RUN_TRACED, Op::NEW_REGULAR,
                      Op::RUN_REGULAR, Op::ALLOC_TENSOR, Op::DROP_GRAPH},
                     42, 200, true}),
    [](const ::testing::TestParamInfo<StressParams> &info) {
      return info.param.name;
    });

} // namespace
