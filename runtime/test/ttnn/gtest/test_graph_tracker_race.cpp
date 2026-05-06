// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Race reproducer for the tt::tt_metal::GraphTracker singleton.
//
// Two threads share a process and the GraphTracker singleton:
//
//   - Compile thread: randomly picks one of four OpModel constraint
//     queries (two valid, two with shape mismatches). Each query goes
//     through op_model::ttnn::executeConstraintQuery -> ScopedGraphCapture
//     -> push_processor / pop_processor on the singleton.
//
//   - Runtime thread: repeatedly calls tt::runtime::submit on a
//     pre-built .ttnn flatbuffer. Every ttnn op dispatched fires
//     track_function_start, which iterates the singleton's
//     `processors` vector.
//
// Without serialization the runtime thread can iterate `processors`
// while the compile thread is mid-mutation. Build with
// -DCMAKE_BUILD_TYPE=TSan and TSAN should report a data race on
// tt::tt_metal::GraphTracker::processors.
//
// Set TTMLIR_RACE_TEST_FB to the path of a compiled .ttnn flatbuffer
// (see README in this directory for the recipe).

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

#include <tt-metalium/mesh_device.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <random>
#include <sched.h>
#include <thread>
#include <vector>

namespace {

using ::mlir::tt::ttnn::AddOp;
using ::mlir::tt::ttnn::BufferType;
using ::mlir::tt::ttnn::TensorMemoryLayout;
using ::mlir::tt::ttnn::TTNNLayoutAttr;
using ::mlir::tt::ttnn::op_model::OpModel;
using ::mlir::tt::ttnn::op_model::SingletonDeviceContext;

void pinThisThreadToCpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  // Best effort. If pinning fails (e.g. cpuset restrictions), the test still
  // runs; pinning just narrows the interleaving window.
  ::pthread_setaffinity_np(::pthread_self(), sizeof(set), &set);
}

// Inherit from OpModelFixture for the layout helpers (CreateTiledLayout,
// CreateWorkerGrid, the MLIR context/builder members) but skip its SetUp:
// it calls SingletonDeviceContext::openDevice() and we need to share the
// runtime's MeshDevice instead via setExternalDevice.
class GraphTrackerRaceTest : public OpModelFixture {
protected:
  void SetUp() override {
    const char *fbPath = std::getenv("TTMLIR_RACE_TEST_FB");
    ASSERT_NE(fbPath, nullptr)
        << "TTMLIR_RACE_TEST_FB is not set. Compile a flatbuffer and set the "
           "env var to its path. Recipe:\n"
           "  ttrt query --save-artifacts\n"
           "  SD=$(pwd)/ttrt-artifacts/system_desc.ttsys\n"
           "  ttmlir-opt "
           "--ttir-to-ttnn-backend-pipeline=\"system-desc-path=$SD\" \\\n"
           "    -o /tmp/phi.mlir "
           "test/ttmlir/models/single_blocks_and_layers/"
           "phi_1_decode_layer.mlir\n"
           "  ttmlir-translate --ttnn-to-flatbuffer -o /tmp/phi.ttnn "
           "/tmp/phi.mlir\n"
           "  TTMLIR_RACE_TEST_FB=/tmp/phi.ttnn ./test_graph_tracker_race\n";

    // MLIR context + module setup (mirrors OpModelFixture::SetUp's body
    // minus the device open, which we do ourselves below).
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());

    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTNN);

    // Load the flatbuffer. Path comes from env var so the test stays
    // self-contained — no build-time codegen, no checked-in binary.
    binary_ = std::make_unique<tt::runtime::Binary>(
        tt::runtime::Binary::loadFromPath(fbPath));

    // Open the runtime's mesh device first; OpModel's SingletonDeviceContext
    // will piggyback on this same MeshDevice via setExternalDevice. Two
    // independent MeshDevice instances would fight over the underlying
    // hardware.
    tt::runtime::MeshDeviceOptions opts;
    opts.meshShape = std::vector<uint32_t>{1, 1};
    runtimeDevice_ = std::make_unique<tt::runtime::Device>(
        tt::runtime::openMeshDevice(opts));

    auto &meshRef =
        runtimeDevice_->as<::tt::tt_metal::distributed::MeshDevice>(
            tt::runtime::DeviceRuntime::TTNN);
    // Aliasing shared_ptr: shares ownership with the runtime's handle but
    // points at the typed MeshDevice. Avoids a second close.
    std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> meshShared(
        runtimeDevice_->handle, &meshRef);
    SingletonDeviceContext::setExternalDevice(meshShared);

    // Allocate persistent zeroed host buffers + cache layouts for each
    // input. Device tensors are re-created per submit iteration via
    // makeFreshInputs() because TTNN programs consume their inputs.
    const auto descs = binary_->getProgramInputs(0);
    inputDescs_ = descs;
    for (size_t i = 0; i < descs.size(); ++i) {
      const auto &desc = descs[i];
      size_t numElements = 1;
      for (auto d : desc.shape) {
        numElements *= d;
      }
      const size_t bytes = numElements * desc.elementSize();
      auto data = std::make_shared<std::vector<std::byte>>(bytes, std::byte{0});
      ownedData_.push_back(data);
      inputLayouts_.push_back(tt::runtime::getLayout(
          *binary_, /*programIndex=*/0,
          /*inputIndex=*/static_cast<uint32_t>(i)));
    }

    // Pre-build the OpModel-side layouts. Use multi-tile shapes so each
    // query exercises real allocator/grid logic. Tile is 32x32; we use
    // 8x8, 4x4, 8x4 tiles.
    shapeBigA_ = {256, 256};      // 8x8 tiles
    shapeBigSmall_ = {128, 128};  // 4x4 tiles (mismatch with BigA)
    shapeBigCol_ = {256, 128};    // 8x4 tiles (col-mismatch with BigA)

    layoutDramBigA_ = CreateTiledLayout(shapeBigA_, BufferType::DRAM,
                                        TensorMemoryLayout::Interleaved);
    layoutL1BigA_ = CreateTiledLayout(shapeBigA_, BufferType::L1,
                                      TensorMemoryLayout::Interleaved);
    layoutDramBigSmall_ = CreateTiledLayout(
        shapeBigSmall_, BufferType::DRAM, TensorMemoryLayout::Interleaved);
    layoutDramBigCol_ = CreateTiledLayout(shapeBigCol_, BufferType::DRAM,
                                          TensorMemoryLayout::Interleaved);

    workerGrid_ = CreateWorkerGrid();
  }

  std::vector<tt::runtime::Tensor> makeFreshInputs() {
    std::vector<tt::runtime::Tensor> result;
    result.reserve(inputDescs_.size());
    for (size_t i = 0; i < inputDescs_.size(); ++i) {
      const auto &desc = inputDescs_[i];
      tt::runtime::Tensor hostTensor = tt::runtime::createOwnedHostTensor(
          ownedData_[i]->data(), desc.shape, desc.stride,
          static_cast<uint32_t>(desc.elementSize()), desc.dataType);
      result.push_back(
          tt::runtime::toLayout(hostTensor, *runtimeDevice_, inputLayouts_[i]));
    }
    return result;
  }

  void TearDown() override {
    inputLayouts_.clear();
    inputDescs_.clear();
    ownedData_.clear();
    if (SingletonDeviceContext::getInstance().isDeviceInitialized()) {
      SingletonDeviceContext::closeInstance();
    }
    if (runtimeDevice_) {
      tt::runtime::closeMeshDevice(*runtimeDevice_);
      runtimeDevice_.reset();
    }
    binary_.reset();
  }

  // Run one of four constraint queries. Two are valid; two have shape
  // mismatches and are expected to come back as Error from
  // query_op_constraints (the throw is caught inside tt-metal). The
  // push/pop on GraphTracker::processors fires either way.
  void runQuery(int idx) {
    switch (idx) {
    case 0: {
      auto r = OpModel<AddOp>::getOpConstraints(
          workerGrid_, shapeBigA_, layoutDramBigA_, shapeBigA_, layoutDramBigA_,
          layoutDramBigA_);
      if (auto e = r.takeError()) {
        llvm::consumeError(std::move(e));
      }
      break;
    }
    case 1: {
      auto r = OpModel<AddOp>::getOpConstraints(
          workerGrid_, shapeBigA_, layoutL1BigA_, shapeBigA_, layoutL1BigA_,
          layoutL1BigA_);
      if (auto e = r.takeError()) {
        llvm::consumeError(std::move(e));
      }
      break;
    }
    case 2: {
      // Shape mismatch: 256x256 + 128x128 — not broadcastable.
      auto r = OpModel<AddOp>::getOpConstraints(
          workerGrid_, shapeBigA_, layoutDramBigA_, shapeBigSmall_,
          layoutDramBigSmall_, layoutDramBigA_);
      if (auto e = r.takeError()) {
        llvm::consumeError(std::move(e));
      }
      break;
    }
    case 3: {
      // Shape mismatch: 256x256 + 256x128 — last-dim mismatch.
      auto r = OpModel<AddOp>::getOpConstraints(
          workerGrid_, shapeBigA_, layoutDramBigA_, shapeBigCol_,
          layoutDramBigCol_, layoutDramBigA_);
      if (auto e = r.takeError()) {
        llvm::consumeError(std::move(e));
      }
      break;
    }
    default:
      break;
    }
  }

  std::unique_ptr<tt::runtime::Binary> binary_;
  std::unique_ptr<tt::runtime::Device> runtimeDevice_;
  std::vector<tt::runtime::TensorDesc> inputDescs_;
  std::vector<tt::runtime::Layout> inputLayouts_;
  std::vector<std::shared_ptr<std::vector<std::byte>>> ownedData_;

  TTNNLayoutAttr layoutDramBigA_;
  TTNNLayoutAttr layoutL1BigA_;
  TTNNLayoutAttr layoutDramBigSmall_;
  TTNNLayoutAttr layoutDramBigCol_;
  llvm::SmallVector<int64_t> shapeBigA_;
  llvm::SmallVector<int64_t> shapeBigSmall_;
  llvm::SmallVector<int64_t> shapeBigCol_;
  mlir::tt::ttcore::GridAttr workerGrid_;
};

TEST_F(GraphTrackerRaceTest, CompileVsSubmit) {
  // Tunable via env vars for ad-hoc longer/shorter runs.
  const char *durEnv = std::getenv("TTMLIR_RACE_DURATION_SECONDS");
  const int durationSeconds = durEnv != nullptr ? std::atoi(durEnv) : 30;
  ASSERT_GT(durationSeconds, 0);

  std::atomic<bool> stop{false};
  std::atomic<size_t> compileCount{0};
  std::atomic<size_t> submitCount{0};

  const bool disableCompile =
      std::getenv("TTMLIR_RACE_DISABLE_COMPILE") != nullptr;
  // Pool size controls whether we include the throwing queries.
  // 4 = mix of valid + invalid (exercises throw/leak path).
  // 2 = valid only (tests pure race-vs-lock hypothesis).
  const char *poolEnv = std::getenv("TTMLIR_RACE_POOL_SIZE");
  const unsigned poolSize = poolEnv != nullptr
                                ? static_cast<unsigned>(std::atoi(poolEnv))
                                : 4u;

  std::thread compileThread;
  if (!disableCompile) {
    compileThread = std::thread([&] {
      pinThisThreadToCpu(0);
      std::mt19937 rng(0xC0FFEEu);
      while (!stop.load(std::memory_order_relaxed)) {
        runQuery(static_cast<int>(rng() % poolSize));
        compileCount.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Optional: disable the runtime thread entirely to test whether the
  // compile thread crashes in isolation.
  const bool disableRuntime =
      std::getenv("TTMLIR_RACE_DISABLE_RUNTIME") != nullptr;
  std::thread runtimeThread;
  if (!disableRuntime) {
    runtimeThread = std::thread([&] {
      pinThisThreadToCpu(0);
      while (!stop.load(std::memory_order_relaxed)) {
        std::vector<tt::runtime::Tensor> freshInputs = makeFreshInputs();
        auto outputs =
            tt::runtime::submit(*runtimeDevice_, *binary_, /*programIndex=*/0,
                                freshInputs);
        for (auto &t : outputs) {
          if (tt::runtime::isTensorAllocated(t)) {
            (void)tt::runtime::toHost(t, /*untilize=*/false, /*blocking=*/true);
          }
        }
        submitCount.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  std::this_thread::sleep_for(std::chrono::seconds(durationSeconds));
  stop.store(true, std::memory_order_relaxed);
  if (compileThread.joinable()) {
    compileThread.join();
  }
  if (runtimeThread.joinable()) {
    runtimeThread.join();
  }

  std::cerr << "[graph-tracker-race] compile iterations: "
            << compileCount.load() << ", submit iterations: "
            << submitCount.load() << "\n";

  // Pass unconditionally — the assertion of value here is the TSAN report
  // (or absence thereof). gtest pass/fail is just lifecycle.
  SUCCEED();
}

} // namespace
