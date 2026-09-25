// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutPropagation.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

namespace {
enum class Policy { Disabled, All, PerOperand };
Policy policy = Policy::PerOperand;
size_t validations = 0;
} // namespace

// Link wrapping isolates benchmark policies from production code.
extern "C" bool realPolicy(mlir::Operation *, unsigned) asm(
    "__real__ZN4mlir2tt4ttnn21shouldExploreReshardsEPNS_9OperationEj");
extern "C" bool wrappedPolicy(mlir::Operation *, unsigned) asm(
    "__wrap__ZN4mlir2tt4ttnn21shouldExploreReshardsEPNS_9OperationEj");
extern "C" bool wrappedPolicy(mlir::Operation *op, unsigned operandIdx) {
  if (policy == Policy::Disabled) {
    return false;
  }
  return policy == Policy::All || realPolicy(op, operandIdx);
}

namespace mlir::tt::ttnn {
// Permissive mock for measuring CPU search overhead only.
op_constraint_validation::ValidationResult
mockValidation(Operation *, llvm::ArrayRef<TTNNLayoutAttr>, const OpConfig &,
               uint64_t) asm("__wrap__ZN4mlir2tt4ttnn24op_constraint_"
                             "validation17validateOperationEPNS_"
                             "9OperationEN4llvm8ArrayRefINS1_"
                             "14TTNNLayoutAttrEEERKNS1_8OpConfigEm");
op_constraint_validation::ValidationResult
mockValidation(Operation *, llvm::ArrayRef<TTNNLayoutAttr> inputs,
               const OpConfig &, uint64_t) {
  ++validations;
  return op_constraint_validation::ValidationResult::success(0, inputs.front());
}

namespace {
struct Counts : LayoutPropagationObserver {
  std::vector<size_t> operands;
  size_t crossProduct = 0;
  size_t evaluations = 0;
  void onOpSetup(Operation *,
                 const std::vector<std::vector<InputCandidate>> &inputs,
                 const OutputHints &, size_t product) override {
    for (const auto &input : inputs) {
      operands.push_back(input.size());
    }
    crossProduct = product;
  }
  void onEvaluation(Operation *, const OpConfig &, size_t,
                    llvm::ArrayRef<TTNNLayoutAttr>, bool, const BeamCandidate *,
                    llvm::StringRef) override {
    ++evaluations;
  }
};

void require(bool condition, const char *message) {
  if (!condition) {
    llvm::errs() << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

struct Result {
  double microseconds;
  std::vector<size_t> operands;
  size_t evaluations;
};

Result run(MLIRContext &context, size_t cap) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());
  ttcore::registerDevice(*module);
  module->getOperation()->setAttr(utils::g_TensorL1UsageCapAttrName,
                                  builder.getF32FloatAttr(1.0f));
  auto device = ttcore::lookupDevice(*module);
  llvm::SmallVector<int64_t> shape{1, 1, 2048, 64};
  auto tile = ttcore::TileType::get(builder.getBF16Type());
  auto makeLayout = [&](BufferType buffer, TensorMemoryLayout memory,
                        llvm::ArrayRef<int64_t> grid) {
    return TTNNLayoutAttr::Builder(&context, shape, tile)
        .setBufferType(buffer)
        .setMemoryLayout(memory)
        .setGridShape(grid)
        .buildWithCanonicalCorePlacement(device);
  };
  auto dram =
      makeLayout(BufferType::DRAM, TensorMemoryLayout::Interleaved, {1, 1});
  auto type = RankedTensorType::get(shape, builder.getBF16Type(), dram);
  auto func = builder.create<func::FuncOp>(
      loc, "rotary", builder.getFunctionType({type, type, type}, {type}));
  auto *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  OperationState state(loc, RotaryEmbeddingOp::getOperationName());
  state.addOperands(block->getArguments());
  state.addTypes(type);
  auto *rotary = builder.create(state);
  builder.create<func::ReturnOp>(loc, rotary->getResults());
  require(succeeded(verify(*module)), "benchmark input must be valid MLIR");

  TensorTypeLayoutsMap layouts;
  auto bare = RankedTensorType::get(shape, builder.getBF16Type());
  auto &sharded =
      layouts[bare][builder.getBF16Type()]
             [static_cast<size_t>(TensorPageLayout::Tiled)]
             [static_cast<size_t>(TensorMemoryLayoutIndex::Sharded)];
  for (auto grid : std::vector<std::vector<int64_t>>{
           {1, 1}, {2, 1}, {4, 1}, {8, 1}, {16, 1}, {32, 1}, {64, 1}}) {
    sharded.push_back(
        makeLayout(BufferType::L1, TensorMemoryLayout::HeightSharded, grid));
  }
  llvm::DenseMap<Operation *, std::vector<OpConfig>> configs;
  configs[rotary] = {OpConfig(dram)};
  auto observer = std::make_unique<Counts>();
  auto *counts = observer.get();
  MemoryLayoutPropagation search(func, configs, &layouts, 8, 64, cap,
                                 std::move(observer));
  validations = 0;
  const auto start = std::chrono::steady_clock::now();
  search.run();
  const auto end = std::chrono::steady_clock::now();

  const size_t n = policy == Policy::Disabled ? 1 : cap + 1;
  const size_t cacheN = policy == Policy::All ? n : 1;
  if (counts->operands != std::vector<size_t>{n, cacheN, cacheN}) {
    llvm::errs() << "policy=" << static_cast<int>(policy) << " cap=" << cap
                 << " counts:";
    for (auto count : counts->operands) {
      llvm::errs() << ' ' << count;
    }
    llvm::errs() << '\n';
  }
  require(counts->operands == std::vector<size_t>{n, cacheN, cacheN},
          "unexpected input/cache candidate counts");
  require(counts->crossProduct == n * cacheN * cacheN,
          "unexpected cross product");
  require(counts->evaluations == counts->crossProduct &&
              validations == counts->evaluations,
          "every candidate must reach backend validation");
  require(!search.getBeamState().lookup(rotary).empty(),
          "search must produce a candidate");
  if (policy == Policy::PerOperand && cap > 0) {
    require(rotary->getOperand(1) == block->getArgument(1) &&
                rotary->getOperand(2) == block->getArgument(2),
            "cache operands must remain unchanged");
    require(rotary->getOperand(0) != block->getArgument(0),
            "mock scoring should select an input reshard");
  }
  require(succeeded(verify(*module)), "optimizer must produce valid MLIR");
  return {std::chrono::duration<double, std::micro>(end - start).count(),
          counts->operands, counts->evaluations};
}
} // namespace

int benchmark() {
  MLIRContext context;
  context.loadDialect<ttcore::TTCoreDialect, TTNNDialect, func::FuncDialect>();
  llvm::outs() << "cap,policy,input_candidates,cos_candidates,sin_candidates,"
                  "evaluations,median_us,p10_us,p90_us\n";
  for (size_t cap : {0, 1, 4, 7}) {
    std::vector<double> samples[3];
    Result last[3];
    // Interleave policies and discard warmups to reduce timing bias.
    for (size_t repeat = 0; repeat < 111; ++repeat) {
      for (size_t offset = 0; offset < 3; ++offset) {
        size_t index = (repeat + offset) % 3;
        policy = static_cast<Policy>(index);
        last[index] = run(context, cap);
        if (repeat >= 10) {
          samples[index].push_back(last[index].microseconds);
        }
      }
    }
    const char *names[] = {"disabled", "all", "per_operand"};
    for (size_t index = 0; index < 3; ++index) {
      auto &times = samples[index];
      std::sort(times.begin(), times.end());
      llvm::outs() << cap << ',' << names[index];
      for (auto count : last[index].operands) {
        llvm::outs() << ',' << count;
      }
      llvm::outs() << ',' << last[index].evaluations << ',' << times[50] << ','
                   << times[10] << ',' << times[90] << '\n';
    }
  }
  return 0;
}
} // namespace mlir::tt::ttnn

int main() { return mlir::tt::ttnn::benchmark(); }
