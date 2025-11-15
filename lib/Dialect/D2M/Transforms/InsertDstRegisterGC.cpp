// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERGC
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class InterferenceGraph {
  llvm::DenseMap<Value, llvm::SmallVector<Value, 4>> graph;

public:
  void addNode(Value v) {
    if (graph.find(v) == graph.end()) {
      graph[v] = {};
    }
  }

  void addEdge(Value u, Value v) {
    addNode(u);
    addNode(v);
    if (std::find(graph[u].begin(), graph[u].end(), v) == graph[u].end()) {
      graph[u].push_back(v);
    }
    if (std::find(graph[v].begin(), graph[v].end(), u) == graph[v].end()) {
      graph[v].push_back(u);
    }
  }

  bool hasInterference() { return !graph.empty(); }

  void print(raw_ostream &os) {
    for (const auto &[v, neighbors] : graph) {
      os << "Value: ";
      v.print(os);
      os << "\n  Interferes with:\n";
      for (auto n : neighbors) {
        os << "    ";
        n.print(os);
        os << "\n";
      }
    }
  }
};

struct D2MInsertDstRegisterGCPass
    : public impl::D2MInsertDstRegisterGCBase<D2MInsertDstRegisterGCPass> {

  D2MInsertDstRegisterGCPass() = default;
  D2MInsertDstRegisterGCPass(const D2MInsertDstRegisterGCPass &pass) = default;
  D2MInsertDstRegisterGCPass(const D2MInsertDstRegisterGCOptions &options)
      : D2MInsertDstRegisterGCBase(options) {}

  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal()) {
      return;
    }

    auto &liveness = getAnalysis<Liveness>();

    InterferenceGraph ig;
    for (auto &block : func.getBody()) {
      if (block.empty()) {
        continue;
      }
      auto liveIn = liveness.getLiveIn(&block);
      for (auto it1 = liveIn.begin(); it1 != liveIn.end(); ++it1) {
        for (auto it2 = std::next(it1); it2 != liveIn.end(); ++it2) {
          ig.addEdge(*it1, *it2);
        }
      }
    }

    if (ig.hasInterference()) {
      ig.print(llvm::outs());
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
