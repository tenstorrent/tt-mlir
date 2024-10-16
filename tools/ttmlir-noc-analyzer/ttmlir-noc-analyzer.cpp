#include "mlir/Parser/Parser.h"
#include "noc-analytical-model.hpp"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>
#include <jsoncpp/json/json.h>
#include <fstream>

namespace analyzer {
mlir::LogicalResult load(const std::string &filename,
                         std::shared_ptr<mlir::ModuleOp> &module,
                         mlir::MLIRContext &context) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return mlir::failure();
  }

  // Parse the input mlir.

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> owningModule =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!owningModule) {
    llvm::errs() << "Error can't load file " << filename << "\n";
    return mlir::failure();
  }

  module = std::make_shared<mlir::ModuleOp>(owningModule.release());
  return mlir::success();
}

struct NocTxStep {
  int start_cycle;
  int end_cycle;
  int duration;
};

struct NocTx {
  NocTxStep total_issue_latency;
  NocTxStep round_trip_latency;
  NocTxStep wait;
  NocTxStep flit_latency;
  int64_t buffer_size;

  friend std::ostream &operator<<(std::ostream &os, NocTx const &tx) {
    os << "Issue Start: " << tx.total_issue_latency.start_cycle << std::endl;
    os << "Issue End: " << tx.total_issue_latency.end_cycle << std::endl;
    os << "Issue Duration: " << tx.total_issue_latency.duration << std::endl;

    os << "Round Trip Start: " << tx.round_trip_latency.start_cycle
        << std::endl;
    os << "Round Trip End: " << tx.round_trip_latency.end_cycle << std::endl;
    os << "Round Trip Duration: " << tx.round_trip_latency.duration
        << std::endl;

    os << "Wait Start: " << tx.wait.start_cycle << std::endl;
    os << "Wait End: " << tx.wait.end_cycle << std::endl;
    os << "Wait Duration: " << tx.wait.duration << std::endl;

    os << "Flit Start: " << tx.flit_latency.start_cycle << std::endl;
    os << "Flit End: " << tx.flit_latency.end_cycle << std::endl;
    os << "Flit Duration: " << tx.flit_latency.duration << std::endl;

    os << "Buffer Size: " << tx.buffer_size << std::endl;

    return os;
  }
};

const Json::Value tx_step_to_json(NocTxStep const &tx_step) {
  Json::Value tx_step_json;
  tx_step_json["start_cycle"] = tx_step.start_cycle;
  tx_step_json["end_cycle"] = tx_step.end_cycle;
  tx_step_json["duration"] = tx_step.duration;
  return tx_step_json;
}

const Json::Value tx_to_json(NocTx &tx) {
  Json::Value tx_json;
  tx_json["total_issue_latency"] = tx_step_to_json(tx.total_issue_latency);
  tx_json["round_trip_latency"] = tx_step_to_json(tx.round_trip_latency);
  tx_json["wait"] = tx_step_to_json(tx.wait);
  tx_json["flit_latency"] = tx_step_to_json(tx.flit_latency);
  tx_json["buffer_size"] = tx.buffer_size;
  return tx_json;
}

std::vector<std::pair<mlir::tt::ttmetal::CoreRangeAttr, std::vector<NocTx>>> read_txs;
std::vector<std::pair<mlir::tt::ttmetal::CoreRangeAttr, std::vector<NocTx>>> write_txs;

void analyze_operation(mlir::Operation *op) {
  if (llvm::isa<mlir::tt::ttmetal::DispatchOp>(op)) {
    auto core_ranges = op->getAttrOfType<mlir::ArrayAttr>("core_ranges");
    int region_num = 0;
    for (mlir::Region &region : op->getRegions()) {
      for (mlir::Block &block : region.getBlocks()) {
        std::vector<NocTx> core_read_txs;
        std::vector<NocTx> core_write_txs; 
        for (mlir::Operation &inner_op : block.getOperations()) {
          if (llvm::isa<mlir::tt::ttkernel::NocAsyncReadOp>(inner_op) ||
              llvm::isa<mlir::tt::ttkernel::NocAsyncWriteOp>(inner_op)) {
            mlir::Value operand = inner_op.getOperand(2);
            auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(
                operand.getDefiningOp());
            auto constAttr = constOp.getValue();
            auto buffer_size = mlir::dyn_cast<mlir::IntegerAttr>(constAttr).getInt();
            auto &queue_ref =
                llvm::isa<mlir::tt::ttkernel::NocAsyncReadOp>(inner_op) ? core_read_txs : core_write_txs;
            NocTx tx = {};
            tx.buffer_size = buffer_size;
            queue_ref.push_back(tx);
          }
        }
        auto core_read =
            std::pair<mlir::tt::ttmetal::CoreRangeAttr, std::vector<NocTx>>(
                mlir::dyn_cast<mlir::tt::ttmetal::CoreRangeAttr>(
                    core_ranges[region_num]),
                core_read_txs);
        read_txs.push_back(core_read);

        auto core_write =
            std::pair<mlir::tt::ttmetal::CoreRangeAttr, std::vector<NocTx>>(
                mlir::dyn_cast<mlir::tt::ttmetal::CoreRangeAttr>(
                    core_ranges[region_num]),
                core_write_txs);
        write_txs.push_back(core_write);
      }
      region_num++;
    }
  }
}

void assign_step_durations(std::vector<NocTx> &txs, int read) {
  for (NocTx &tx : txs) {
    auto analysis =
        read
            ? analyzer::model::get_read_latency(analyzer::model::noc_params, tx.buffer_size)
            : analyzer::model::get_write_latency(analyzer::model::noc_params,tx.buffer_size);

    tx.total_issue_latency.duration = std::get<0>(analysis);
    tx.round_trip_latency.duration = std::get<1>(analysis);
    tx.flit_latency.duration = std::get<2>(analysis);    
  }
}

void timeline_read_txs(std::vector<NocTx> &txs) {
  for (size_t i = 0; i < txs.size(); i++) {
    NocTx &tx = txs.at(i);
    if (i == 0) {
      tx.total_issue_latency.start_cycle = 0;
      tx.total_issue_latency.end_cycle = tx.total_issue_latency.duration;

      tx.round_trip_latency.start_cycle = tx.total_issue_latency.end_cycle;
      tx.round_trip_latency.end_cycle = tx.round_trip_latency.start_cycle + tx.round_trip_latency.duration;

      tx.wait.start_cycle = tx.round_trip_latency.end_cycle;
      tx.wait.end_cycle = tx.wait.start_cycle;
      tx.wait.duration = 0;
      
      tx.flit_latency.start_cycle = tx.round_trip_latency.end_cycle;
      tx.flit_latency.end_cycle = tx.flit_latency.start_cycle + tx.flit_latency.duration;
    } else {
      NocTx &prev_tx = txs.at(i - 1);
      tx.total_issue_latency.start_cycle = prev_tx.total_issue_latency.end_cycle;
      tx.total_issue_latency.end_cycle = tx.total_issue_latency.start_cycle + tx.total_issue_latency.duration;

      tx.round_trip_latency.start_cycle = tx.total_issue_latency.end_cycle;
      tx.round_trip_latency.end_cycle = tx.round_trip_latency.start_cycle + tx.round_trip_latency.duration;

      if (tx.round_trip_latency.end_cycle < prev_tx.flit_latency.end_cycle) {
        tx.wait.duration = prev_tx.flit_latency.end_cycle - tx.round_trip_latency.end_cycle;
        tx.wait.start_cycle = tx.round_trip_latency.end_cycle;
        tx.wait.end_cycle = tx.wait.start_cycle + tx.wait.duration;
      } else {
        tx.wait.start_cycle = tx.round_trip_latency.end_cycle;
        tx.wait.end_cycle = tx.wait.start_cycle;
        tx.wait.duration = 0;
      }

      tx.flit_latency.start_cycle = tx.wait.end_cycle;
      tx.flit_latency.end_cycle = tx.flit_latency.start_cycle + tx.flit_latency.duration;
    }
  }
}

void timeline_write_txs(std::vector<NocTx> &txs) {
  for (size_t i = 0; i < txs.size(); i++) {
    NocTx &tx = txs.at(i);
    if (i == 0) {
      tx.total_issue_latency.start_cycle = 0;
      tx.total_issue_latency.end_cycle = tx.total_issue_latency.duration;

      tx.wait.start_cycle = tx.total_issue_latency.end_cycle;
      tx.wait.end_cycle = tx.wait.start_cycle;
      tx.wait.duration = 0;

      tx.flit_latency.start_cycle = tx.total_issue_latency.end_cycle;
      tx.flit_latency.end_cycle = tx.flit_latency.start_cycle + tx.flit_latency.duration;

      tx.round_trip_latency.start_cycle = tx.flit_latency.end_cycle;
      tx.round_trip_latency.end_cycle = tx.round_trip_latency.start_cycle + tx.round_trip_latency.duration;
    } else {
      NocTx &prev_tx = txs.at(i - 1);
      tx.total_issue_latency.start_cycle = prev_tx.total_issue_latency.end_cycle;
      tx.total_issue_latency.end_cycle = tx.total_issue_latency.start_cycle + tx.total_issue_latency.duration;

      if (tx.total_issue_latency.end_cycle < prev_tx.flit_latency.end_cycle) {
        tx.wait.duration = prev_tx.flit_latency.end_cycle - tx.total_issue_latency.end_cycle;
        tx.wait.start_cycle = tx.total_issue_latency.end_cycle;
        tx.wait.end_cycle = tx.wait.start_cycle + tx.wait.duration;
      } else {
        tx.wait.start_cycle = prev_tx.total_issue_latency.end_cycle;
        tx.wait.end_cycle = tx.wait.start_cycle;
        tx.wait.duration = 0;
      }

      tx.flit_latency.start_cycle = tx.wait.end_cycle;
      tx.flit_latency.end_cycle = tx.flit_latency.start_cycle + tx.flit_latency.duration;

      tx.round_trip_latency.start_cycle = tx.flit_latency.end_cycle;
      tx.round_trip_latency.end_cycle = tx.round_trip_latency.start_cycle + tx.round_trip_latency.duration;
    }
  }
}

void print_txs(std::vector<NocTx> &txs) {
  for (size_t i = 0; i < txs.size(); i++) {
    std::cout << "TX #: " << i << std::endl;
    std::cout << txs.at(i) << std::endl << std::endl;
  }
}

void export_json(
    std::vector<std::pair<mlir::tt::ttmetal::CoreRangeAttr, std::vector<NocTx>>>
        &read_txs,
    std::vector<std::pair<mlir::tt::ttmetal::CoreRangeAttr, std::vector<NocTx>>> &write_txs) {
  Json::Value root;
  // loop cores
  for (auto &core : read_txs) {
    Json::Value core_json;
    // loop core txs
    for (size_t i = 0; i < core.second.size(); i++) {
      Json::Value tx_json = tx_to_json(core.second.at(i));
      core_json[std::to_string(i)] = tx_json;
    }
    std::string output_core_range_string;
    llvm::raw_string_ostream outputStream(output_core_range_string);
    core.first.printStripped(outputStream);
    root["read"][output_core_range_string] = core_json;
  }

  for (auto &core : write_txs) {
    Json::Value core_json;
    // loop core txs
    for (size_t i = 0; i < core.second.size(); i++) {
      Json::Value tx_json = tx_to_json(core.second.at(i));
      core_json[std::to_string(i)] = tx_json;
    }
    std::string output_core_range_string;
    llvm::raw_string_ostream outputStream(output_core_range_string);
    core.first.printStripped(outputStream);
    root["write"][output_core_range_string] = core_json;
  }

  std::ostream *file = new std::ofstream("analyzer_out.json");
  *file << root << std::endl;
  delete file;
}

} // namespace analyzer


int main() {
  mlir::MLIRContext context = mlir::MLIRContext();
  context.loadDialect<mlir::tt::ttmetal::TTMetalDialect>();
  context.loadDialect<mlir::tt::ttkernel::TTKernelDialect>();

  auto root = std::make_shared<mlir::ModuleOp>();
  mlir::LogicalResult status = analyzer::load(
      "build/test/ttmlir/Silicon/TTMetal/Output/to_layout.mlir.tmp.mlir", root, context);

  if (status.failed()) {
    exit(1);
  }
  std::cout << "STARTING WALK" << std::endl;
  root->walk<mlir::WalkOrder::PreOrder>(analyzer::analyze_operation);
  for (auto &core : analyzer::read_txs) {
    analyzer::assign_step_durations(core.second, 1);
    analyzer::timeline_read_txs(core.second);
    llvm::outs() << "READ on CORE: ";
    core.first.dump();
    analyzer::print_txs(core.second);
  }
  for (auto &core : analyzer::write_txs) {
    analyzer::assign_step_durations(core.second, 0);
    analyzer::timeline_write_txs(core.second);
    llvm::outs() << "WRITE on CORE: ";
    core.first.dump();
    analyzer::print_txs(core.second);
  }

  analyzer::export_json(analyzer::read_txs, analyzer::write_txs);

  std::cout << "FINISHED WALK" << std::endl;
}