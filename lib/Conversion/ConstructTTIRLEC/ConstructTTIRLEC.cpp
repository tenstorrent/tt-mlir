// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/ConstructTTIRLEC/ConstructTTIRLEC.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

#define GEN_PASS_DEF_CONSTRUCTTTIRLEC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt

namespace {

/// Adapt a value of width `srcWidth` to the target width `dstWidth`:
/// - if srcWidth < dstWidth: zero-extend (concat zeros on the high side)
/// - if srcWidth > dstWidth: extract low bits
/// - else: identity
static Value adaptWidth(OpBuilder &b, Location loc, Value v, unsigned srcWidth,
                        unsigned dstWidth) {
  if (srcWidth == dstWidth) {
    return v;
  }
  if (srcWidth < dstWidth) {
    unsigned extBits = dstWidth - srcWidth;
    auto zeros = smt::BVConstantOp::create(b, loc, 0, extBits);
    auto concatTy = smt::BitVectorType::get(b.getContext(), dstWidth);
    return smt::ConcatOp::create(b, loc, concatTy, zeros, v);
  }
  // srcWidth > dstWidth: truncate to low bits
  auto extractTy = smt::BitVectorType::get(b.getContext(), dstWidth);
  return smt::ExtractOp::create(b, loc, extractTy, /*lowBit=*/0, v);
}

/// Inline the body of a func.func into the current insertion point, mapping
/// its arguments to the given ValueRange. Returns the values returned by the
/// function.
static SmallVector<Value> inlineFuncBody(OpBuilder &b, Location loc,
                                         func::FuncOp fn, ValueRange args) {
  IRMapping mapping;
  Block &entry = fn.getBody().front();
  assert(entry.getNumArguments() == args.size() && "arg count mismatch");
  for (auto [blockArg, argVal] : llvm::zip(entry.getArguments(), args)) {
    mapping.map(blockArg, argVal);
  }

  Operation *terminator = nullptr;
  for (Operation &op : entry.without_terminator()) {
    b.clone(op, mapping);
  }
  terminator = entry.getTerminator();
  auto returnOp = dyn_cast<func::ReturnOp>(terminator);
  assert(returnOp && "expected func.return as terminator");

  SmallVector<Value> results;
  for (Value v : returnOp.getOperands()) {
    results.push_back(mapping.lookup(v));
  }
  return results;
}

struct ConstructTTIRLECPass
    : public mlir::tt::impl::ConstructTTIRLECBase<ConstructTTIRLECPass> {
  using Base = mlir::tt::impl::ConstructTTIRLECBase<ConstructTTIRLECPass>;

  ConstructTTIRLECPass() : Base() {}

  ConstructTTIRLECPass(const mlir::tt::ConstructTTIRLECOptions &options)
      : Base() {
    this->firstFunc = options.firstFunc;
    this->secondFunc = options.secondFunc;
    this->checkOutput = options.checkOutput;
    this->checkOutputIdx = options.checkOutputIdx;
    this->nameAttr = options.nameAttr;
  }

  ConstructTTIRLECPass(const ConstructTTIRLECPass &rhs) : Base(rhs) {
    // Workaround: Passes are required to be copy-constructible but autogen'ed
    // base class copy constructors ignore Pass option fields.
    this->firstFunc = rhs.firstFunc;
    this->secondFunc = rhs.secondFunc;
    this->checkOutput = rhs.checkOutput;
    this->checkOutputIdx = rhs.checkOutputIdx;
    this->nameAttr = rhs.nameAttr;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (firstFunc.empty() || secondFunc.empty()) {
      module.emitError() << "construct-ttir-lec: -c1 and -c2 must be specified";
      return signalPassFailure();
    }

    auto fn1 = module.lookupSymbol<func::FuncOp>(firstFunc);
    auto fn2 = module.lookupSymbol<func::FuncOp>(secondFunc);
    if (!fn1) {
      module.emitError() << "function '" << firstFunc << "' not found";
      return signalPassFailure();
    }
    if (!fn2) {
      module.emitError() << "function '" << secondFunc << "' not found";
      return signalPassFailure();
    }

    auto fty1 = fn1.getFunctionType();
    auto fty2 = fn2.getFunctionType();

    if (fty1.getNumInputs() != fty2.getNumInputs()) {
      module.emitError() << "function input count differs: " << firstFunc
                         << " has " << fty1.getNumInputs() << ", " << secondFunc
                         << " has " << fty2.getNumInputs();
      return signalPassFailure();
    }
    if (fty1.getNumResults() != fty2.getNumResults()) {
      module.emitError() << "function result count differs";
      return signalPassFailure();
    }

    // Determine which output indices to compare. Default: all of them.
    SmallVector<size_t> selectedIndices;
    if (checkOutputIdx >= 0) {
      if (static_cast<size_t>(checkOutputIdx) >= fty1.getNumResults()) {
        module.emitError() << "check-output-idx " << checkOutputIdx
                           << " is out of range (function has "
                           << fty1.getNumResults() << " results)";
        return signalPassFailure();
      }
      selectedIndices.push_back(static_cast<size_t>(checkOutputIdx));
    } else if (!checkOutput.empty()) {
      // Search both functions' result attributes for the nameAttr key ==
      // checkOutput. Some lowering pipelines can strip result attrs from one
      // side, so we accept finding the name in either function.
      auto findInFunc = [&](func::FuncOp fn) -> ssize_t {
        ArrayAttr resAttrs = fn.getAllResultAttrs();
        if (!resAttrs) {
          return -1;
        }
        for (size_t i = 0; i < resAttrs.size(); ++i) {
          auto dict = dyn_cast<DictionaryAttr>(resAttrs[i]);
          if (!dict) {
            continue;
          }
          if (auto portName = dict.getAs<StringAttr>(nameAttr)) {
            if (portName.getValue() == checkOutput) {
              return static_cast<ssize_t>(i);
            }
          }
        }
        return -1;
      };
      ssize_t foundIdx = findInFunc(fn1);
      if (foundIdx < 0) {
        foundIdx = findInFunc(fn2);
      }
      if (foundIdx < 0) {
        module.emitError() << "check-output: no result port named '"
                           << checkOutput << "' in either '" << firstFunc
                           << "' or '" << secondFunc << "'";
        return signalPassFailure();
      }
      selectedIndices.push_back(static_cast<size_t>(foundIdx));
    } else {
      for (size_t i = 0; i < fty1.getNumResults(); ++i) {
        selectedIndices.push_back(i);
      }
    }

    // Compute per-input/output widths and the "common" (min) widths.
    // Width-adapter logic only applies to bitvector ports. Array ports must
    // have identical types between the two functions and are passed through
    // directly (no extension/truncation).
    SmallVector<unsigned> inWidths1, inWidths2, inCommonWidths;
    for (auto [t1, t2] : llvm::zip(fty1.getInputs(), fty2.getInputs())) {
      auto bv1 = dyn_cast<smt::BitVectorType>(t1);
      auto bv2 = dyn_cast<smt::BitVectorType>(t2);
      if (bv1 && bv2) {
        inWidths1.push_back(bv1.getWidth());
        inWidths2.push_back(bv2.getWidth());
        inCommonWidths.push_back(std::min(bv1.getWidth(), bv2.getWidth()));
        continue;
      }
      // Non-bv (e.g. smt.array): must match exactly. Encode width as 0 to
      // mean "passthrough"; adaptWidth treats 0->0 as identity.
      if (t1 != t2) {
        module.emitError() << "non-bitvector input types differ: " << t1
                           << " vs " << t2 << " (no width adapter possible)";
        return signalPassFailure();
      }
      inWidths1.push_back(0);
      inWidths2.push_back(0);
      inCommonWidths.push_back(0);
    }

    SmallVector<unsigned> outWidths1, outWidths2, outCommonWidths;
    for (auto [t1, t2] : llvm::zip(fty1.getResults(), fty2.getResults())) {
      auto bv1 = dyn_cast<smt::BitVectorType>(t1);
      auto bv2 = dyn_cast<smt::BitVectorType>(t2);
      if (bv1 && bv2) {
        outWidths1.push_back(bv1.getWidth());
        outWidths2.push_back(bv2.getWidth());
        outCommonWidths.push_back(std::min(bv1.getWidth(), bv2.getWidth()));
        continue;
      }
      if (t1 != t2) {
        module.emitError() << "non-bitvector result types differ: " << t1
                           << " vs " << t2;
        return signalPassFailure();
      }
      outWidths1.push_back(0);
      outWidths2.push_back(0);
      outCommonWidths.push_back(0);
    }

    // Create the LEC entry function: func.func @lec_main()
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToEnd(module.getBody());
    Location loc = module.getLoc();

    StringRef entryName = "lec_main";
    auto entryFnTy = builder.getFunctionType({}, {});
    auto entryFn = func::FuncOp::create(builder, loc, entryName, entryFnTy);
    Block *entryBlock = entryFn.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // smt.solver () : () -> ()  (no inputs, no outputs)
    auto solver =
        smt::SolverOp::create(builder, loc, TypeRange{}, ValueRange{});
    Block *solverBlock = builder.createBlock(&solver.getBodyRegion());
    builder.setInsertionPointToStart(solverBlock);

    // Declare a fresh symbolic input for each function arg. Bitvector args
    // are declared at the common (min) width and adapted per-circuit.
    // Non-bitvector args (e.g. smt.array) are declared with their original
    // type and shared between both circuits unchanged.
    SmallVector<Value> commonInputs;
    SmallVector<Value> args1, args2;
    auto inTypes = fty1.getInputs();
    for (size_t i = 0; i < inCommonWidths.size(); ++i) {
      std::string name = "arg" + std::to_string(i);
      Value declared;
      if (inCommonWidths[i] > 0) {
        auto bvTy =
            smt::BitVectorType::get(builder.getContext(), inCommonWidths[i]);
        declared = smt::DeclareFunOp::create(builder, loc, bvTy,
                                             builder.getStringAttr(name));
        commonInputs.push_back(declared);
        args1.push_back(adaptWidth(builder, loc, declared, inCommonWidths[i],
                                   inWidths1[i]));
        args2.push_back(adaptWidth(builder, loc, declared, inCommonWidths[i],
                                   inWidths2[i]));
      } else {
        // Non-bv (array) — declare with the matching type, no adapter.
        declared = smt::DeclareFunOp::create(builder, loc, inTypes[i],
                                             builder.getStringAttr(name));
        commonInputs.push_back(declared);
        args1.push_back(declared);
        args2.push_back(declared);
      }
    }

    // Inline both circuits.
    SmallVector<Value> outs1 = inlineFuncBody(builder, loc, fn1, args1);
    SmallVector<Value> outs2 = inlineFuncBody(builder, loc, fn2, args2);

    // For each selected output: adapt to common width (if bv), then assert
    // distinct. Non-bv outputs (arrays) are compared directly. We OR the
    // per-output distinct predicates: any mismatched output counts.
    Value anyDiff = smt::BoolConstantOp::create(builder, loc, false);
    for (size_t i : selectedIndices) {
      Value o1, o2;
      if (outCommonWidths[i] > 0) {
        o1 = adaptWidth(builder, loc, outs1[i], outWidths1[i],
                        outCommonWidths[i]);
        o2 = adaptWidth(builder, loc, outs2[i], outWidths2[i],
                        outCommonWidths[i]);
      } else {
        o1 = outs1[i];
        o2 = outs2[i];
      }
      Value diffI = smt::DistinctOp::create(builder, loc, o1, o2);
      anyDiff = smt::OrOp::create(builder, loc, ValueRange{anyDiff, diffI});
    }

    smt::AssertOp::create(builder, loc, anyDiff);

    // smt.check sat {} unknown {} unsat {} -> ()
    auto checkOp = smt::CheckOp::create(builder, loc, TypeRange{});
    builder.createBlock(&checkOp.getSatRegion());
    smt::YieldOp::create(builder, loc, ValueRange{});
    builder.createBlock(&checkOp.getUnknownRegion());
    smt::YieldOp::create(builder, loc, ValueRange{});
    builder.createBlock(&checkOp.getUnsatRegion());
    smt::YieldOp::create(builder, loc, ValueRange{});

    // Solver yields nothing.
    builder.setInsertionPointToEnd(solverBlock);
    smt::YieldOp::create(builder, loc, ValueRange{});

    // Entry function returns nothing.
    builder.setInsertionPointAfter(solver);
    func::ReturnOp::create(builder, loc, ValueRange{});

    // Erase the original two circuit funcs.
    fn1.erase();
    if (firstFunc != secondFunc) {
      fn2.erase();
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>>
createConstructTTIRLECPass(const ConstructTTIRLECOptions &options) {
  return std::make_unique<ConstructTTIRLECPass>(options);
}

} // namespace mlir::tt
