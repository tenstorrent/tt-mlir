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
  if (srcWidth == dstWidth)
    return v;
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
  for (auto [blockArg, argVal] : llvm::zip(entry.getArguments(), args))
    mapping.map(blockArg, argVal);

  Operation *terminator = nullptr;
  for (Operation &op : entry.without_terminator()) {
    b.clone(op, mapping);
  }
  terminator = entry.getTerminator();
  auto returnOp = dyn_cast<func::ReturnOp>(terminator);
  assert(returnOp && "expected func.return as terminator");

  SmallVector<Value> results;
  for (Value v : returnOp.getOperands())
    results.push_back(mapping.lookup(v));
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
  }

  ConstructTTIRLECPass(const ConstructTTIRLECPass &rhs) : Base(rhs) {
    // Workaround: Passes are required to be copy-constructible but autogen'ed
    // base class copy constructors ignore Pass option fields.
    this->firstFunc = rhs.firstFunc;
    this->secondFunc = rhs.secondFunc;
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

    // Compute per-input/output widths and the "common" (min) widths.
    SmallVector<unsigned> inWidths1, inWidths2, inCommonWidths;
    for (auto [t1, t2] :
         llvm::zip(fty1.getInputs(), fty2.getInputs())) {
      auto bv1 = dyn_cast<smt::BitVectorType>(t1);
      auto bv2 = dyn_cast<smt::BitVectorType>(t2);
      if (!bv1 || !bv2) {
        module.emitError() << "input types must be smt.bv (run "
                              "--convert-ttir-to-smt first)";
        return signalPassFailure();
      }
      inWidths1.push_back(bv1.getWidth());
      inWidths2.push_back(bv2.getWidth());
      inCommonWidths.push_back(std::min(bv1.getWidth(), bv2.getWidth()));
    }

    SmallVector<unsigned> outWidths1, outWidths2, outCommonWidths;
    for (auto [t1, t2] :
         llvm::zip(fty1.getResults(), fty2.getResults())) {
      auto bv1 = dyn_cast<smt::BitVectorType>(t1);
      auto bv2 = dyn_cast<smt::BitVectorType>(t2);
      if (!bv1 || !bv2) {
        module.emitError() << "result types must be smt.bv";
        return signalPassFailure();
      }
      outWidths1.push_back(bv1.getWidth());
      outWidths2.push_back(bv2.getWidth());
      outCommonWidths.push_back(std::min(bv1.getWidth(), bv2.getWidth()));
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
    auto solver = smt::SolverOp::create(builder, loc, TypeRange{}, ValueRange{});
    Block *solverBlock = builder.createBlock(&solver.getBodyRegion());
    builder.setInsertionPointToStart(solverBlock);

    // Declare a fresh symbolic input for each function arg, at the common
    // (min) width. Then build per-circuit arg lists with width adapters.
    SmallVector<Value> commonInputs;
    SmallVector<Value> args1, args2;
    for (size_t i = 0; i < inCommonWidths.size(); ++i) {
      auto bvTy = smt::BitVectorType::get(builder.getContext(),
                                          inCommonWidths[i]);
      std::string name = "arg" + std::to_string(i);
      auto declared =
          smt::DeclareFunOp::create(builder, loc, bvTy,
                                    builder.getStringAttr(name));
      commonInputs.push_back(declared);
      args1.push_back(adaptWidth(builder, loc, declared, inCommonWidths[i],
                                 inWidths1[i]));
      args2.push_back(adaptWidth(builder, loc, declared, inCommonWidths[i],
                                 inWidths2[i]));
    }

    // Inline both circuits.
    SmallVector<Value> outs1 = inlineFuncBody(builder, loc, fn1, args1);
    SmallVector<Value> outs2 = inlineFuncBody(builder, loc, fn2, args2);

    // For each output: adapt to common width, then assert distinct.
    // We OR the per-output distinct predicates: any mismatched output counts.
    Value anyDiff = smt::BoolConstantOp::create(builder, loc, false);
    bool first = true;
    (void)first;
    for (size_t i = 0; i < outCommonWidths.size(); ++i) {
      Value o1 = adaptWidth(builder, loc, outs1[i], outWidths1[i],
                            outCommonWidths[i]);
      Value o2 = adaptWidth(builder, loc, outs2[i], outWidths2[i],
                            outCommonWidths[i]);
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
    if (firstFunc != secondFunc)
      fn2.erase();
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>>
createConstructTTIRLECPass(const ConstructTTIRLECOptions &options) {
  return std::make_unique<ConstructTTIRLECPass>(options);
}

} // namespace mlir::tt
