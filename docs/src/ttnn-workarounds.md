# TTNN Workarounds

The current state of TTNN ops isn’t ideal for a generic compiler development. For example, element-wise operations and the matmul operation only work with tensors in the tile layout. As a result, we’ve added assumptions throughout the compiler and runtime to handle this. Tracking all these assumptions throughout the code has become difficult. Therefore, we developed a workaround framework for the TTNN dialect to systematically centralize all the necessary workarounds.

The workaround framework is implemented as a TTNN pass within the TTNN backend pipeline. This pass is executed after lowering the TTIR dialect to the TTNN dialect, ensuring that all necessary workarounds are systematically applied.

```c++
void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  createTTNNPipelineTTIRPasses(pm, options);
  createTTNNPipelineTTIRBroadcastFoldPass(pm, options);
  createTTNNPipelineLoweringPasses(pm, options);
  createTTNNPipelineWorkaroundPass(pm, options); // Workaround pass
  createTTNNPipelineAnalysisPasses(pm, options);
  createTTNNPipelineLayoutDecompositionPass(pm, options);
  createTTNNPipelineDeallocPass(pm, options);
}
```

Workaround pass consists of two main sub-passes:
1. Decomposition workaround pass
2. Layout workaround pass

## Decomposition workaround pass
Some TTNN operations have limitations that can be addressed by decomposing them into a set of other TTNN operations known to function correctly. To manage these workarounds, we introduced a decomposition sub-pass within the workaround pass.

### How to add new decomposition workaround
The decompositions are implemented by extending the `OpRewritePattern` and overriding the matchAndRewrite function of the inherited class.
```C++
class ReduceOpsKeepDimRewritePattern : public OpRewritePattern<ReduceOp> {
//...
  LogicalResult matchAndRewrite(ReduceOp srcOp,
                                PatternRewriter &rewriter) const override {
  // Decomposition logic...
  }
}
```

1. Define a decomposition workaround pattern in [/include/ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition](/include/ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition) folder.
2. Register your decomposition rewrite pattern in decomposition sub-pass in [/lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkarounds.cpp](/lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkarounds.cpp):
```C++
    if (decompositionWorkaroundsEnabled) {
      // Workaround decompositions
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNAllReduceWorkarounds,
                   workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
                       ttnn::SumOp>,
                   workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
                       ttnn::MaxOp>,
                   workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
                       ttnn::MeanOp>,
                   workarounds::decomposition::ReduceOpsAllDimsRewritePattern<
                       ttnn::SumOp>,
                   workarounds::decomposition::ReduceOpsAllDimsRewritePattern<
                       ttnn::MaxOp>,
                   workarounds::decomposition::ReduceOpsAllDimsRewritePattern<
                       ttnn::MeanOp>>(&getContext());

      runRewritePatterns(std::move(patterns),
                         GreedyRewriteConfig::kNoLimit /*maxIterations*/);
    }
```

## Layout workaround pass
As mentioned earlier, some TTNN operations do not support all tensor configurations. For instance, the matmul operation only supports inputs in a tile layout, while the maxpool operation only supports inputs in the bf16 data format. The layout workaround pass addresses these limitations by inserting `ttnn::to_layout` operations to adjust the inputs and outputs of the affected operations. This approach ensures that the workaround is localized to the specific operation requiring it.

The graphical representation of applying layout workarounds is presented here. Imagine that this op only supports row major input and produces a single row-major output:

```C++
            |                                           |
            | (input tile layout)                       | (input tile layout)
    -----------------                           -----------------
    |      Op X     |                           |   to_layout   |
    |               |                  =>       |               |
    -----------------                           -----------------
            | (output tile layout)                      | (output row-major layout)
            |                                           |
                                                -----------------
                                                |      Op X     |
                                                |               |
                                                -----------------
                                                        | (output row-major layout)
                                                        |
                                                -----------------
                                                |   to_layout   |
                                                |               |
                                                -----------------
                                                        | (output tile layout)
                                                        |
```

For now, following part of the layout are supported for workaround:
- Tensor Layout workaround: **Tile**, **RowMajor**
- Tensor Buffer Type workaround: **SystemMemory**, **DRAM**, **L1**
- Tensor Memory Layout workaround: **Interleaved**, **SingleBank**, **HeightSharded**, **WidthSharded**, **BlockSharded**
- Tensor Data Type workaround: **Float32**, **Float16**, **BFloat16**, **BFP_Float8**, **BFP_BFloat8**, **BFP_Float4**, **BFP_BFloat4**, **BFP_Float2**, **BFP_BFloat2**, **UInt32**, **UInt16**, **UInt8**

### How to add new layout workaround

Layout workarounds are implemented with the following MLIR op interface defined for each TTNN op:
```C++
TTNNOperandsWorkarounds mlir::tt::ttnn::wa::TTNNWorkaroundInterface::getOperandsWorkarounds() {}
```

By default each op has an empty list of workarounds, but in case of need we can easily override the default implementation.
1. Create a factory method for your op workaround in [/include/ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h](/include/ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h). The purpose of this is to locate all the op layout workarounds in one central place:
```C++
// Workaround factory class that creates workarounds for ops.
class TTNNOperandsWorkaroundsFactory {
public:
/// ...
  // Create workarounds for matmul op operands.
  static TTNNOperandsWorkarounds createMatmulOpOperandsWorkarounds();
/// ...
}
```
2. Implement the factory method in [/lib/Dialect/TTNN/IR/TTNNWorkarounds.cpp](/lib/Dialect/TTNN/IR/TTNNWorkarounds.cpp). Ensure to create issues for the workarounds on the tt-metal side and link them in the comments for tracking purposes. This will help in removing the workarounds when they are no longer needed.
```C++
// ...
// Factory method to create a set of workarounds for matmul operation operands.
// The matmul operation expects the input to be in tile layout and it produce
// output in tile layout.
//
// Metal issue for input and output operand workaround:
// TBD
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createMatmulOpOperandsWorkarounds() {
  TTNNOperandWorkarounds tileLayoutWorkaround = TTNNOperandWorkarounds(Layout::Tile);
  return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(0, 0)
      .addInputOperandWorkaround(tileLayoutWorkaround)      // Input A workaround
      .addInputOperandWorkaround(tileLayoutWorkaround)      // Input B workaround
      .addInputOperandWorkaround(tileLayoutWorkaround)      // DPS output workaround
      .addOutputOperandWorkaround(tileLayoutWorkaround);    // Output workaround
}
// ...
```
  <span style="color:gray">**Note 1**</span>: Workarounds must be applied to each operand in the operation. The order of the workarounds should match the order of operands as defined in the MLIR TableGen file for the operation. For example, in the TableGen file for the matmul operation, the inputs are defined as input $a, input $b, and DPS $output.
  <br/><span style="color:gray">**Note 2**</span>: If an operand does not require a workaround, an empty workaround must be provided in the corresponding position according to the defined order of inputs/outputs.
  <br/><span style="color:gray">**Note 3**</span>: DPS input and output workarounds must match, and this is enforced by the interface verifier.

3. Override MLIR interface method in a [TTNN op table definition file](/include/ttmlir/Dialect/TTNN/IR/TTNNOps.td) by calling the above factory method:
```td
def TTNN_MatmulOp : TTNN_NamedDPSOp<"matmul"> {
    // ...

    let extraClassDeclaration = [{
      // ...
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createMatmulOpOperandsWorkarounds();
      }
      // ...
    }];

    // ...
}
```
  <span style="color:gray">**Note 1**</span> About extraClassDeclaration section, in TableGen, when you inherit from a parent definition, fields are overridden, not extended. If you redefine a field or a list in a child definition, it completely replaces the parent’s value, so don't forget to copy\paste all other defined functions from a parent.

4. Add a test under the [/test/ttmlir/Dialect/TTNN/Transforms/Workarounds/](/test/ttmlir/Dialect/TTNN/Transforms/Workarounds) to verify your workaround.
