Numerical debugging of MLIR graphs

Section 1 - Overview
1.1 Description and Purpose


Training models through the forge stack doesn’t support intermediate tensor analysis which is an essential feature for the model training team.

During inference, numerical errors are only accumulated through one call of the graph; during training, forward pass is connected to backward pass, and backward is connected to optimizer pass that modifies the weights of the forward pass. This process runs in a loop. Numerical errors are accumulated, so we need more sophisticated tools for debugging numerical problems.
1.2 Scope/Goals

The scope of this project is to create a tool needed to debug numerical stability issues of the compiler OPs. It should be able to identify the source of the issue in all of the possible scenarios of OP problems in a single graph.

1.3 Non-goals
1.3.1 Compile-time debugging support
Currently, we won’t be relying on CPU hoisting, where we would add parallel CPU execution as well as golden comparison in the MLIR graph.


1.3.2 Multi-turn error accumulation
Debugging of errors that accumulate through multiple graph executions, or multiple minibatch iterations.

Section 2 - Proposed Solution
2.1 Debugging Levels

We need multiple levels of debugging:
Level 1 that works in standard runtime through current frontends (ffe, tt-xla, tt-torch…) that does not add too much overhead, so we can run it in parallel with model training/inference.
It is used to find pcc errors on graph outputs.
Level 2 is used when we notice errors in PCC in level 1 and want finer debugging. For this, we will use standalone ttrt.
2.1.1 Level 1 - Unified frontends with simple graph output comparisons
As a first level of debugging, we propose ModelDebugger, a tool designed to validate and compare multiple model implementations across different frameworks. It entails basic debugging and verification across the forward, backward, and optimization stages of model execution.



Execution Pipeline:
Forward Pass: Model and Golden Model receive the same input. Their outputs are collected.
Output Verification: Outputs are compared against the Golden Model outputs for discrepancies.
Backward Pass: Gradients are computed.
Gradient Verification: Gradients are verified to match the Golden Model’s gradients.
Optimization Step: Weights are updated.
State Verification: Golden comparison of optimizer state updates
Weight Verification: Post-update weights are compared to ensure consistency.
Dumping Mechanism:
At any verification stage (outputs, gradients, optimizer states, weights), mismatches can trigger a dump operation for in-depth offline analysis. During the dumping step we need the following files:
report.csv: contains the PCC and other metrics gathered from output/gradient/weights comparison.
log.txt: Log file capturing debugging details, and runtime information in free form.
flatbuffer.ttnn: Binary used to run the compiled program with the input tensors embedded within
ttir.mlir: TTIR representation of the computation graph
ttnn.mlir: TTNN representation of the computation graph
2.1.2 Level 2 - Debugging the graph

To address the issues of numerical instability we need to have multiple ways to track errors across the computation graph. In order to address all of the possible types of errors, we propose several outpwut propagation flows of a single op. The objective of this is to isolate the malfunctioning operations.
Parallel Flow

In this mode we propagate respective inputs to device and CPU. Using just this propagation type, we have accumulated PCC error, the same one that we measured in the previous step in ModelDebugger just this time we are measuring for each intermediate.
TT Propagation

Here, we propagate the output computed on the device to CPU OP.

CPU Propagation

In this mode we are essentially skipping the OP1 and enabling checking the rest of the graph for further numerical errors.

2.2 Level 2 - Case Studies

To illustrate an effective utilization of the propagation strategies, we’ll examine two case studies. In each of them, the problem would first manifest itself in the ModelDebugger, on the output of a stage (fwd, bwd, opt). Then, we would load the graph in our tool, and try to pinpoint the source of the issue.

2.2.1 Bad OP

Assume that we have a model that suddenly has a PCC drop due to a bad OP. First, we could “pause” the execution, and skip that node by setting the proper propagation flow types on the neighbouring nodes. If the problem persists, we could repeat this process. Output of this process would be a list of isolated nodes (OPs) that are malfunctioning.

PCC value on OP 3 would be an actual PCC value of that op in vacuum and not the accumulated error.

2.2.2. Accumulation Error

In this case we have bad output pcc due to small accumulation errors that occur within the model.
Here, we could use a strategy where we single out every single OP in a similar fashion to the previous approach. Afterwards, we can determine statistics (mean, max, z-value…) for different OP categories.
OP categories could just be a differentiation between OP types (add, matmul…), or OP type + config (layout, fidelity, dtype…). We could then swap out the outputs of these OPs with golden outputs and re-verify the model.


Section 3 - Technical Details
3.1 CPU Eval
For the cpu eval we would want to use the TTIRBuilder. It has mapping from TTIR ops to torch ops that we need for this. What is left is to adapt TTIRBuilder not to execute the TTIR op as we are already in the graph and we don't need that functionality. This needs further verification with mlir tooling team. Also we need to adapt TTIRBuilder to execute ops on the fly as we need it to have the ability to propagate different types of inputs.
3.2 TTRT Pre-Op and Post-Op Callbacks Description and Purpose
Currently there is a callback in ttrt --debug that is executed after the op. To support parallel flow and cpu-propagation, we need the callback before op calling in case there are some in-place ops so we can save inputs into the op.
Currently there is an issue for this feature request 2594.

3.3 TT-Explorer

All of the proposed functionalities can be visualized through tt-explorer.
Currently there is support to visualize pcc errors. It is left to be seen in what exact ways different strategies and debugging flows can be visualized. This needs further verification with mlir tooling team.

3.4 Project Structure

For the proposed solution we’ve envisioned that the project structure would look like the following:
ModelDebugger: a tool that wraps around different frontends and provides level 1 debugging. It doesn’t rely on anything other than the frontends.

GraphChisel: a tool that provides level 2 debugging; it relies on ttrt and ttir builder to manage the graph execution and the flows described above.

TT-Explorer: ideally, as described above, the user wouldn’t have to interact with GraphChisel manually and would be able to utilize its functionalities through TT-Explorer. The minimal integration with TT-Explorer would be to just visualize the PCC results.


3.4.1 Draft API specification


ModelDebugger
forward()
Calls the appropriate frontend forward on both CPU and TT
backward()
Calls the appropriate frontend backward on both CPU and TT
optimizer()
Calls the appropriate frontend optimizer on both CPU and TT
_dump()
Does the dumping mechanism described in 2.1.1
_verify()
Does verification on the outputs of the programs



GraphChisel
--input-dir
Points to the dump dir specified in ModelDebugger
--output-dir
Specifies the directory where the results will be stored
--debugging-mode
For now, we propose having
[“bad_op”, “accumulation”] described in section 2.2
--run-explorer
Starts tt-explorer and visualizes the results
--op-config
Custom override of propagation flows


Section 4 - Future Improvements
4.1 Autosearch

All of the processes described above could be automated by automatically swapping out the OPs if a certain criteria is met (similar to what has been done in 2.2.1 and 2.2.2).
The output of the tool could be a list of all problematic OPs in a graph, along with their issues.

4.2 OP Patterns

Potentially, certain patterns of numerically sensitive OPs could produce accumulated errors, and finding those problematic patterns is a topic for further discussion.
We could imagine a scenario where having patterns of OPs can have troublesome effects. For example, imagine that we have a pattern of matmul -> exp. Exp is a highly sensitive OP, and repeating this pattern would probably cause issues. It would be useful to have the tool automatically detect problems that occur because of that.
