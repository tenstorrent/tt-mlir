# Optimizer Stack Refactoring Proposal

## Summary

We propose replacing the current DF Sharding Policy and ShardSolver with a simpler pass-based architecture. Our empirical analysis across 50+ models shows that a greedy approach can match or exceed current performance while being significantly easier to understand and maintain.

## Motivation

The current optimizer architecture has grown complex over time. The combination of L1 chain building, chain merging, and ShardSolver constraint propagation creates a system that is difficult to reason about, debug, and extend. More importantly, our analysis reveals that this complexity doesn't translate to better results—the sophisticated backtracking mechanism rarely provides practical benefit.

## Design Philosophy

- **Simple mental model:** "Keep data in L1 unless an operation requires otherwise, then return to L1 as soon as possible."
- **Defer to the backend:** Let the backend decide optimal configs and layouts via its query APIs. The optimizer's job is to respect those choices, not second-guess them.

## Problems with Current Approach

### Operand 0 Limitation

The ShardSolver only propagates sharding decisions through operand 0 edges. This means that for binary operations like `subtract(a, b)`, if operand 0 cannot be sharded, the solver ignores operand 1 entirely—even if it's perfectly valid to keep operand 1 in L1. Our analysis found this causes significant unnecessary spills.

### All-or-Nothing Chain Failure

When any edge in an L1 chain fails validation, the entire chain spills to DRAM. There's no mechanism for partial success—a single incompatible operation forces all connected operations out of L1 memory.

### Designed for Linear Chains

The ShardSolver's bitset-based constraint propagation assumes linear chain structure. Complex graph topologies like forks, joins, and diamonds require special-case handling outside the solver, adding to the overall complexity.

### Backtracking Provides Little Practical Benefit

In theory, the constraint satisfaction approach allows the solver to backtrack and find valid configurations. In practice, operations typically either keep their input layout or require a reshard—there's rarely a meaningful choice to optimize over.

## Empirical Findings

We analyzed the compiled IR for 50+ models including Segformer, ResNet50, and 45 LLM variants (Llama, Falcon, Gemma, Phi, Qwen). Key findings:

**Memory pressure is rare.** All tested models show 40-94% L1 headroom. The L1 budget of ~1364 KB per core is rarely stressed.

**Spills are constraint-driven, not memory-driven.** The vast majority of spills occur because specific operations require DRAM inputs (reduce ops, permute, reshape), not because L1 is full.

**Unnecessary spills have clear causes:**
- Operand 0 limitation: 36-65 spills per model in patterns where one operand is DRAM while another could stay in L1
- Fork handling: 15 spills in ResNet50 residual connections that could remain in L1 with proper liveness tracking

**Greedy decisions would have been correct.** In every case we analyzed, the optimal choice was apparent from local information—no backtracking was needed to find it.

## Proposed Architecture

We propose two independent passes with clear responsibilities:

### Pass 1: L1 Sharding Propagation

A simple edge-based sharding picker. For each operation in schedule order:

1. Look at input edges and their layouts
2. Enumerate valid (config, layout) pairs for the op
3. Pick the first pair that works with the inputs
4. If no L1 layout works, fall back to DRAM

This pass considers all operands (fixing the operand 0 limitation) but has no notion of chains or memory pressure. It simply propagates layouts edge by edge, inserting reshards where layouts are incompatible.

**Note on op configs:** This pass also selects op-specific configs (conv2d, matmul, compute configs). For layouts, we already delegate to the backend—the query op constraints API returns the output layout given inputs. The same should apply to op configs: the backend should auto-select optimal configs given input layouts. Today this isn't possible because the allocator is a dummy when query APIs are invoked. Once the allocator is integrated into the query path, both layouts and configs become fully backend-driven.

### Pass 2: L1 Spill Management

Decides when to spill tensors out of L1. Uses a greedy strategy that keeps L1 as saturated as possible while respecting the memory budget.

**Core strategy:**
- Track live tensors and their sizes
- Keep tensors in L1 as long as possible
- When budget is exceeded, spill the tensor with longest remaining lifetime

**Fork handling (borrowed from DF sharding 2.0):**
- Allow fork tensors to stay in L1 for their full lifetime when space permits
- Modify op configs as needed (e.g., conv2d `deallocate_activation=false` for fork inputs)

**Memory hierarchy:**
- Use L1 Interleaved when sharding is not feasible
- Fall back to DRAM only when necessary

## Simplicity Benefits

The proposed architecture is substantially simpler:

- No chain state machine or chain merging logic
- No bitset-based constraint solver
- Each pass has a single, well-defined responsibility
- Decisions are local and easy to trace
- Fewer special cases for graph patterns

This simplicity translates to faster development, easier debugging, and more predictable behavior.

## Optimization Strategies

While greedy allocation provides a functional baseline, the pass-based architecture enables more sophisticated strategies to reach parity with the current optimizer.

### Beam Search for Layout Propagation

Pure greedy (K=1) can lock in suboptimal choices early. For example, an early op might choose 32-core sharding because it avoids a reshard, but this propagates forward and forces downstream matmuls to also use 32 cores—losing significant compute throughput.

**The problem:** Greedy's strategy is "use working config without reshard, fall back to reshard only if none exists." This avoids reshards but may miss globally better paths.

**Solution:** Beam search with K candidates (e.g., K=4 or K=8) per op. Beam search has two phases:

**Forward phase (candidate selection):** Process ops in schedule order. For each op:
1. Enumerate candidates from configs compatible with input layouts (no reshard) and configs requiring reshards but enabling more cores
2. For binary ops, consider combinations from both inputs (K × K pairs)
3. Score all candidates, keep top K
4. Store back-pointers to parent candidates

**Backward phase (trace-back):** Starting from leaf nodes, trace back through best candidates to reconstruct the optimal path. At fork points, resolve conflicts (see below).

**Scoring (heuristic mode):** Without device access, use core count as proxy:
- Primary: maximize `minCores` (bottleneck core count on path)
- Tiebreaker: minimize `reshardCount`

**Scoring (cost mode, opt level 3):** With device access, use `getOpRuntime()` for actual runtime estimates. Score = accumulated runtime. This enables precise tradeoffs but is slower and requires device.

**Complexity:** O(K² × n) where K = beam width, n = number of ops. The K² factor comes from binary ops where we evaluate K × K input combinations. For ops with more inputs (e.g., concat with 4-5 operands), the combinations remain tractable since K is small (4-8).

**Why this reaches parity with current optimizer:** ShardSolver explores configurations via constraint propagation and backtracking. Beam search achieves similar exploration with bounded complexity, but considers all operands and doesn't suffer from chain-level failures.

**Handling fork points:** During backward trace-back, different consumer paths may prefer different layouts from a forked tensor:

```
        fork_op (keeps K candidates: [HS, BS, WS, ...])
           /              \
      consumer_A      consumer_B
      (path wants HS) (path wants BS)
```

At each fork during trace-back:
1. Collect what layout each consumer path wants
2. For each of fork's K candidates, compute total reshard cost to satisfy all consumers
3. Pick the candidate with minimum total reshard cost

This is a local decision—no tree traversal needed. Beam search reduces the global problem to local decisions by preserving K candidates at each op.

### Dynamic Programming for Optimal Spill Selection

When L1 pressure exists and multiple tensors compete for limited space, the spill decision becomes a classic register allocation problem. A DP-based approach can find the globally optimal set of tensors to keep in L1:

**Problem formulation:** Given a schedule of operations and their tensor lifetimes, select which tensors to keep in L1 at each point such that total memory never exceeds budget and total spill cost is minimized.

**DP state:** At each operation in the schedule, track which subset of live tensors are in L1. Transitions occur when tensors become live (allocate or spill) or die (deallocate).

**Cost model:** Assign costs to spills based on tensor size and access patterns. Tensors accessed multiple times have higher spill cost than single-use tensors.

This approach guarantees optimal allocation but has exponential complexity in the number of simultaneously live tensors. For most models this is tractable (typically 5-15 live tensors), but may need pruning heuristics for complex graphs.

### Progression Path

**Phase 1 - Greedy (MVP):** Start with pure greedy (K=1). This fixes the operand 0 limitation and fork handling issues. Validates the pass-based architecture with minimal complexity. Sufficient for models where early layout choices don't constrain downstream ops.

**Phase 2 - Beam Search with Heuristics (Parity):** Add beam search (K=4 or K=8) with heuristic scoring: maximize cores, break ties by reshard count. Explores reshard paths even when reshard-free paths exist. No device access needed, fast. Expected to match or exceed current optimizer quality.

**Phase 3 - Beam Search with Cost Mode (Opt Level 3):** Enable `getOpRuntime()` for actual runtime estimates. Precise cost-based tradeoffs between reshards and compute. Requires device access, slower, but more accurate for complex models.

**Phase 4 - DP Extensions (Edge Cases):** For models with genuine memory pressure, add DP-based spill selection. Our empirical data (40-94% headroom) suggests this is rarely needed, but the architecture supports it.

## Conclusion

The current optimizer's complexity is not justified by its results. A simpler, pass-based approach can achieve the same or better outcomes while being easier to understand, maintain, and extend. Greedy provides a working foundation; beam search provides parity with current optimizer; and the architecture supports further extensions if needed. The empirical evidence strongly supports this direction.
