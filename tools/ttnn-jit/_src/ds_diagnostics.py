# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Explain, per matmul, whether DRAM-sharding was advised — and if not, why.

Without this the advisor reports a silent zero: a capture whose weights are
bf16, or whose decode batch is 1, can never receive DRAM-sharded advice, and
nothing in report.txt says so. That is a real trap -- a QB2 capture built its
weights in bf16 "because dtype is not an advisor decision", which was true until
DRAM-sharding landed and made dtype *the* gate. It looked like "the advisor
considered DRAM-sharding and declined".

The gates mirror `isDRAMShardEligible` in
lib/Dialect/TTNN/Analysis/OpRules/MatmulRules.cpp. Keep them in sync; this is a
diagnostic, so when in doubt it reports "not-considered (unknown)" rather than
inventing a reason.
"""
import json
import re

TILE = 32
NUM_IN0_CORES = 8  # kNumIn0Cores

# "ttnn.linear"(%a, %b) <{...}> : (tensor<...>, tensor<...>[, tensor<...>]) -> tensor<...>
_OP_RE = re.compile(r'"(ttnn\.(?:matmul|linear))"\(([^)]*)\)(.*)$')
_SIG_RE = re.compile(r":\s*\((.*)\)\s*->")
_TENSOR_RE = re.compile(r"tensor<([^,]+(?:,\s*!ttcore\.tile<[^>]*>)?)[^>]*>")


def _tensor_operands(signature):
    """Split a `(tensor<..>, tensor<..>)` input signature into shape/dtype pairs."""
    out, depth, cur = [], 0, ""
    for ch in signature:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return [o for o in out if o.startswith("tensor<")]


def _parse_tensor(text):
    """-> (dims, dtype) from `tensor<1x1x32x4096xbf16, #layout>` or a tile type."""
    body = text[len("tensor<"):]
    tile = re.search(r"!ttcore\.tile<\d+x\d+,\s*([a-z_0-9]+)>", body)
    head = body.split(",")[0]
    parts = head.split("x")
    dims, dtype = [], None
    for p in parts:
        if re.fullmatch(r"\d+", p):
            dims.append(int(p))
        else:
            dtype = p.strip()
            break
    if tile:
        dtype = tile.group(1)
    return dims, (dtype or "?")


def _activation_rows(act_dims):
    """Logical row count M of the activation (all dims but the last)."""
    m = 1
    for d in act_dims[:-1]:
        m *= d
    return m


def _gate_failure(act, weight, has_bias):
    """First DS eligibility gate that rejects, or None if all pass."""
    (a_dims, _a_dtype), (w_dims, w_dtype) = act, weight
    if has_bias:
        return "linear has a bias operand (outside the DS contract)"
    if w_dtype not in ("bfp_bf4", "bfp_bf8"):
        return (f"weight dtype is {w_dtype}, DS needs bfp_bf4/bfp_bf8 "
                f"-- does the capture match the shipped precision?")
    if len(w_dims) < 2 or any(d != 1 for d in w_dims[:-2]):
        return f"weight shape {w_dims} is a batched matmul (leading dims must be 1)"
    if len(a_dims) < 2:
        return f"activation shape {a_dims} is not a matrix"
    K, N = w_dims[-2], w_dims[-1]
    M = _activation_rows(a_dims)
    if K % TILE or N % TILE:
        return f"K/N = {K}/{N} not tile-aligned"
    if (K // TILE) % NUM_IN0_CORES:
        return (f"K={K} -> {K // TILE} tiles is not divisible by the "
                f"{NUM_IN0_CORES} in0 cores")
    # No M gate: the rule book offers a DS candidate for any batch and lets the
    # op model answer. tt-metal currently accepts only an in0 height of one tile
    # ("M == 1 ... currently only support in0 tensor height of tile height",
    # matmul_device_operation.cpp), so M > 32 comes back as a normal rejection
    # carrying that message rather than a gate invented here.
    return None


def _rejection_from_trace(trace_path, op_index):
    """The op-model's own reason for rejecting the canonical DS candidate."""
    try:
        with open(trace_path) as f:
            trace = json.load(f)
    except (OSError, ValueError):
        return None
    for entry in trace.get("forwardPass", []):
        if entry.get("opIndex") != op_index:
            continue
        for ev in entry.get("evaluations", []):
            ins = ev.get("inputs", [])
            # Only getExtraInputReshardCandidates injects this pair.
            if (len(ins) > 1 and ins[0].endswith(f"width_sharded>/1x{NUM_IN0_CORES}")
                    and ins[1].startswith("dram") and "width_sharded" in ins[1]):
                reason = (ev.get("failureReason") or "").strip()
                if reason:
                    return " ".join(reason.split())[:160]
    return None


def analyze(final_ir, trace_path=None):
    """-> (summary dict, list of per-matmul dicts)."""
    rows = []
    for line in final_ir.splitlines():
        m = _OP_RE.search(line)
        if not m:
            continue
        op_name, _operands, rest = m.groups()
        sig = _SIG_RE.search(rest)
        if not sig:
            continue
        tensors = _tensor_operands(sig.group(1))
        if len(tensors) < 2:
            continue
        act, weight = _parse_tensor(tensors[0]), _parse_tensor(tensors[1])
        has_bias = op_name == "ttnn.linear" and len(tensors) > 2
        advised = "dram_sharded" in rest
        row = {"op": op_name, "advised": advised, "activation_rows": _activation_rows(act[0]),
               "weight_shape": weight[0], "weight_dtype": weight[1]}
        if not advised:
            row["why"] = _gate_failure(act, weight, has_bias) or "considered, but rejected"
            row["considered"] = row["why"] == "considered, but rejected"
        else:
            row["considered"] = True
        rows.append(row)

    # For ops that passed every gate, quote the op-model's actual reason.
    if trace_path:
        idx = 0
        for line in final_ir.splitlines():
            if not _OP_RE.search(line):
                continue
            if idx < len(rows) and rows[idx].get("considered") and not rows[idx]["advised"]:
                detail = _rejection_from_trace(trace_path, idx)
                if detail:
                    rows[idx]["why"] = detail
            idx += 1

    advised = sum(1 for r in rows if r["advised"])
    considered = sum(1 for r in rows if r["considered"])
    # Rows below a full tile are exactly the ones the old gate rejected: it
    # required M % 32 == 0, so only batch 32 qualified. Flagging them is
    # attribution -- "this pick is newly reachable" -- not a general warning.
    # (M > 32 is still refused, but by tt-metal, and it says so itself.)
    sub_tile = sorted({r["activation_rows"] for r in rows
                       if r["advised"] and 0 < r["activation_rows"] < TILE})
    return ({"matmuls": len(rows), "dram_sharded_advised": advised,
             "dram_sharded_considered": considered,
             "sub_tile_batch_rows": sub_tile}, rows)


def render(final_ir, trace_path=None):
    summary, rows = analyze(final_ir, trace_path)
    if not rows:
        return "", summary, rows
    out = [f"=== DRAM-sharded matmuls: {summary['dram_sharded_advised']} of "
           f"{summary['matmuls']} ({summary['dram_sharded_considered']} considered) ==="]
    for i, r in enumerate(rows):
        if r["advised"]:
            out.append(f"  [{i}] {r['op']} {r['weight_shape']} {r['weight_dtype']}  -> DRAM-sharded")
        else:
            out.append(f"  [{i}] {r['op']} {r['weight_shape']} {r['weight_dtype']}  -> no: {r['why']}")
    if summary.get("sub_tile_batch_rows"):
        rows_str = ", ".join(str(m) for m in summary["sub_tile_batch_rows"])
        out.append(f"  NOTE: {rows_str} activation row(s), under a full 32-row tile — DS here was")
        out.append("        withheld until the one-tile-row gate started rounding up, so these are")
        out.append("        newly reachable picks rather than a re-confirmation.")
    if summary["dram_sharded_considered"] == 0 and rows:
        out.append("  NOTE: DRAM-sharding was never even a candidate here. That is a property")
        out.append("        of the capture, not a verdict on the model -- fix the reason above")
        out.append("        and re-capture before concluding anything about DRAM-sharding.")
    return "\n".join(out) + "\n", summary, rows
