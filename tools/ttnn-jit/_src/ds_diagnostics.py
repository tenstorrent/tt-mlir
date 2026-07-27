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

Several of these are policy rather than kernel limits, and the messages say so,
because the distinction decides what you do about it: a policy limit has a
switch, a kernel limit needs a different config.

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


def _gate_failure(act, weight, has_bias, allow_bf16=False):
    """First DS eligibility gate that rejects, or None if all pass.

    Mirrors isDRAMShardEligible. Deliberately short: most limits that used to
    live here turned out to be policy or advisor-side constants and have since
    been removed, so anything not caught below is left to the op model, whose
    own failureReason is quoted instead of a reason invented here.
    """
    (a_dims, _a_dtype), (w_dims, w_dtype) = act, weight
    del has_bias  # a bias is fine: verified on silicon (PCC 0.9996)
    if len(w_dims) < 2:
        return f"weight shape {w_dims} is not a matrix"
    if any(d != 1 for d in w_dims[:-2]):
        return (f"weight {w_dims} is a batched matmul (per-expert weights); DS "
                f"takes a single [K, N] matrix")
    if len(a_dims) < 2:
        return f"activation shape {a_dims} is not a matrix"
    K, N = w_dims[-2], w_dims[-1]
    if K % TILE or N % TILE:
        return f"K/N = {K}/{N} not tile-aligned"
    if w_dtype not in ("bfp_bf4", "bfp_bf8") and not (allow_bf16 and w_dtype == "bf16"):
        if w_dtype == "bf16":
            return ("bf16 weights are not offered by default -- policy, not a "
                    "kernel limit (bf16 DS runs at PCC 1.0000). DS streams the "
                    "weights, so bf16 moves 2x bfp8's bytes. Enable with "
                    "--pipeline-options allow-bf16-dram-sharded-matmul=true, or "
                    "capture at the shipped precision if the model ships BFP.")
        return f"weight dtype is {w_dtype}, DS needs bfp_bf4/bfp_bf8"
    # No M gate and no K-divisibility gate: the in0-core count is chosen from K
    # (chooseNumIn0Cores) and tt-metal answers on the activation height itself.
    return None


def _matmul_rejections(trace_path):
    """[reason or None] per matmul-like op, in trace order.

    The trace keys entries by GLOBAL opIndex, not by matmul ordinal, so the two
    have to be zipped by walking forwardPass in order -- indexing the trace with
    the ordinal silently looked up an unrelated op.
    """
    try:
        with open(trace_path) as f:
            trace = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for entry in trace.get("forwardPass", []):
        if entry.get("opName") not in ("ttnn.matmul", "ttnn.linear"):
            continue
        reason = None
        for ev in entry.get("evaluations", []):
            ins = ev.get("inputs", [])
            # Only getExtraInputReshardCandidates injects this pair.
            if (len(ins) > 1 and "width_sharded>/1x" in ins[0]
                    and ins[0].startswith("l1")
                    and ins[1].startswith("dram") and "width_sharded" in ins[1]):
                r = (ev.get("failureReason") or "").strip()
                if r:
                    reason = " ".join(r.split())[:160]
                    break
        out.append(reason)
    return out


def analyze(final_ir, trace_path=None, allow_bf16=False):
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
            row["why"] = (_gate_failure(act, weight, has_bias, allow_bf16)
                          or "considered, but rejected")
            row["considered"] = row["why"] == "considered, but rejected"
        else:
            row["considered"] = True
        rows.append(row)

    # For ops that passed every gate, quote the op-model's actual reason.
    if trace_path:
        reasons = _matmul_rejections(trace_path)
        for i, row in enumerate(rows):
            if row.get("considered") and not row["advised"] and i < len(reasons):
                if reasons[i]:
                    row["why"] = reasons[i]

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


def render(final_ir, trace_path=None, allow_bf16=False):
    summary, rows = analyze(final_ir, trace_path, allow_bf16)
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
