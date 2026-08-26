"""Rebuild ttir_llama_3_2_1b_512_fwd.mlir from the tt-kurbla log attached to
tt-mlir#9189: take the first `===== TTIR module =====` block (the forward
graph) and give it the argument annotations a tt-xla TTIR dump carries.

    python extract_ttir.py [<log.txt> [<out.mlir>]]
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "321b512.txt")
DST = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "ttir_llama_3_2_1b_512_fwd.mlir")
MODULE_NAME = "llama_3_2_1b_512_fwd"
MODULE_ATTRS = 'attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>}'

with open(SRC) as f:
    lines = f.read().split("\n")

# The first "[tt_kurbla] ===== TTIR module =====" block is the forward graph.
start = next(i for i, l in enumerate(lines) if l.startswith("[tt_kurbla] ===== TTIR module ====="))
assert lines[start + 1] == "module {", lines[start + 1]
end = next(i for i in range(start + 2, len(lines)) if lines[i] == "}")
sig = lines[start + 2]
ops = lines[start + 3:end - 1]
assert lines[end - 1] == "  }", repr(lines[end - 1])
assert ops[-1].lstrip().startswith("return "), ops[-1][:60]

# Split the func signature into: args / results.
m = re.match(r"^  func\.func @main\((.*)\) -> \((.*)\) \{$", sig)
assert m, sig[:200]
args = re.split(r", (?=%arg)", m.group(1))
results = m.group(2)
assert all(re.fullmatch(r"%arg\d+: tensor<[^<>]*>", a) for a in args)

# Per-decoder-layer parameter order, as verified against the graph:
# q, k, v, o, gate, up, down, input_layernorm, post_attention_layernorm.
LAYER_PARAMS = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
]
NUM_LAYERS = 16
FIRST_LAYER_ARG = 1
LAST_PARAM_ARG = FIRST_LAYER_ARG + NUM_LAYERS * len(LAYER_PARAMS) + 1  # + model.norm, + inv_freq

INPUT_NAMES = {
    LAST_PARAM_ARG + 1: "input_ids",
    LAST_PARAM_ARG + 2: "attention_mask",
    LAST_PARAM_ARG + 3: "labels_one_hot",
    LAST_PARAM_ARG + 4: "loss_weight",
}


def arg_name(idx):
    if idx == 0:
        return "model.embed_tokens.weight"
    if FIRST_LAYER_ARG <= idx < FIRST_LAYER_ARG + NUM_LAYERS * len(LAYER_PARAMS):
        off = idx - FIRST_LAYER_ARG
        return "model.layers.%d.%s" % (off // len(LAYER_PARAMS), LAYER_PARAMS[off % len(LAYER_PARAMS)])
    if idx == LAST_PARAM_ARG - 1:
        return "model.norm.weight"
    if idx == LAST_PARAM_ARG:
        return "model.rotary_emb.inv_freq"
    return INPUT_NAMES[idx]


annotated = []
for a in args:
    idx = int(re.match(r"%arg(\d+):", a).group(1))
    ty = a.split(": ", 1)[1]
    kind = "parameter" if idx <= LAST_PARAM_ARG else "input"
    annotated.append(
        '%s {ttcore.argument_type = #ttcore.argument_type<%s>, '
        'ttcore.local_shape = #ttcore<local_shape local_shape = %s>, '
        'ttcore.shard_status = #ttcore.shard_status<unsharded>, '
        'ttir.name = "%s"}' % (a, kind, ty, arg_name(idx))
    )

out = []
out.append("module @%s %s {" % (MODULE_NAME, MODULE_ATTRS))
out.append("  ttcore.device_module {")
out.append("    builtin.module @%s %s {" % (MODULE_NAME, MODULE_ATTRS))
out.append("      func.func @main(%s) -> (%s) {" % (", ".join(annotated), results))
out.extend("    " + l if l else l for l in ops)
out.append("      }")
out.append("    }")
out.append("  }")
out.append("}")

with open(DST, "w") as f:
    f.write("\n".join(out) + "\n")

print("args:", len(args), "ops:", len(ops))
print("params:", sum(1 for a in annotated if "<parameter>" in a),
      "inputs:", sum(1 for a in annotated if "<input>" in a))
