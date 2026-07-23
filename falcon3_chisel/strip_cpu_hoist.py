#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Remove CPU-hoisted const-eval from a TTNN-dialect .mlir so it can be
translated by `ttmlir-translate --ttnn-to-flatbuffer` (which does not register
the TTIR dialect used inside `ttcore.cpu_module`).

The single hoisted function is a numeric no-op (reshape 128 -> 1x1x1x128 on a
system-memory f32 tensor). We:
  1. Replace the `call @cpu_hoisted_...  {ttir.cpu_hoist_call}` in the const-eval
     function with an on-device `ttnn.reshape` producing the same result type.
  2. Delete the `ttcore.cpu_module { ... }` block (balanced-brace removal).
  3. Delete the `forward_cpu_declaration` private func.func line.
"""
import re
import sys


def strip(text: str) -> str:
    # 1. Replace the cpu_hoist_call with ttnn.reshape.
    #    Grab the call line and rewrite it, deriving the target rank-4 shape
    #    from the result tensor type.
    call_re = re.compile(
        r'(%\w+)\s*=\s*call @cpu_hoisted_const_eval_\w+\((%\w+)\)\s*'
        r'\{ttir\.cpu_hoist_call\}\s*:\s*\((tensor<[^)]*?>)\)\s*->\s*(tensor<[^>]*?>)'
    )

    def repl(m):
        res_name, arg_name, in_ty, out_ty = m.groups()
        # out_ty like tensor<1x1x1x128xf32, #ttnn_layout4>
        dims = re.search(r'tensor<([0-9x]+)x[a-z0-9_]+,', out_ty)
        shape = dims.group(1).split("x")
        shape_attr = ", ".join(f"{d} : i32" for d in shape)
        return (f'{res_name} = "ttnn.reshape"({arg_name}) '
                f'<{{shape = [{shape_attr}]}}> : ({in_ty}) -> {out_ty}')

    text, n_call = call_re.subn(repl, text)

    # 2. Remove the ttcore.cpu_module { ... } block via brace matching.
    start = text.find("ttcore.cpu_module")
    n_mod = 0
    if start != -1:
        brace = text.find("{", start)
        depth = 0
        i = brace
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        end = i + 1
        # also swallow trailing ' loc(#loc)' if present
        tail = re.match(r'\s*loc\(#loc\d*\)', text[end:])
        if tail:
            end += tail.end()
        text = text[:start] + text[end:]
        n_mod = 1

    # 3. Remove the forward_cpu_declaration private func line.
    decl_re = re.compile(
        r'^\s*func\.func private @cpu_hoisted_const_eval_\w+\(.*forward_cpu_declaration.*\n',
        re.MULTILINE,
    )
    text, n_decl = decl_re.subn("", text)

    sys.stderr.write(
        f"[strip_cpu_hoist] replaced {n_call} call(s), removed {n_mod} cpu_module, "
        f"{n_decl} declaration(s)\n"
    )
    return text


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp) as f:
        text = f.read()
    text = strip(text)
    with open(outp, "w") as f:
        f.write(text)
    print(outp)


if __name__ == "__main__":
    main()
