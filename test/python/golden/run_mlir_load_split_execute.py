# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load a TTIR MLIR file, split into per-op modules, compile and execute with intermediate verification.

Uses mesh 2x4 to match forge Llama 70B MLIR (ttcore.meshes<[<"mesh" = 2x4>]>).

Usage (from repo root after source env/activate):

  python test/python/golden/run_mlir_load_split_execute.py forge_llama_3_1_70b_5_layer.mlir
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Mesh 2x4 as in forge_llama_3_1_70b_5_layer.mlir (ttcore.meshes<[<"mesh" = 2x4>]>)
MESH_SHAPE = (2, 4)


def main():
    ap = argparse.ArgumentParser(
        description="Load TTIR MLIR, split, compile and execute with intermediate verification."
    )
    ap.add_argument("mlir_file", help="Path to .mlir file (e.g. forge_llama_3_1_70b_5_layer.mlir)")
    ap.add_argument("--artifact-dir", default="./builder-artifacts/load_split_execute")
    args = ap.parse_args()

    path = Path(args.mlir_file)
    if not path.exists():
        print("Error: file not found:", path, file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        mlir_text = f.read()

    from builder.base.builder_apis import load_mlir_file, split_mlir_file, compile_ttir_module_to_flatbuffer
    from builder.base.builder_runtime import execute_fb
    from collections import OrderedDict

    print("Loading", path, "target=ttir")
    try:
        module, builder = load_mlir_file(mlir_text, target="ttir")
    except Exception as e:
        print("Load failed:", e, file=sys.stderr)
        sys.exit(1)

    modules_and_builders = split_mlir_file(module, builder, target="ttir")
    print("Split into", len(modules_and_builders), "modules.")

    import _ttmlir_runtime as tt_runtime
    mesh_dict = OrderedDict([("x", MESH_SHAPE[0]), ("y", MESH_SHAPE[1])])
    artifact_base = Path(args.artifact_dir)
    artifact_base.mkdir(parents=True, exist_ok=True)

    tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
    opts = tt_runtime.runtime.MeshDeviceOptions()
    opts.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
    opts.mesh_shape = MESH_SHAPE
    device = tt_runtime.runtime.open_mesh_device(opts)

    for idx, (mod, bld) in enumerate(modules_and_builders):
        prog_dir = artifact_base / f"program_{idx}"
        prog_dir.mkdir(parents=True, exist_ok=True)
        print("Compiling program", idx)
        try:
            compiled_bin, io_goldens, inter_goldens = compile_ttir_module_to_flatbuffer(
                mod, bld, target="ttnn", mesh_dict=mesh_dict,
                save_artifacts=True, artifact_dir=str(prog_dir),
            )
        except Exception as e:
            print("Compile failed:", e, file=sys.stderr)
            continue
        print("Executing with intermediate verification...")
        try:
            golden_report, _ = execute_fb(
                compiled_bin, io_goldens, inter_goldens, device=device,
                enable_intermediate_verification=True,
                save_artifacts=True, artifact_dir=str(prog_dir),
            )
            report_path = prog_dir / "golden_report.json"
            with open(report_path, "w") as rf:
                json.dump(golden_report, rf, indent=2)
            print("Report:", report_path)
            for loc, results in golden_report.items():
                for _, data in results.items():
                    if isinstance(data, dict):
                        r, pcc = data.get("result", ""), data.get("actual_pcc", 1.0)
                        if r != "pass" or (isinstance(pcc, (int, float)) and pcc < 0.99):
                            print("  Low PCC @", loc, "pcc=", pcc)
        except Exception as e:
            print("Execute failed:", e, file=sys.stderr)

    tt_runtime.runtime.close_mesh_device(device)
    print("Artifacts:", artifact_base.resolve())


if __name__ == "__main__":
    main()
