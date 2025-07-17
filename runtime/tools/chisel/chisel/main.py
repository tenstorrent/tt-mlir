# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import datetime
import os

from chisel.core.context import ChiselContext
from chisel.core.enums import ExecutionType
from ttmlir.ir import Operation
import sys


def main():
    """
    To run this script:
    You need to have ttir.mlir and ttnn.mlir files in the base directory.
    To generate ttir.mlir and ttnn.mlir please run chisel.sh on your starting ttir file
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <model>")
        sys.exit(1)

    model = sys.argv[1]
    base_dir = Path(
        f"/proj_sw/user_dev/sgligorijevic/jaxbringups/tt-mlir/chiselingv4/{model}"
    )
    main_fn = "main"
    program_index = 0
    tensor_folder = base_dir / "tensors"

    input_dir = base_dir
    output_dir = base_dir / "output"

    ttnn_path = base_dir / "ttnn.mlir"
    ttir_path = base_dir / "ttir.mlir"
    flatbuffer_path = base_dir / "fb.ttnn"

    chisel_context = ChiselContext(
        ttir_text=ttir_path.read_text(),
        ttnn_text=ttnn_path.read_text(),
        output_dir=output_dir,
        report_path=base_dir
        / f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        main_fn=main_fn,
        program_index=program_index,
        flatbuffer_path=flatbuffer_path,
        should_skip_op=lambda op: '"ttnn.gelu"' in op.get_asm(enable_debug_info=True),
    )
    chisel_context.bind_callbacks()
    input_paths = [
        f"{tensor_folder}/{i}.pt" for i in range(len(os.listdir(tensor_folder)))
    ]
    print(f"Input paths:")
    print(*input_paths, sep="\n")
    chisel_context.load_inputs_from_disk(input_paths)
    chisel_context.registry.load_all_ops()
    chisel_context.run()


if __name__ == "__main__":
    main()
