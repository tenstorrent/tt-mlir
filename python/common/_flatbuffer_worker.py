# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Subprocess worker for running flatbuffers on device.

Executed as subprocess (not multiprocessing) to ensure clean hardware shutdown.
"""

import json
import sys


def main():
    """Run flatbuffer and write result to output file."""
    if len(sys.argv) != 3:
        print(
            "Usage: python -m python.common._flatbuffer_worker <flatbuffer_path> <result_path>"
        )
        sys.exit(1)

    flatbuffer_path = sys.argv[1]
    result_path = sys.argv[2]

    try:
        from ttrt.common.api import API

        API.initialize_apis()
        run_instance = API.Run(args={"binary": flatbuffer_path})
        return_code, _ = run_instance()

        if return_code != 0:
            with open(result_path, "w") as f:
                json.dump({"status": "error", "error": f"{return_code}"}, f)
        else:
            with open(result_path, "w") as f:
                json.dump({"status": "success", "return_code": return_code}, f)

    except Exception as e:
        with open(result_path, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)


if __name__ == "__main__":
    main()
