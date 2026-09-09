# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Subprocess worker for generating the system descriptor.

Executed as subprocess (not multiprocessing) so that the device is released
cleanly when the process exits, freeing the hardware for subsequent device
opens in other subprocesses.
"""

import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m python.common._system_desc_worker <output_path>")
        sys.exit(1)

    output_path = sys.argv[1]

    import _ttmlir_runtime as tt_runtime

    system_desc = tt_runtime.runtime.get_current_system_desc()
    system_desc.store(output_path)


if __name__ == "__main__":
    main()
