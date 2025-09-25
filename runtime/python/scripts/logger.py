# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
def log_message(message):
    """Log a message from C++ runtime."""
    print(message)


def log_operation(operation_info):
    """Log operation execution information."""
    print(f"Operation executed: {operation_info}")
