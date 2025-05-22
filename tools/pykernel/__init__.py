# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

## DEFINE THE PYKERNEL VERSION DYNAMICALLY ##

import datetime

# Get today's date as calver
today = datetime.datetime.now()
calver = today.strftime("%y.%m.%d")

# Set version to 0.1.{calver}
__version__ = f"0.1.{calver}"
