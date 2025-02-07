
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os




def main():
    
    print(os.environ.get('changed_files_json'))
    print(os.environ.get('pull_request'))

main()