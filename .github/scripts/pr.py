
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import re
from typing import Dict

no_build_re = re.compile('^')

def to_github_output():
    pass

def pr(changed_files: Dict, pull_request: dict ):
    run_test_and_build = True
    all_changed_and_modified_files = changed_files.get('all_changed_and_modified_files').split(' ')
    print(all_changed_and_modified_files)


def main():
    print(os.environ['output'])
    #changed_files = json.loads(os.environ.get('changed_files'))
    #pull_request = json.loads(os.environ.get('pull_request'))
    #pr(changed_files, pull_request)

main()