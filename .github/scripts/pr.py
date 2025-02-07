
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import re
from pathlib import Path

os.environ["pull_request"] = '{"draft": "false"}'
os.environ["changed_files"] =  '{ "all_changed_and_modified_files": ".github/scripts/pr.py docs/src/build.md" }' 

#os.environ["pull_request"] = '{"draft": "true"}'
#os.environ["changed_files"] =  '{ "all_changed_and_modified_files": ".github/scripts/pr.py docs/src/build.md" }' 
#
#os.environ["pull_request"] = '{"draft": "false"}'
#os.environ["changed_files"] =  '{ "all_changed_and_modified_files": "docs/src/build.md" }' 

pull_request: dict = json.loads(os.environ.get('pull_request', '{}'))
changed_files: dict = json.loads(os.environ.get('changed_files', '{}'))

def to_github_output(outputs: dict):
    github_output = os.environ['GITHUB_OUTPUT']
    payload =json.dumps(outputs)
    print_builder = ''
    print_builder += 'Here are your available outputs'
    for x in outputs:
        pass
    # TODO: add debug flag
    with open(github_output, "a") as myfile:
        myfile.write(f"results={payload}")
        
    

def files_regex(action: str, manifest: dict) -> dict:
    if not manifest:
        return False
    # Avaible actions
    # https://github.com/tj-actions/changed-files?tab=readme-ov-file#outputs-
    changed_files_list: list = changed_files.get(action).split(' ')
    changed_files_regex = re.compile('|'.join(manifest))
    return  all([changed_files_regex.match(x) for x in changed_files_list])
    
def draft_pr(manifest: str):
    is_draft = pull_request.get("draft")
    if manifest == 'true' and is_draft == 'true':
        return True
    return False
        
def skip(manifest: dict) -> bool:
    skip_manifest = manifest.get('skip')
    if not skip_manifest:
        return False
    if draft_pr(skip_manifest.get('draft_pr')):
        return True
    if files_regex('all_changed_and_modified_files', skip_manifest.get('files_regex')):
        return True
    return False

def main():
    json_config: Path = Path.joinpath(Path(__file__).resolve().parent, 'pr.json')
    manifests: dict = json.loads(json_config.read_text())

    outputs = {}

    for manifest in manifests:
        job_name = manifest['job_name']
        outputs[job_name] = {} 
        outputs[job_name]['skip']= skip(manifest)
    to_github_output(outputs)

main()