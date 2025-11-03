#!/usr/bin/env python3
'''
Usage - run ./check_base_commits.py

It will clone / pull all relevant repos and display a table of the tt-mlir commits used by each repo.

Take notice of the tt-mlir commit/tt-mlir commit details column. This is the stable base commit used by that repo in CI.

Example output:
repo name                      | tt-mlir commit                           | ttmlir commit details
----------------------------------------------------------------------------------------------------------------------
tt-xla                         | 8f318886f6cb299a753f5dff1bbf0e5083c4dfd5 | 8f318886f | Tapasvi Patel | 2025-09-06 17:01:42 -0500 | Remove shardy roundtrip patch due to jax0.7 uplift (#4816)
tt-forge-fe                    | 17e1c32cd023ab1541ae56a8cdf9c6cb715e130e | 17e1c32cd | Saber Gholami | 2025-09-02 17:25:39 -0400 | [Optimizer] Add constraint API for memory management ops (#4734)
'''

import os
import subprocess
import re
from pathlib import Path

REPOS = {
    # "tt-torch": "https://github.com/tenstorrent/tt-torch.git",
    "tt-xla": "https://github.com/tenstorrent/tt-xla.git",
    "tt-forge-fe": "https://github.com/tenstorrent/tt-forge-fe.git",
    "tt-mlir": "https://github.com/tenstorrent/tt-mlir.git",
}

def clone_or_pull(repo_name, url):
    if repo_name == "tt-mlir" or repo_name == "tt-xla":
        # Always clone or pull with full history for tt-mlir and xla
        if not Path(repo_name).exists():
            print(f"Cloning {repo_name} (full)...")
            subprocess.run(["git", "clone", url, repo_name], check=True)
        else:
            print(f"Pulling {repo_name} (full)...")
            subprocess.run(["git", "-C", repo_name, "fetch"], check=True)
    else:
        # Shallow clone for other repos
        if not Path(repo_name).exists():
            print(f"Cloning {repo_name} (shallow)...")
            subprocess.run(["git", "clone", "--depth", "1", url, repo_name], check=True)
        else:
            print(f"Pulling {repo_name}...")
            subprocess.run(["git", "-C", repo_name, "fetch"], check=True)

def get_mlir_commit_from_cmakelists(repo_dir):
    cmake_path = Path(repo_dir) / "third_party" / "CMakeLists.txt"
    if not cmake_path.exists():
        return None
    with open(cmake_path) as f:
        for line in f:
            m = re.search(r'set\(TT_MLIR_VERSION\s+"([0-9a-f]+)"\)', line)
            if m:
                return m.group(1)
    return None

def get_mlir_commit_from_submodule(repo_dir):
    submodule_path = Path(repo_dir) / "third_party" / "tt-mlir"
    if not submodule_path.exists():
        return None
    # Get the commit hash the submodule is on
    result = subprocess.run(
        ["git", "-C", str(submodule_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()

def get_commit_details(mlir_dir, commit):
    result = subprocess.run(
        ["git", "-C", mlir_dir, "log", "-1", "--format=%h | %an | %ad | %s", "--date=iso", commit],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()

def main():
    # Clone or pull all repos
    for name, url in REPOS.items():
        clone_or_pull(name, url)

    # Always pull latest tt-mlir
    subprocess.run(["git", "-C", "tt-mlir", "fetch", "--all"], check=True)

    report = []

    # Handle other repos
    for repo in ["tt-xla", "tt-forge-fe"]:
        if repo == "tt-xla":
            clone_or_pull('tt-xla', REPOS['tt-xla'])
            commit = get_mlir_commit_from_cmakelists(repo)
        elif repo == "tt-forge-fe":
            # Make sure submodules are initialized
            subprocess.run(["git", "-C", repo, "submodule", "update", "--init", "--depth", "1"], check=True)
            commit = get_mlir_commit_from_submodule(repo)
        else:
            commit = None

        if commit:
            # Check if commit exists locally before fetching
            try:
                subprocess.run(["git", "-C", "tt-mlir", "cat-file", "-e", commit], check=True)
            except subprocess.CalledProcessError:
                # Commit does not exist locally, try to fetch from remote
                try:
                    subprocess.run(["git", "-C", "tt-mlir", "fetch", "origin", commit], check=True)
                except subprocess.CalledProcessError:
                    print(f"Warning: Could not fetch commit {commit} from remote. It may not exist on origin.")
            details = get_commit_details("tt-mlir", commit)
        else:
            details = "N/A"
        report.append((repo, commit or "N/A", details))

    # Print table
    print('\n')
    print(f"{'repo name':<30} | {'tt-mlir commit':<40} | ttmlir commit details")
    print("-" * 120)
    for row in report:
        print(f"{row[0]:<30} | {row[1]:<40} | {row[2]}")

if __name__ == "__main__":
    main()