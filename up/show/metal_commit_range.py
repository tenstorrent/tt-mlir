#!/usr/bin/env python3
import sys
import requests
import csv

GITHUB_API = "https://api.github.com/repos/tenstorrent/tt-metal/compare"
GITHUB_DIFF = "https://github.com/tenstorrent/tt-metal/compare"

def fetch_commits(from_commit, to_commit):
    url = f"{GITHUB_API}/{from_commit}...{to_commit}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Failed to fetch commits: {resp.status_code} {resp.text}")
        sys.exit(1)
    data = resp.json()
    commits = []
    for c in data.get("commits", []):
        date = c["commit"]["author"]["date"][:10]
        sha = c["sha"][:7]
        author = c["commit"]["author"]["name"]
        # only get the commit title
        message = c["commit"]["message"].splitlines()[0].replace('"', "'").replace('\n', ' ')
        commits.append({
            "date": date,
            "hash": sha,
            "author": author,
            "message": message,
            "longsha": c["sha"]
        })
    return commits

def fetch_and_save_diff(from_commit, to_commit, out_file):
    url = f"{GITHUB_DIFF}/{from_commit}...{to_commit}.diff"
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(out_file, "w") as f:
            f.write(resp.text)
        print(f"Diff written to {out_file}")
    else:
        print(f"Failed to fetch diff: {resp.status_code}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python metal_commit_range.py <from_commit> <to_commit>")
        sys.exit(1)
    from_commit, to_commit = sys.argv[1:3]

    commits = fetch_commits(from_commit, to_commit)
    print("Date,Commit,Author,Message,Link")
    for commit in commits:
        print(f"{commit['date']},{commit['hash']},{commit['author']},\"{commit['message']}\",https://github.com/tenstorrent/tt-metal/commit/{commit['longsha']}")

    out_file = f"diff_from_{from_commit}_to_{to_commit}.diff"
    fetch_and_save_diff(from_commit, to_commit, out_file)

if __name__ == "__main__":
    main()
    
    
    
# Usage:
# ./metal_commit_range.py <from_commit> <to_commit>
# copy paste the output (CSV) to google sheets
# Data > Split text to columns
#   Can attach hyperlink vi eqn =HYPERLINK(<cell>, "link") because google sheets doesn't autopopulate the links