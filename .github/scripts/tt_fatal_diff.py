#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Report TT_FATAL changes between two tt-metal commits.

Extracts every TT_FATAL(...) call (full, multi-line, balanced-paren) from all
source files under a given path at two commits and prints the calls that were
added, removed, or modified. Intended to run on tt-metal uplift to flag
validation changes in tt-train metal ops.

Usage:
  tt_fatal_diff.py --repo /path/to/tt-metal OLD_COMMIT NEW_COMMIT \
      [--path tt-train/sources/ttml/metal/ops] [--slack-out FILE]

The text diff is printed to stdout. With --slack-out, a Slack mrkdwn message
with GitHub permalinks to the changed lines is also written to FILE, suitable
for posting via a webhook.

Exit code: 0 if no TT_FATAL changes, 2 if changes were found, 1 on error.
"""

import argparse
import difflib
import re
import subprocess
import sys

SOURCE_EXTENSIONS = (".cpp", ".hpp", ".h", ".cc", ".cxx", ".hxx", ".c")
MACRO_RE = re.compile(r"\bTT_FATAL\s*\(")


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def strip_comments(text):
    """Replace comment contents with spaces, preserving offsets and newlines."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            out.append(c)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == "\\":
                    i += 1
                    if i < n:
                        out.append(text[i])
                elif text[i] == quote:
                    i += 1
                    break
                i += 1
                continue
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append("  ")
                i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def extract_calls(text):
    """Return list of (line_number, call_text) for each TT_FATAL(...) call."""
    stripped = strip_comments(text)
    calls = []
    for match in MACRO_RE.finditer(stripped):
        start = match.start()
        i = match.end()  # just past the opening paren
        depth = 1
        n = len(stripped)
        while i < n and depth > 0:
            c = stripped[i]
            if c == '"' or c == "'":
                quote = c
                i += 1
                while i < n:
                    if stripped[i] == "\\":
                        i += 1
                    elif stripped[i] == quote:
                        break
                    i += 1
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        line = stripped.count("\n", 0, start) + 1
        calls.append((line, text[start:i]))
    return calls


def normalize(call):
    """Whitespace-insensitive form so pure reformatting is not a change."""
    return " ".join(call.split())


def list_files(repo, commit, path):
    out = git(repo, "ls-tree", "-r", "--name-only", commit, "--", path)
    return [f for f in out.splitlines() if f.endswith(SOURCE_EXTENSIONS)]


def calls_by_file(repo, commit, path):
    result = {}
    for f in list_files(repo, commit, path):
        text = git(repo, "show", f"{commit}:{f}")
        calls = extract_calls(text)
        if calls:
            result[f] = calls
    return result


def indent(text, prefix="    "):
    return "\n".join(prefix + line for line in text.splitlines())


def diff_file(old_calls, new_calls):
    """Pair up removed/added calls in order; return (modified, removed, added).

    modified is a list of (old, new) pairs matched by SequenceMatcher on the
    normalized call lists, so an edited condition or message shows up as one
    modification instead of an unrelated remove + add.
    """
    old_norm = [normalize(c) for _, c in old_calls]
    new_norm = [normalize(c) for _, c in new_calls]
    matcher = difflib.SequenceMatcher(a=old_norm, b=new_norm, autojunk=False)
    modified, removed, added = [], [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        olds = old_calls[i1:i2]
        news = new_calls[j1:j2]
        for old, new in zip(olds, news):
            modified.append((old, new))
        removed.extend(olds[len(news) :])
        added.extend(news[len(olds) :])
    return modified, removed, added


def collect_changes(old, new):
    """Return [(file, modified, removed, added)] for files with changes."""
    changes = []
    for f in sorted(set(old) | set(new)):
        modified, removed, added = diff_file(old.get(f, []), new.get(f, []))
        if modified or removed or added:
            changes.append((f, modified, removed, added))
    return changes


def render_text(changes, args):
    lines = []
    for f, modified, removed, added in changes:
        lines.append(f"\n== {f}")
        for line, call in added:
            lines.append(f"  ADDED (line {line}):")
            lines.append(indent(call))
        for line, call in removed:
            lines.append(f"  REMOVED (was line {line}):")
            lines.append(indent(call))
        for (old_line, old_call), (new_line, new_call) in modified:
            lines.append(f"  MODIFIED (line {new_line}):")
            lines.append(indent(old_call, "    - "))
            lines.append(indent(new_call, "    + "))
    lines.append(
        f"\nTT_FATAL changes found in {args.path} "
        f"between {args.old_commit} and {args.new_commit}"
    )
    return "\n".join(lines)


def mrkdwn_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def excerpt(call, limit=140):
    """One-line, escaped, truncated body of a TT_FATAL call for Slack."""
    s = re.sub(r"^TT_FATAL\s*\(\s*", "", normalize(call))
    if s.endswith(")"):
        s = s[:-1].rstrip()
    s = s.replace("`", "'")
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return mrkdwn_escape(s)


def render_slack(changes, args, old_sha, new_sha):
    gh = args.github_url.rstrip("/")
    header = (
        f":warning: *TT_FATAL changes in `{args.path}`* "
        f"(tt-metal uplift <{gh}/compare/{old_sha}...{new_sha}|"
        f"`{old_sha[:9]}` → `{new_sha[:9]}`>)"
    )
    lines = [header]
    for f, modified, removed, added in changes:
        lines.append(f"\n*{f}*")
        for line, call in added:
            url = f"{gh}/blob/{new_sha}/{f}#L{line}"
            lines.append(f"• Added <{url}|L{line}>: `{excerpt(call)}`")
        for (_, _), (new_line, new_call) in modified:
            url = f"{gh}/blob/{new_sha}/{f}#L{new_line}"
            lines.append(f"• Modified <{url}|L{new_line}>: `{excerpt(new_call)}`")
        for line, call in removed:
            url = f"{gh}/blob/{old_sha}/{f}#L{line}"
            lines.append(f"• Removed <{url}|was L{line}>: `{excerpt(call)}`")

    out = []
    used = 0
    truncated = False
    for line in lines:
        if used + len(line) + 1 > args.max_chars:
            truncated = True
            break
        out.append(line)
        used += len(line) + 1
    if truncated:
        out.append("_…output truncated; see the workflow run for the full diff._")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old_commit", help="Older tt-metal commit (base)")
    parser.add_argument("new_commit", help="Newer tt-metal commit")
    parser.add_argument("--repo", required=True, help="Path to a tt-metal checkout")
    parser.add_argument(
        "--path",
        default="tt-train/sources/ttml/metal/ops",
        help="Repo path to inspect (default: %(default)s)",
    )
    parser.add_argument(
        "--slack-out",
        metavar="FILE",
        help="Also write a Slack mrkdwn message with permalinks to FILE",
    )
    parser.add_argument(
        "--github-url",
        default="https://github.com/tenstorrent/tt-metal",
        help="Base repo URL for Slack permalinks (default: %(default)s)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3500,
        help="Character budget for the Slack message (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        old = calls_by_file(args.repo, args.old_commit, args.path)
        new = calls_by_file(args.repo, args.new_commit, args.path)
        changes = collect_changes(old, new)
        if changes and args.slack_out:
            old_sha = git(args.repo, "rev-parse", args.old_commit).strip()
            new_sha = git(args.repo, "rev-parse", args.new_commit).strip()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not changes:
        print(
            f"No TT_FATAL changes in {args.path} "
            f"between {args.old_commit} and {args.new_commit}"
        )
        return 0

    print(render_text(changes, args))
    if args.slack_out:
        with open(args.slack_out, "w") as f:
            f.write(render_slack(changes, args, old_sha, new_sha) + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
