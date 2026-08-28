#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Report TT_FATAL changes between two tt-metal checkouts.

Extracts every TT_FATAL(...) call (full, multi-line, balanced-paren) from all
source files under a given path in two checkout directories and prints the
calls that were added, removed, or modified. Intended to run on tt-metal
uplift to flag validation changes in tt-train metal ops; each directory is
typically a sparse checkout of one commit.

Usage:
  tt_fatal_diff.py OLD_DIR NEW_DIR \
      [--path tt-train/sources/ttml/metal/ops] \
      [--old-sha SHA --new-sha SHA] [--slack-out FILE]

The text diff is printed to stdout. With --slack-out, a Slack mrkdwn message
with GitHub permalinks to the changed lines is also written to FILE, suitable
for posting via a webhook; this requires --old-sha and --new-sha to build the
links.

Exit code: 0 if no TT_FATAL changes, 2 if changes were found, 1 on error.
"""

import argparse
import difflib
import os
import re
import sys

SOURCE_EXTENSIONS = (".cpp", ".hpp", ".h", ".cc", ".cxx", ".hxx", ".c")
MACRO_RE = re.compile(r"\bTT_FATAL\s*\(")


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


def calls_by_file(root, path):
    """Map repo-relative file path -> TT_FATAL calls, for sources under root/path."""
    result = {}
    base = os.path.join(root, path)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in sorted(filenames):
            if not name.endswith(SOURCE_EXTENSIONS):
                continue
            full = os.path.join(dirpath, name)
            with open(full, encoding="utf-8", errors="replace") as f:
                calls = extract_calls(f.read())
            if calls:
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                result[rel] = calls
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


def render_text(changes, args, old_label, new_label):
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
        f"between {old_label} and {new_label}"
    )
    return "\n".join(lines)


def mrkdwn_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def top_level_args(call):
    """Split the body of a TT_FATAL call into its top-level arguments."""
    body = re.sub(r"^TT_FATAL\s*\(\s*", "", normalize(call))
    if body.endswith(")"):
        body = body[:-1].rstrip()
    parts = []
    cur = []
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == '"' or c == "'":
            quote = c
            cur.append(c)
            i += 1
            while i < n:
                cur.append(body[i])
                if body[i] == "\\":
                    i += 1
                    if i < n:
                        cur.append(body[i])
                elif body[i] == quote:
                    break
                i += 1
        elif c in "([{":
            depth += 1
            cur.append(c)
        elif c in ")]}":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append("".join(cur).strip())
    return parts


def excerpt(call, limit=90):
    """Short link label for Slack: the condition (plus message if trivial)."""
    parts = top_level_args(call)
    s = parts[0] if parts else normalize(call)
    if len(s) < 16 and len(parts) > 1:
        s = f"{s}, {parts[1]}"
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    # "|" would terminate the Slack link label.
    s = s.replace("|", "¦")
    return mrkdwn_escape(s)


def render_slack(changes, args):
    gh = args.github_url.rstrip("/")
    old_sha, new_sha = args.old_sha, args.new_sha
    header = (
        f":warning: *TT_FATAL changes in `{args.path}`* "
        f"(tt-metal uplift <{gh}/compare/{old_sha}...{new_sha}|"
        f"`{old_sha[:9]}` → `{new_sha[:9]}`>)"
    )
    lines = [header]
    for f, modified, removed, added in changes:
        entries = (
            [(line, "added", call, new_sha) for line, call in added]
            + [(nl, "modified", nc, new_sha) for _, (nl, nc) in modified]
            + [(line, "removed", call, old_sha) for line, call in removed]
        )
        entries.sort()
        file_sha = (
            old_sha if all(kind == "removed" for _, kind, _, _ in entries) else new_sha
        )
        name = f.rsplit("/", 1)[-1]
        lines.append(f"*<{gh}/blob/{file_sha}/{f}|{name}>*")
        for line, kind, call, sha in entries:
            lines.append(f"• {kind} <{gh}/blob/{sha}/{f}#L{line}|{excerpt(call)}>")

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
    parser.add_argument("old_dir", help="Checkout of the older tt-metal commit")
    parser.add_argument("new_dir", help="Checkout of the newer tt-metal commit")
    parser.add_argument(
        "--path",
        default="tt-train/sources/ttml/metal/ops",
        help="Repo-relative path to inspect (default: %(default)s)",
    )
    parser.add_argument("--old-sha", help="Commit SHA of old_dir, for Slack links")
    parser.add_argument("--new-sha", help="Commit SHA of new_dir, for Slack links")
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
        default=12000,
        help="Character budget for the Slack message (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.slack_out and not (args.old_sha and args.new_sha):
        print("error: --slack-out requires --old-sha and --new-sha", file=sys.stderr)
        return 1
    if not any(
        os.path.isdir(os.path.join(d, args.path)) for d in (args.old_dir, args.new_dir)
    ):
        print(f"error: {args.path} not found in either directory", file=sys.stderr)
        return 1

    old = calls_by_file(args.old_dir, args.path)
    new = calls_by_file(args.new_dir, args.path)
    changes = collect_changes(old, new)

    old_label = args.old_sha or args.old_dir
    new_label = args.new_sha or args.new_dir
    if not changes:
        print(
            f"No TT_FATAL changes in {args.path} "
            f"between {old_label} and {new_label}"
        )
        return 0

    print(render_text(changes, args, old_label, new_label))
    if args.slack_out:
        with open(args.slack_out, "w") as f:
            f.write(render_slack(changes, args) + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
