# Claude Code Permission Configuration for Chisel Development

When using Claude Code for Chisel development, you can configure it so that it
never blocks waiting for permission approval. Instead, any tool it doesn't have
pre-approved access to is auto-denied, and Claude is forced to find an
alternative approach.

## Setup

Use the `dontAsk` permission mode in your settings file. This can be set at
the user level (`~/.claude/settings.json`) or the project level
(`.claude/settings.local.json`).

### Minimal Configuration

```json
{
  "defaultMode": "dontAsk",
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Edit",
      "Write"
    ]
  }
}
```

With this setup:
- Tools in the `allow` list work without prompting.
- Everything else is **auto-denied** — no prompt appears, Claude gets told
  "denied" and must find another way.

### Recommended Configuration for Chisel

```json
{
  "defaultMode": "dontAsk",
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Edit",
      "Write",
      "Bash(cmake *)",
      "Bash(python *)",
      "Bash(pytest *)",
      "Bash(source env/activate *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(ls *)",
      "Bash(pre-commit *)"
    ]
  }
}
```

This allows Claude to build, test, and navigate the codebase but prevents it
from running arbitrary shell commands, pushing code, or performing other
potentially destructive operations.

## How It Works

- **`dontAsk`**: Auto-denies any tool call not in the allowlist. Claude receives
  a denial message and must adapt its approach without user intervention.
- **`permissions.allow`**: Allowlist of tools and command patterns. Supports
  glob-style wildcards (e.g., `Bash(git diff *)` matches any `git diff`
  invocation).
- **`permissions.deny`**: Optional denylist that takes precedence over allow
  rules. Useful for carving out exceptions (e.g., allow all git but deny
  `git push`).

## Other Permission Modes

| Mode                | Behavior                                              |
|---------------------|-------------------------------------------------------|
| `default`           | Prompts for permission on first use                   |
| `acceptEdits`       | Auto-accepts file edits, prompts for shell commands   |
| `auto`              | Auto-approves with background safety checks           |
| `bypassPermissions` | Skips all prompts (use in isolated environments only) |
| `dontAsk`           | Auto-denies unless pre-approved                       |
| `plan`              | Analyze-only mode, no modifications                   |

## Settings Precedence

Settings are evaluated in this order (highest priority first):

1. Managed settings (cannot be overridden)
2. CLI flags (`--dangerously-skip-permissions`)
3. Local project settings (`.claude/settings.local.json`)
4. Shared project settings (`.claude/settings.json`)
5. User settings (`~/.claude/settings.json`)
