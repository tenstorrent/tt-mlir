---
name: validate-tt-mlir-against-tt-xla
description: >
  Validate a tt-mlir PR against tt-xla by creating a cherry-picked branch and triggering CI.
  Invoked as: /validate-tt-mlir-against-tt-xla <PR number or URL>.
  Use this skill whenever the user wants to test, validate, qualify, or check a tt-mlir PR
  in tt-xla, or mentions running uplift qualification test suite, or asks to trigger tt-xla CI
  for a tt-mlir change. Also triggers when the user mentions "xla validate", "xla test",
  or "validate in xla".
---

# Validate tt-mlir PR in tt-xla

This skill automates cross-repo validation of tt-mlir PRs against tt-xla CI. The core
idea: tt-xla pins a specific tt-mlir commit. To test a PR before it merges, you create
a temporary branch that layers the PR's commits on top of that pinned commit, then trigger
tt-xla's test workflows against it. The branch is ephemeral — it is deleted once CI
completes, and always recreated fresh on each run.

## Repos

- **tt-mlir**: `tenstorrent/tt-mlir` (the PR lives here)
- **tt-xla**: `tenstorrent/tt-xla` (CI runs here)

## Workflow

### 1. Identify the PR

The PR number or URL can be provided as an argument (e.g. `/validate-tt-mlir-against-tt-xla 1234`).
If not provided, detect the PR for the current branch:

```bash
gh pr view $(git branch --show-current) --repo tenstorrent/tt-mlir --json number,headRefName,baseRefName,body,commits,state
```

If no PR is found for the current branch, ask once: `PR number or URL:`

Extract the PR metadata:

```bash
gh pr view <PR> --repo tenstorrent/tt-mlir --json number,headRefName,baseRefName,body,commits,state
```

If the PR is merged or closed, stop — nothing to do.

### 2. Resolve tt-xla branch

Determine `XLA_BRANCH` before anything else — both the uplifted commit and test suite
discovery depend on it.

Check if there is a related branch in tt-xla that the user might want to test against.
Extract the PR author's GitHub username from the PR metadata, then search for their
branches in tt-xla. The tt-xla repo has many branches (100+), so use `--paginate`:

```bash
gh api "repos/tenstorrent/tt-xla/branches" --paginate \
  -q '.[].name' | grep -i "<author-username>"
```

tt-xla branches follow the convention `username/branch-name` (e.g. `acicovic/rms-test`).
If any branches match the same author, evaluate whether each candidate branch is
**semantically related** to the tt-mlir PR branch — not just keyword overlap. Strip the
`username/` prefix from both branches and compare the remaining names. Consider them
related only if they clearly refer to the same feature, fix, or work item (e.g.
`add-permute-reshape-canon` and `permute-reshape-test` are related; `fix-reshape` and
`reshape-perf-dashboard` are not). When in doubt, treat the branch as unrelated.

If exactly one semantically related branch is found, present a choice via `AskUserQuestion`:

- question: "Found a related tt-xla branch. Which branch to run CI against?"
- `"main (Recommended)"`
- `"<matched-branch-name>"` — with description noting the author match

If multiple related branches are found, list them all as options alongside `main`.
If no related branch is found, silently default to `main`. The user can also override
by specifying a branch as part of their invocation or in conversation. Call the result
`XLA_BRANCH`.

### 3. Resolve the uplifted tt-mlir commit

Read the version pin from `XLA_BRANCH`:

```bash
gh api "repos/tenstorrent/tt-xla/contents/third_party/CMakeLists.txt?ref=<XLA_BRANCH>" \
  -q .content | base64 -d \
  | grep -oP 'set\(TT_MLIR_VERSION "\K[^"]+'
```

This returns the SHA that `XLA_BRANCH` currently builds against. Call it `UPLIFTED_SHA`.

### 4. Gather remaining user inputs

#### Discover available test suites

Fetch the valid `test_suite` options directly from the workflow file — this is the
authoritative source for what values `manual-test.yml` will actually accept:

```bash
gh api "repos/tenstorrent/tt-xla/contents/.github/workflows/manual-test.yml?ref=<XLA_BRANCH>" \
  -q .content | base64 -d \
  | grep -A100 'test_suite:' | grep -oP "'\K[^']+"
```

Do not truncate this output — capture all options before selecting the default.

If the workflow uses `options:` under `test_suite`, parse those. If it uses a free-text
`input` with no enumerated options, fall back to listing JSON files in
`test-matrix-presets/`:

```bash
gh api "repos/tenstorrent/tt-xla/git/trees/<XLA_BRANCH>?recursive=1" \
  -q '.tree[] | select(.path | test("test-matrix-presets/.*\\.json$")) | .path | split("/") | last | rtrimstr(".json")'
```

From the discovered suite names, identify the **default suite** by finding the one most
likely intended for uplift/qualification testing. Look for names containing `uplift`,
`qualification`, or `mlir` — in that priority order. If multiple match, prefer the most
specific. If none match, present all suites to the user and ask them to designate a
default. Never hardcode a suite name — always select from what was actually discovered.

The default suite always runs. From the remaining suites, pick the 3 most relevant to
surface as additional options. If discovery fails entirely, fall back to a text prompt.

#### Ask remaining questions via AskUserQuestion

Use the `AskUserQuestion` tool with both questions batched together:

**Question 1** — header: `Extra suites`, multiSelect: true
- question: "`<default-suite>` will always run (discovered dynamically). Select additional suites to run in parallel, if any."
- Options are the 3 most relevant non-default suites discovered above. Do NOT include
  the default suite as an option — it always runs.
- The auto-added "Type something" option lets the user type any suite name not shown.
- Leave all unselected to run the default suite only.

**Question 2** — header: `Baseline`
- question: "Baseline failure detection? Type a run ID to compare against a previous run."
- "Skip — just report failures (Recommended)"
- "Run baseline in parallel"
- The auto-added "Type something" option lets the user paste a run ID as baseline.

Parse all answers before proceeding. The default suite always runs; selected extra
suites run in parallel alongside it.

### 5. Re-run detection and stale uplift warning

Check if the `xla-validate/<pr-number>` branch already exists on the remote:

```bash
git ls-remote --exit-code origin xla-validate/<pr-number>
```

If the branch exists, CI from a previous run is likely still in progress. Warn the user:

> **Note:** `xla-validate/<pr-number>` already exists — a previous CI run may still be
> in progress. Re-triggering will delete and recreate the branch. Continue?

If they confirm, proceed. If not, stop.

If the branch does not exist but a previous validation block is present in the PR
description, this is a re-run after the previous CI completed. Extract the previous
`UPLIFTED_SHA` from the `**Base (uplifted):**` line in the description. If it differs
from the current `UPLIFTED_SHA`, inform the user:

> **Note:** tt-xla has uplifted to a newer tt-mlir commit since your last validation
> (`<PREV_SHA_SHORT>` → `<UPLIFTED_SHA_SHORT>`). The cherry-pick base has changed.

### 6. Create the branch

The branch name is always `xla-validate/<pr-number>`.

If the branch already exists (user confirmed re-trigger above), delete it first:

```bash
git push origin --delete xla-validate/<pr-number>
```

Get the list of commits from the PR:

```bash
gh api repos/tenstorrent/tt-mlir/pulls/<pr-number>/commits --paginate -q '.[].sha'
```

Fetch the PR's head ref to ensure all commit objects are present locally before
cherry-picking (the commits may not exist in the local clone if the PR branch was
never checked out):

```bash
git fetch origin refs/pull/<pr-number>/head
```

Create the branch fresh by cherry-picking those commits onto `UPLIFTED_SHA`:

```bash
git fetch origin <UPLIFTED_SHA>
git checkout -B xla-validate/<pr-number> <UPLIFTED_SHA>
git cherry-pick <commit1> <commit2> ... <commitN>
git push origin xla-validate/<pr-number>
```

If cherry-pick has conflicts, stop and tell the user — they need to resolve manually.

Get the HEAD SHA of the pushed branch — this is the `MLIR_OVERRIDE_SHA` for CI.

### 7. Trigger tt-xla CI

Dispatch the "Run test" workflow for each selected test suite:

```bash
gh workflow run manual-test.yml \
  --repo tenstorrent/tt-xla \
  --ref <XLA_BRANCH> \
  -f test_suite=<suite> \
  -f mlir_override=<MLIR_OVERRIDE_SHA>
```

If the user chose baseline option 2, also dispatch baseline runs for each suite **without**
`mlir_override`:

```bash
gh workflow run manual-test.yml \
  --repo tenstorrent/tt-xla \
  --ref <XLA_BRANCH> \
  -f test_suite=<suite>
```

After dispatching, wait ~5 seconds then find the newly created runs. Record the
timestamp just before dispatching and use it to filter:

```bash
# capture before dispatch
DISPATCH_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ... dispatch commands ...

# wait for GitHub to register the run
sleep 5

gh run list --repo tenstorrent/tt-xla --workflow=manual-test.yml --limit 20 \
  --json databaseId,status,name,createdAt,url \
  --jq "[.[] | select(.createdAt >= \"$DISPATCH_TIME\")]"
```

Match each dispatched suite to exactly one run created at or after `DISPATCH_TIME`.
If multiple candidates appear for the same suite, take the most recent. If no run
appears within 15 seconds, retry once — GitHub Actions dispatch is occasionally slow.

**Important:** Re-triggering failed jobs within an existing run does NOT create a new
run ID. The run ID is stable for the lifetime of the workflow run. Only dispatch
creates new IDs. Once IDs are recorded here, treat them as final — never update them
during polling.

Collect run URLs and IDs. These are the **only** IDs tracked for the rest of the skill.

### 8. Update the PR description (initial — in_progress)

Immediately append (or replace) a validation block at the end of the PR description.
Use HTML comment markers so the skill can find and replace it on re-runs.

**IMPORTANT: Always use the tempfile approach for editing PR descriptions.** Writing
markdown with backticks, pipes, and special characters into shell variables causes
escaping issues that can wipe the PR body. See the "Editing PR descriptions safely"
section below for the exact procedure.

Format with no baseline (option 1):

```markdown
<!-- xla-validate -->
---
### tt-xla Validation

**Base (uplifted):** `<UPLIFTED_SHA_SHORT>`
**tt-xla branch:** `<XLA_BRANCH>` ← omit this line when XLA_BRANCH is `main`

| Test Suite | Run | Status |
|------------|-----|--------|
| <default-suite> | [Run #<id>](<url>) | :hourglass: in_progress |
<!-- /xla-validate -->
```

Use the actual suite name discovered dynamically — never write a hardcoded name here.

Format with parallel baseline (option 2) — add a **Baseline** row beneath each PR row,
marked with a `(baseline)` label:

```markdown
| Test Suite | Run | Status |
|------------|-----|--------|
| <default-suite> | [Run #<id>](<url>) | :hourglass: in_progress |
| <default-suite> (baseline) | [Run #<id>](<url>) | :hourglass: in_progress |
```

Format with provided baseline (option 3) — same as option 2 but the baseline row links
to the user-supplied run and starts as either in_progress or already-resolved depending
on that run's current state.

On re-runs, replace everything between `<!-- xla-validate -->` and `<!-- /xla-validate -->`
with the fresh block.

### 9. Report to user and start polling

Tell the user:
- CI runs triggered with links
- PR description updated with in_progress status
- That you will monitor the runs and update the PR when they finish

Use the `/loop` skill (main agent) to poll every 3 minutes so the user can continue
other work.

### 10. Poll for run completion

Each iteration, check the exact run IDs recorded in step 7 — never scan for new runs
or add IDs during polling. Re-triggered jobs stay under the same run ID and will be
reflected in the existing run's status automatically.

```bash
gh run view <run-id> --repo tenstorrent/tt-xla --json status,conclusion,url
```

A run with re-triggered jobs may cycle through `queued` or `in_progress` again after
previously being `completed` — this is normal. Keep polling until the run settles in a
terminal state and stays there for one full poll cycle.

Report status in the terminal each iteration (e.g. `[12:34] <default-suite>:
in_progress, next check in 3min`). Keep polling until all tracked runs (PR runs + any
baseline runs) reach a terminal state (`completed`, `failure`, `cancelled`, `timed_out`).

### 11. Finalize

Once all runs are done:

1. **Delete the branch** — it is no longer needed:
   ```bash
   git push origin --delete xla-validate/<pr-number>
   ```

2. **Update the PR description** with final statuses. Map conclusions to status indicators:
   - `success` → `:white_check_mark: passed`
   - `failure` → `:x: failed`
   - `cancelled` → `:no_entry_sign: cancelled`
   - `timed_out` → `:alarm_clock: timed_out`

   **If any PR runs failed**, run the failure analysis (step 12) first so the failure
   table is included in the same edit.

3. **Notify the user** with a summary of results.

### 12. Analyze failures and report

When a PR run fails, analyze the CI logs to identify which tests failed and why.
Then add a compact failure table to the validation block.

#### Finding failed tests

1. Get the failed jobs:
```bash
gh run view <run-id> --repo tenstorrent/tt-xla --json jobs \
  --jq '.jobs[] | select(.conclusion == "failure") | {name, databaseId}'
```

2. For each failed job, extract test names and errors:
```bash
gh api repos/tenstorrent/tt-xla/actions/jobs/<job-id>/logs 2>&1 \
  | grep -E "FAILED|PASSED" | head -30
```

3. Get error details for each failed test:
```bash
gh api repos/tenstorrent/tt-xla/actions/jobs/<job-id>/logs 2>&1 \
  | grep -B5 -A15 "_____ test_all_models_jax\[<test-name>" | head -40
```

#### Comparing against baseline (options 2 and 3)

If the user requested a baseline, compare the failed tests against the baseline run's
results to classify each failure:

- If the same test **failed in the baseline too** → `(pre-existing)`
- If the test **passed in the baseline** → `(regression)`
- If the baseline is still in_progress, note that classification is pending

If no baseline was requested (option 1), omit the classification column entirely.

#### Failure table format

Add this directly below the status table in the validation block:

Without baseline:
```markdown
#### Failures

| Test | Arch | Error |
|------|------|-------|
| `gpt2/causal_lm/jax-Base-inference` | n150, p150 | `XlaRuntimeError: Error code 13` |
```

With baseline:
```markdown
#### Failures

| Test | Arch | Error | Classification |
|------|------|-------|----------------|
| `gpt2/causal_lm/jax-Base-inference` | n150, p150 | `XlaRuntimeError: Error code 13` | regression |
| `t5/summarization/jax-Base-inference` | n150, p150 | `XlaRuntimeError: Error code 13` | pre-existing |
```

Rules for the failure table:
- Shorten test names: drop `single_device` and `test_all_models_jax[]` wrapper
- Combine architectures if the same test fails on multiple (e.g. "n150, p150")
- Error column: just the exception class and code/message
- No graph URLs unless CI actually uploaded graph artifacts (check artifacts list first)

## Editing PR descriptions safely

Shell variables containing markdown (backticks, pipes, dollar signs) break when passed
through `--body`. **Always use the tempfile + `--body-file` approach:**

### Writing the body

1. Use the Write tool (or `cat <<'HEREDOC' > /tmp/pr_body_<pr-number>.md`) to write
   the full PR body to a temp file. Using a **quoted heredoc** (`<<'EOF'`) prevents
   any shell expansion of the content.

2. Apply the edit:
```bash
gh pr edit <PR> --repo tenstorrent/tt-mlir --body-file /tmp/pr_body_<pr-number>.md
```

3. Clean up:
```bash
rm /tmp/pr_body_<pr-number>.md
```

### Verification (required after every edit)

After every `gh pr edit`, immediately verify the body was applied correctly:

```bash
BODY=$(gh pr view <PR> --repo tenstorrent/tt-mlir --json body -q .body)
if [ -z "$BODY" ]; then
  echo "ERROR: PR body is empty after edit!"
elif echo "$BODY" | grep -q "xla-validate"; then
  echo "OK: PR body looks correct"
else
  echo "WARNING: PR body may be corrupted — missing validation block"
fi
```

If verification fails:
1. Re-read the current PR body to assess damage
2. Reconstruct the full body (original content + validation block) using the Write tool
3. Retry with `--body-file`
4. Verify again

### Constructing updated bodies for re-runs

When replacing an existing validation block, use Python for reliable text manipulation
instead of shell tools like awk/sed/perl which struggle with markdown:

```bash
python3 -c "
import sys
body = open('/tmp/pr_body_old_<pr-number>.md').read()
new_block = open('/tmp/pr_validation_block_<pr-number>.md').read()
start = body.find('<!-- xla-validate -->')
end = body.find('<!-- /xla-validate -->')
if start >= 0 and end >= 0:
    end = end + len('<!-- /xla-validate -->')
    body = body[:start] + new_block + body[end:]
else:
    body = body + '\n\n' + new_block
open('/tmp/pr_body_<pr-number>.md', 'w').write(body)
"
```

## Error handling

- **Cherry-pick conflicts**: Stop immediately, show the conflicting commit, and tell
  the user to resolve manually. Do not force through conflicts.
- **Branch push fails**: Report clearly and stop.
- **Workflow dispatch fails**: Delete the `xla-validate/<pr-number>` branch (it is no
  longer needed if CI cannot run), then check `gh` auth and repo permissions. Report
  the error.
- **Test suite discovery fails**: Fall back to prompting the user to enter a suite name
  manually rather than erroring out.
- **PR not found**: Verify the PR number/URL and repo.
- **PR body corrupted**: See "Editing PR descriptions safely" — always verify and retry.
