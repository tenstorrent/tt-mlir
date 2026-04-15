# Stacked PRs Workflow

A workflow for splitting a large feature branch into reviewable PRs that build
on each other, while continuing active development.

## Structure

```
main
  └── pr/a    (PR1, targets main)
        └── pr/b    (PR2, targets pr/a)
              └── pr/c    (PR3, targets pr/b)
                    └── pr/d    (PR4, targets pr/c)
```

Each PR on GitHub targets its parent branch, so reviewers only see the
incremental diff.

## Setup

Start from your kitchen-sink development branch. Use interactive rebase to
reorganize commits into logical units, then branch off at each commit:

```bash
git rebase -i main
# reorder/squash commits into logical groups

git checkout -b pr/a <hash-of-commit-A>
git checkout -b pr/b <hash-of-commit-B>
git checkout -b pr/c <hash-of-commit-C>
git checkout -b pr/d <hash-of-commit-D>
```

Open PRs targeting the parent branch:
```bash
gh pr create --base main --head pr/a
gh pr create --base pr/a --head pr/b
gh pr create --base pr/b --head pr/c
gh pr create --base pr/c --head pr/d
```

## Rebasing the Whole Stack

After any upstream change (PR merged, review fix), rebase from the tip with
`--update-refs` to cascade all branch pointers in one command:

```bash
git fetch origin
git checkout pr/d   # tip of stack

git rebase --update-refs main
```

Then force push all branches:
```bash
git push --force-with-lease origin pr/a pr/b pr/c pr/d
```

Make it the default so you never have to remember the flag:
```bash
git config --global rebase.updateRefs true
```

## Addressing Review Comments

### Squash fix into the target commit

```bash
git checkout pr/b
# make fix
git add <files>
git commit --fixup=HEAD   # or --fixup=<hash/branch-name>

git checkout pr/d         # tip
git rebase -i --autosquash --update-refs main
# fixup is auto-placed after B and squashed, just confirm editor
```

### Insert a standalone commit mid-stack

There is no `--fixup` equivalent for non-squashing inserts. Use interactive
rebase from the tip, mark the target commit as `edit`, make the fix, then
continue:

```bash
git checkout pr/d
git rebase -i --update-refs main
# mark pr/b's commit as 'edit' in the editor

# git pauses after replaying B
git add <files>
git commit -m "address review: fix xyz"
git rebase --continue
# git replays C and D, updates all branch pointers
```

## Continuing Development

Keep your original branch as a personal development branch. New work goes
there; when ready, extract it as the next PR in the stack:

```bash
git checkout my-dev-branch
# ... code ...
git commit -m "wip"

# When ready to add to stack:
git checkout -b pr/e pr/d
git cherry-pick <commits-from-dev-branch>
gh pr create --base pr/d --head pr/e
```
