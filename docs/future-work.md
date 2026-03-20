# Future Work

## SWE-bench Style Patch Generation Evaluation

The current experiment evaluates **bug detection** (did the tool find the bug?). A natural extension is **bug fixing** (can the tool write a correct patch?), following the SWE-bench evaluation paradigm:

| | Current (detection) | Extension (patch generation) |
|---|---|---|
| **Task** | Review introducing PR, find bugs | Given bug report at introducing commit, write a fix |
| **Input** | Introducing PR diff + repo context | Issue body + repo at introducing commit |
| **Output** | Comments (file, line, description) | Patch (code diff) |
| **Metric** | Catch rate (file+line match) | Resolved rate (fix tests pass) |
| **Ground truth** | Buggy lines from diff intersection | Fix commit diff + test suite |

The dataset construction pipeline already provides everything needed:
- `base_commit` (the buggy state) as the starting point
- `fix_commit` as the gold-standard solution
- `bug_description` (from issue body or fix PR) as the task prompt
- Issue bodies, PR discussions, and review comments as optional context

Implementation would add a new evaluation mode (`bugeval evaluate --mode patch`) that:
1. Checks out the repo at `base_commit` (the buggy state)
2. Presents the agent with the bug description and asks it to write a fix
3. Applies the agent's patch and runs the repo's test suite
4. Compares against the fix commit: exact match, semantic equivalence (tests pass), or failure

This reuses the same cases, blame, and ground truth pipeline -- only the evaluation runner and scoring change.
