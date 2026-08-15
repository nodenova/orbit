{task_prompt}

The repository is at {worktree}.

Work in this order and do not skip a step:

1. **Find and read** the file you need to change before editing anything.
2. **Edit it** with `write_file` or `edit_file`.
3. **Confirm the edit landed** — `run` the command `git diff` and check your change appears
   in the output. If the diff is empty, the edit did not happen and you must do it again.
4. **Run the checks** — `pytest -q`, then `ruff check .`, then `mypy` — and fix what they
   report.
5. Only then call `finish`, in one sentence, saying what you changed and why.

An answer that describes an edit which is not in `git diff` scores zero, however good the
description is. The diff is the deliverable; the sentence only reports it.
