#!/usr/bin/env bash
# PostToolUse(Edit|Write|NotebookEdit): format and lint the one file that was touched.
#
# The repo rule is "format and lint every file you create or change, passing the touched
# paths and not `.`" — advisory in CLAUDE.md, deterministic here. Exit 2 hands whatever
# `--fix` could not fix back to the model, which is the part that needs a decision.
set -uo pipefail

file=${1:-}
case "$file" in
*.py) ;;
*) exit 0 ;;
esac
[ -f "$file" ] || exit 0

root=${CLAUDE_PROJECT_DIR:-}
if [ -n "$root" ]; then
	case "$(cd "$(dirname "$file")" && pwd -P)/" in
	"$(cd "$root" && pwd -P)"/*) ;;
	*) exit 0 ;;
	esac
fi

ruff=$root/.venv/bin/ruff
[ -x "$ruff" ] || ruff=$(command -v ruff) || exit 0

"$ruff" format --quiet "$file" >/dev/null 2>&1
"$ruff" check --fix --quiet "$file" >/dev/null 2>&1

if ! remaining=$("$ruff" check --no-fix --quiet --output-format concise "$file" 2>&1); then
	printf 'ruff still reports these in %s — fix them or add a scoped `# noqa: RULE` with a reason:\n%s\n' \
		"$file" "$remaining" >&2
	exit 2
fi
exit 0
