#!/usr/bin/env bash
# PreToolUse(Bash): refuse to create a git branch.
#
# CLAUDE.md says commit on `main` and push directly. A permission `deny` rule catches the
# plain spellings; this catches them inside a compound command, which prefix matching
# cannot see.
set -uo pipefail

command=$(jq -r '.tool_input.command // ""')

if printf '%s' "$command" | grep -Eq \
	'git[[:space:]]+(checkout[[:space:]]+(-b|-B)|switch[[:space:]]+(-c|-C)|branch[[:space:]]+[^-[:space:]])'; then
	jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "CLAUDE.md: do not create branches — commit on main and push directly (git push -u origin main). If a branch is genuinely wanted, the user has to ask for it."
    }
  }'
fi
exit 0
