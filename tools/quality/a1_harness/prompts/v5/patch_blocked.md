Not recorded. The working tree is unchanged — `git status` reports no modified files and
no new ones, so nothing you have described exists on disk. On this task the diff is the
deliverable and the sentence only reports it, so an answer now is graded against an empty
patch however well it reads.

Do the work instead of describing it:

1. `read_file` the file you intend to change, so you have its exact text.
2. `edit_file` it with the old text copied character for character, or `write_file` a new
   file. The tool result will show you `git diff --stat`, which is how you know it landed.
3. Then `finish`.

You are {turns_used} turns into {max_turns}.
