# Claude (claude.ai Projects)

Prefer the skill: zip the `skills/blockout-to-real/` folder and upload it under **Settings > Capabilities > Skills**.
Claude then loads `SKILL.md`, the references and the tools on demand. Its code environment has no Blender and no
fal key, so it can write scene specs, read your contact sheets and plan edits; you run the stages locally.

Without skills, use a Project:

1. Paste `adapters/universal-system-prompt.md` (the part below the line) into the project instructions.
2. Add to project knowledge: `SKILL.md`, `references/inputs.md`, `references/target.md`, `references/loop.md`
   and `references/lessons.md`.

For Claude Code, install the plugin or copy the skill folder (see the repository README). Claude Code runs the
CLI, fal and Blender itself, looks at every sheet it makes, and starts the fresh reviewer and judge as subagents.
