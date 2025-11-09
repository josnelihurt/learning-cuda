# Frontend Delivery Agents

Reusable prompts that coordinate feature delivery tasks for the web stack. Each agent focuses on a single stage so you can chain them to support planning, ticketing, implementation, and PR updates.

## Available Agents

- `orchestrator.prompt` – decides which downstream agent should run next.
- `feature-spec.prompt` – extracts requirements and drafts the implementation outline.
- `ticket-writer.prompt` – turns the plan into a GitHub issue.
- `pr-updater.prompt` – summarises commits and updates PR descriptions or comments.

Usage tips:
1. Start with `orchestrator.prompt` to confirm scope.
2. Run `feature-spec`, then `ticket-writer`, and finally `pr-updater` after coding.
3. Keep hand-offs short and link related tickets or PRs explicitly.

