# Senpai agent plugin

This directory is the OpenHands-native integration point for Senpai workflow
capabilities. Its manifest lives at `.plugin/plugin.json`.

OpenHands receives this directory through `PluginSource` before the first user
message. It natively loads:

- `skills/` as a progressively disclosed workflow catalog; and
- `hooks/hooks.json` for early command-policy and lifecycle feedback.

GitHub mutations and training supervision are native typed Senpai tools, not
skill shell commands. Exa is also a skill/script integration rather than an MCP
server; launch preflight makes one `instant` publication search with one result
to validate the key.

The Python runtime registers the GitHub tools and exposes only those valid for
the current role:

- advisors receive `create_assignment`, `publish_advisor_branch`,
  `repair_assignment_routing`, `send_assignment_feedback`,
  `request_assignment_revision`, `accept_result_on_current_base`,
  `merge_experiment`, and `close_experiment`;
- students receive `post_assignment_comment` and `submit_experiment_result`; and
- both roles receive `get_prs` and `respond_to_human_issue`.

Each tool has one operation-specific schema without a union wrapper and a
complete model-facing description. The
skills in this plugin explain when to use those tools and provide workflow
examples; they do not implement mutations or carry credentials. The plugin has
no MCP server. The Python runtime binds the authenticated role and adds the
canonical `ADVISOR:` or `STUDENT:` prefix to Senpai-authored GitHub comments;
tool payloads contain only the unprefixed message text.

Keep every Senpai-owned skill used by a live advisor or student here rather
than relying on a provider's user skill directory. Target repositories may
supply project skills separately. Human onboarding and developer guides stay
under the runner's `.agents/skills` and are not installed into pods. Never
commit secret values. The plugin remains the source of truth for reusable
runtime guidance, while Python remains the source of truth for verified state
changes.
