# Senpai agent plugin

This directory is the OpenHands-native integration point for Senpai workflow
capabilities. Its manifest lives at `.plugin/plugin.json`.

Role images install this directory as the read-only `$SENPAI_PLUGIN`.
OpenHands receives that explicit path through `PluginSource` before the first
user message. It natively loads:

- `skills/` as a progressively disclosed workflow catalog; and
- `hooks/hooks.json` for early command-policy and lifecycle feedback.

The inclusion path is explicit:

- The [advisor](../../Dockerfile.advisor) and [student](../../Dockerfile.student)
  Dockerfiles copy `plugins/senpai` to `/opt/senpai-plugin` and set
  `SENPAI_PLUGIN` to that path.
- The [runner](../../senpai_agent/openhands_runner.py) selects `--plugin-dir`,
  then `SENPAI_PLUGIN`, then the source-relative `plugins/senpai` default.
  `LocalConversation` receives `plugins=[PluginSource(source=str(config.plugin_dir))]`.
- [Delegation](../../senpai_agent/delegation.py) passes that same directory to
  each child through `--plugin-dir`.

This selects one bundle; an override replaces it. There is no extra-plugin list
or automatic loading from target or home plugin directories. To extend the
standard image, integrate the required assets into this directory and rebuild
the image. Source edits and local plugin tests do not require a rebuild;
deploying those edits to the packaged cluster runtime does. Existing agent
context does not hot-reload plugin changes.

For local tests, keep `SENPAI_PLUGIN` consistent with any `--plugin-dir` override
because helper examples use that environment variable. The bundled hook
manifest targets `/opt/senpai-venv/bin/python`; host hook tests need a development
copy that names the host's trusted interpreter with `-P`. The selector does not
provide a standalone host launcher. See the [runtime deployment notes](../../README.md#other-deployment-environments).

GitHub mutations and training supervision are native typed Senpai tools, not
skill shell commands. Exa is also a skill/script integration rather than an MCP
server; launch preflight makes one `instant` publication search with one result
to validate the key.

The Exa and W&B skills use `uv run --no-sync` to run Python in the target
environment without syncing project dependencies. Read helper libraries from
`$SENPAI_PLUGIN`; keep analysis scripts and generated outputs in the target
workspace or `/tmp`.

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
