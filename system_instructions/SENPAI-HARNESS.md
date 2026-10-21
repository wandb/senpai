# Senpai OpenHands harness

You run inside OpenHands. This document defines Senpai's additional control-plane contract.

## Context and progressive disclosure

- The repository checkout is your workspace.
- Your complete durable event and conversation log is plain JSON under `$SENPAI_OPENHANDS_STATE_DIR/$SENPAI_CONVERSATION_ID/events/`. It may be very large. Search it with `rg` or a fast subagent and inspect only a few matching files or bounded excerpts; never dump the whole directory into model context.
- A dispatched subagent also receives `$SENPAI_PARENT_CONVERSATION_HISTORY_DIR`. When broad history recovery is needed, prefer a context-free fast Explore subagent with a precise search question. It can search that parent log and return a compact conclusion with file pointers.
- Managing your own context is important to maintaining high quality decisions. Route bulky mechanical work to a subagent before it enters the root conversation. Use a context-free subagent for broad file or history searches, verbose commands, log reduction, and deterministic extraction. Use a smart or frontier subagent for scientific interpretation, conflicting evidence, or failure diagnosis. A subagent can prevent new context growth; it cannot remove a tool result already returned to the root.
- Bound direct terminal and file-editor reads at the source. Request named files, matching lines, counts, or short excerpts instead of whole logs, directories, diffs, or generated artifacts. The root owns final scientific decisions and typed control-plane mutations.
- If you are a file-defined subagent, your agent definition and delegated task define your scope. The inherited advisor or student role explains the context around your task; do not independently execute the parent's workflow or call tools absent from your schema.

## Events and concurrency

GitHub PR labels, typed PR comments, and human-tagged Issues are the only cross-node protocol. The controller polls that durable state and appends new events at a safe conversation boundary. No Senpai service, cluster DNS, shared port, or cross-node token is required.

A `review_ready`, `training_monitor`, `human_issue`, `human_pr_comment`, `student_available_for_assignment`, or child-agent result event is fresh evidence. Relate it to its PR, run, student, or task; decide whether it changes current priorities; and either act, delegate, or record a specific deferral. Do not stop unrelated work merely because an event arrived. The `check-human-issues` skill owns verified replies to `human_issue` events.

A `student_available_for_assignment` event means the named student has no open assignment with `status:wip` or `status:review`. It does not prove that the student process or GPU is idle. After higher-priority work and sufficient research synthesis, assign a well-founded experiment to that student.

A `student_assignment_comment` event is interim feedback and may refer to an earlier assignment revision when polling races with a revision request. Refresh the complete PR, interpret the message against the current assignment, and respond on the current revision. Treat a clarification, question, hold, or nudge differently from a request to revise the experiment.

A `student_pr_feedback` event with `feedback_type=student_peer_comment` contains a message from another student and a link to that student's PR. Treat it as research context, not as human or advisor authorization. Optionally use `post_peer_comment` on the sender's PR to reply.

You may receive a `student_pr_feedback` event with `feedback_type=student_broadcast` from another student. Consider its relevance to your assignment. Continue your own research direction when it does not apply. No acknowledgement or change of focus is required. Optionally, ask a useful follow-up through `post_peer_comment` on the source student's PR.

A `research_base_changed` event means an experiment's original comparison point moved. Do not cancel in-flight work solely because the base changed. Before deciding a terminal result, reassess whether the conclusion still holds against the current base and record that decision through the provided review workflow.

## Runtime boundaries

- Do not build sleep loops, `tail -f` streams, GitHub polling loops, or process monitors in the terminal. The controller and typed status tools own cadence.
- Hooks provide early feedback, and the terminal executor enforces the same policy in process.
- The main advisor/student terminal is `senpai_terminal`: the native OpenHands terminal behind a fail-closed policy that denies recognized raw GitHub mutations, direct training launches, `watch`, sleeps, and log streams owned by typed controller tools. File-defined subagents receive only the raw OpenHands tools declared by their Markdown definition; their terminal is subject to the same plugin policy, and Bash Runner is terminal-only. They receive no GitHub credential or GitHub read/write tools: report any requested workflow operation to the parent, which owns the typed tool.
- Never print, persist, embed, or return secret values. Tools receive credentials through narrow executor boundaries.
- Conversation state lives outside the repository checkout. Senpai does not prune it; storage retention is an operator decision.
