# The Assistant: how it works

A conversational front door to the deterministic engine. The model proposes; the engine proves. Every
number comes from a recorded run, never from the model. This is the design behind `dancr/core/assistant/`
and `dancr/ui/assistant.py`.

## The pieces

| Module | Does |
|---|---|
| `core/assistant/client.py` | The model boundary: `Message`/`ToolCall`/`ToolSpec`, an OpenAI-style `OpenAIProvider` over httpx (any `/v1/chat/completions` endpoint — Fireworks, OpenAI, a local server), a `FakeProvider` for tests, and `ModelSettings` (base URL, model, key, temperature, `max_tokens`). Error mapping turns 401 into "bad key", 402/429 into a graceful pause. |
| `core/assistant/context.py` | Re-exports the project **profile** the model reads, built by `core/profile.py`: tables, column roles/types/ranges/counts/categories and relations, from `core.understand`. Values are stripped of control characters and capped, and every payload is wrapped by `data_block()` so data can never close its own delimiter. The model never reads rows here. The same builder backs the knowledge-base export (`dancr context` / `build_context`), so the Assistant's profile and an agent's RAG document stay one schema. |
| `core/assistant/prompts.py` | The system prompt (the trust contract, the data-is-not-instructions rule, the spec format) and a generated answer-engine reference. |
| `core/assistant/tools.py` | The **tool surface**. Read tools: `list_tables`, `list_steps` (the project's map — every step's id, type and title), `describe_table`, `get_stats`, `get_sample`, `read_question`, `suggest_answers`, `answer_reference`, `list_node_types`, `formula_reference`, `node_status`, `list_connections`. Terminal tools: `ask_choice` (show options, never guess), `propose` (an answer spec or a hand-built step list) and `propose_edits` (a batch of project changes — rename steps, change settings, label columns, set Inputs). All are thin wrappers over existing machinery; the two `propose*` tools validate against the engine **without touching the project**. |
| `core/assistant/session.py` | One conversation: builds the messages (system prompt + profile block + thread), runs the tool loop with budgets, and returns an `AssistantReply`. It also checks every figure in a reply against what the tools returned; anything unbacked is flagged `unverified-figure`. |
| `core/assistant/store.py` | The thread, saved in the project's `meta["assistant"]`, so the project format does not change. Only the question, proposal, assumptions, finding and built node/answer are kept — never raw prompts or data. |
| `ui/assistant.py` | The chat in the right dock (`SideDock`: Settings / Assistant tabs). One vertical column that always wraps (never a horizontal scrollbar), with a **live working card** showing each step as the engine is used, typed cards (reply / plan / choice / note / action), context **chips** with an **Add context** picker, a composer that grows and recalls past questions (↑ / Ctrl+↑), history that stays pinned to the bottom, copy on a reply, and a session cost meter. Cards are flat with a hairline border, radius 4, and the accent only on the primary action (the app's own language). The plan card leads with its title, then the step chain, a collapsible assumptions line and a primary **Build & run**; once built it shows the engine's **Result** and **Canvas / Save / Replace**. It changes nothing itself: a proposal is emitted to the window. |
| `ui/mainwindow.py` | `focus_assistant`, `apply_assistant_proposal` (builds through the document's undo stack), `_apply_assistant_steps`, `_assistant_reveal`, `assistant_save_as_project`, `assistant_replace_canvas`, and the run-finished hook that hands the engine's finding back to the card. |
| `core/answers.py` | `project_from_answer` / `save_as_project`: the steps behind one answer as a project of their own (for "Save as project…"). |
| `ui/commands.py` | `SetThread` (the conversation is undoable and marks the project changed) and `ReplacePipeline` (the whole canvas swaps as one undo step, for "Replace the canvas"). |
| `headless.py`, `cli.py`, `mcp_server.py` | `assistant_turn` runs one turn (and optionally builds+runs) with no window: `dancr assistant` and the MCP `assistant` tool both call it. |

## The path a question takes

1. `AssistantPanel.send()` ensures a deep data model is ready (`Understanding.when_full`), snapshots the
   project (`Document.snapshot_executor`), and runs `AssistantSession.turn` on the worker pool.
2. The model may call read tools, each executed against the snapshot. `get_sample` is refused unless the
   person allowed sample rows; by default only profiles, statistics and the engine's own answers are sent.
3. The model ends with `propose`: a spec (preferred), a list of steps, or plain prose. The session does
   **not** change the project.
4. The reply is shown as cards. A figure not seen in a tool result is flagged, not trusted.
5. **Build & run** emits `buildRequested`; `MainWindow.apply_assistant_proposal` builds the answer (or the
   steps) inside one undo macro, runs it, and passes the run's `finding` back to the card.
6. Once built, the card shows the engine's **Result** and offers **Canvas** (reveal the step chain on the map,
   its branch lit), **Save** (just its steps, as a new file) and **Replace** (only its steps, one undo step,
   confirmed first). The conversation is saved with the project, and undoing a thread change restores the
   earlier conversation.
7. While the model works, the live card lists what it is doing (`read_question`, `list_connections`,
   `get_stats`…) with an elapsed readout, so the panel is never a blank pause.

### Changing the project, not just answering (renames, labels, settings)

"Rename all the steps to something friendlier" is not a dataflow answer, so it goes through `propose_edits`:
one call that batches every change (rename, `set_params`, `column_label`, `set_input`). The reply is a
**Changes to the project** card listing each edit; **Apply** makes them all in one undo step. This is how the
Assistant "just does things" while keeping the approval gate.

### Staying on task

- A turn is bounded (rounds and engine calls). If the model reaches the limit, the reply says so with a
  **Carry on** button — never a dead end.
- The same tool with the same arguments more than twice (or any one tool more than five times) is answered
  with a nudge instead of being run again; if a whole round is repeats, the turn ends with "tell me which
  step to change, or try a different way". This stops the old `list_node_types` × 14 loop.
- `list_steps` is what the model reads to see the map, so it does not have to guess the project's steps.
- A reply cut off by the model's token limit (`finish_reason: "length"`) is flagged **truncated** and the card
  offers a **Carry on**, rather than presenting a half answer as complete.
- A built plan is folded back into the turn that proposed it (its `node`, `answer` and the engine's `finding`),
  so reopening the project shows it **built, with its Result** — it is never offered to be built a second time.

## The trust contract, enforced in code

- **Engine does the maths.** `propose` only plans; the window runs the plan; the finding comes from the
  run's `report`. A reply is scanned for numbers and any figure no tool result (or the profile's own engine
  statistics, or an earlier verified turn) backs is flagged and **named** ("Not backed by a run: 987654").
  Written forms are matched either way (`50` ≡ `50.0` ≡ `1,234`); only a single-digit integer is treated as a
  structural count ("2 tables") rather than a claim about the data. Category *cell values* never count as a
  source, so a figure copied from a cell is still unverified.
- **No invented joins.** Connections are the engine's own `understand` relations; the model can only
  reference tables and columns that exist, and `recipes.plan` rejects anything else.
- **Data is never instructions.** Data-derived text is sanitised, wrapped in a labelled block that it
  cannot close, and the system prompt states the hierarchy. The tool surface has no shell, no network and
  no arbitrary writes, so the blast radius of a successful injection is a proposal the person can decline.
- **Nothing is applied without approval.** The session is pure: it returns a proposal. Only the window
  applies it, undoably.

## Connecting data that does not obviously relate (Phase 2)

- The model reads the engine's relations from the profile, and can call `list_connections` for the exact
  match % and cardinality. It never invents a key.
- When a pair of tables has more than one plausible link, the model calls `ask_choice`; the reply is a card
  of options, and picking one sends it back as the next turn. The engine still checks whatever is built.
- When tables are dropped or loaded, the panel offers **Profile these tables** / **How do these tables
  connect?** — it does not call the model on its own, so a drop never spends tokens without a click.
- Composer commands expand to clear directives: `/profile`, `/connections`, `/explain`, `/clean`,
  `/report`; `/build <question>` passes the question through; `/undo` acts at once (no model call).

## Adding to it

- **A new read tool:** a `_t_<name>` method on `ToolRunner` and its schema in `schemas()`. It must return
  `ToolOutcome(content=...)`; never mutate the project.
- **A new model provider:** implement `Provider.chat` and pass it to `AssistantSession`, or extend
  `provider_for`. Keep `Message`/`ToolCall` as they are so the loop is unchanged.
- **A new card:** a `QFrame` in `ui/assistant.py`, added in `_got_reply`/`_reload_cards`.

## Privacy, cost and keys

- The first time real data would leave the machine, the panel says exactly what is sent (a profile, not
  rows, unless sample rows are allowed) and asks; **Allow sending data** in the ⋮ menu can revoke it. Consent
  is remembered **per endpoint**, so pointing DANCR at a different model server asks again rather than sending
  silently. A rigged/fake provider never leaves the machine, so it is never asked. Headless turns (CLI/MCP)
  have no window to ask, so they state the egress in the result (`sent_to`) and the log instead.
- Only the profile, the thread, your questions and the engine's tool results are sent. Sample rows are off
  by default. This is the only feature that uses the internet.
- The footer shows the session's **exact token counts** and a rough cost (input / cached-input / output
  prices are constants at the top of `ui/assistant.py`).
- Keys/settings live in `QSettings` (never in a project file): `assistant/api_key`, `assistant/base_url`,
  `assistant/model`, `assistant/allow_samples`, `assistant/consent`. `ModelSettings.from_env()` reads
  `DANCR_ASSISTANT_API_KEY` / `DANCR_ASSISTANT_BASE_URL` / `DANCR_ASSISTANT_MODEL` (and
  `FIREWORKS_API_KEY`). `DANCR_ASSISTANT_FAKE=1` runs with a scripted fake and no key.
- The project is marked changed when the conversation changes, so it is saved with the file (and the change
  is undoable). A funded shared test key and a small gateway are a later step; the provider boundary is
  where they plug in.

## Not yet

- A persisted, reviewable connection map (the engine's relations are surfaced on demand, not saved as an artifact).
- The funded shared test key / gateway; `.dancr` project-file format changes are deliberately avoided.
- Automatic turn-taking (the model proposing next steps on its own, without being asked).
- Multi-turn result narration: after a run the card shows the engine's own finding; the model is not yet
  re-invoked to explain it (the next question does that).
