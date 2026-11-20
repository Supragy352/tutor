# AGENTS.md — Guide for AI Agents Working on `tutor`

`tutor` is a local, [Codecrafters](https://codecrafters.io)-style hands-on coding tutor for learning deep technical subjects by building them stage by stage.

---

## 1. Project Focus & Strategic Direction

Any agent picking up an issue, designing a feature, or authoring a course in this repository should align with the owner's explicit direction:

1. **Prime Goal — Complete `build-your-own-x` Coverage & Deep Technical Courses:**
   - A north-star goal for `tutor` is covering **every single category** in [`codecrafters-io/build-your-own-x`](https://github.com/codecrafters-io/build-your-own-x) (all 30 categories: 3D Renderer, AI Model, Augmented Reality, BitTorrent Client, Blockchain, Bot, CLI Tool, Database, Distributed Systems, Docker, Emulator/VM, Front-end Framework, Game, Git, Memory Allocator, Network Stack, Neural Network, Operating System, Physics Engine, Processor, Programming Language, Regex Engine, Search Engine, Shell, Template Engine, Text Editor, Visual Recognition, Voxel Engine, Web Browser, Web Server), alongside rigorous engineering tracks like [`challenges/pid`](challenges/pid) and [`challenges/rl`](challenges/rl).
   - Coverage progress is tracked in [`docs/dashboard.html`](docs/dashboard.html) (Mission Control).
   - Course stages must teach the underlying math, physics, or systems concepts clearly without giving away solution code, while reference solutions under `challenges/<slug>/reference/` must be deterministic, self-contained, and verified against every test.

2. **Four Active Strategic Workstreams:**
   - **Design:** Keep the learner cockpit ([`src/web/index.html`](src/web/index.html)), landing page ([`docs/index.html`](docs/index.html)), and Mission Control issue tracker ([`docs/dashboard.html`](docs/dashboard.html)) cohesive, fast, accessible, and clear.
   - **Reasoning Handling:** Support configurable AI backends (OpenAI-compatible `/v1/chat/completions` endpoints, Ollama, and custom CLI agent commands) and deliberate reasoning/thinking configuration (moving away from hardcoded `"think": false`, feeding accurate uncommitted diffs + exit status + persisted hint history).
   - **Course Structure:** Evolve `challenge.toml` and course layouts beyond single-file `stdin`/`stdout` specs to support modular stage definitions, richer fixtures, and multi-mode verification harnesses.
   - **Architecture & Dependencies:** Do **not** feel constrained to keep the Rust codebase in a single file (`src/main.rs`) or at near-zero dependencies. Splitting [`src/`](src) into focused modules (`cli`, `runner`, `llm`, `server`, `challenge`, `progress`) and adding well-maintained crates (`clap`, `tiny_http`/`axum`, `shlex`, `tempfile`, etc.) is encouraged whenever it improves reliability or velocity.

3. **Test Harness Evolution — Beyond `stdin`/`stdout`:**
   - While black-box `args` + `stdin` -> `stdout` remains the baseline protocol, the test runner is evolving to support richer verification modes required by `build-your-own-x`:
     - Process timeouts, exit-code assertions, and safe concurrent pipe handling.
     - Language-specific test harnesses, filesystem/CLI workspace checks, interactive REPL/PTY protocols, TCP/HTTP network testing, and binary/image/headless emulator harnesses.

---

## 2. Repository Structure

```text
tutor/
├── AGENTS.md                   # Project focus, strategic tracks, and agent guidelines
├── Cargo.toml                  # Rust crate manifest
├── src/
│   ├── main.rs                 # Core CLI, test runner, LLM integration, and HTTP server
│   └── web/
│       └── index.html          # Local web dashboard served by `tutor serve`
├── docs/
│   ├── index.html              # Public landing page with interactive PID step-response plot
│   └── dashboard.html          # Mission Control: live GitHub Issues, CI/CD, and Build-Your-Own-X tracker
├── challenges/
│   ├── wc/                     # 3-stage course: build `wc` (lines, words, bytes)
│   │   ├── challenge.toml
│   │   └── reference/main.py
│   ├── pid/                    # 8-stage course: discrete PID controller, plant sim, IAE tuner
│   │   ├── challenge.toml
│   │   └── reference/main.py
│   └── rl/                     # 24-stage course: RL from scratch (CartPole, Pendulum, Q-learning, MLP, DQN, PPO, HPO)
│       ├── challenge.toml
│       └── reference/main.py
└── .github/workflows/
    ├── ci.yml                  # CI: rustfmt, clippy, build, 35-stage verification (Ubuntu + Windows), docs check
    └── pages.yml               # CD: deploys docs/ (Overview + Mission Control dashboard) to GitHub Pages
```

---

## 3. Verification & Development Workflow

Before opening a PR or concluding a task, agents **must** verify their changes locally:

1. **Format & Lint:**
   ```sh
   cargo fmt --all -- --check
   cargo clippy --all-targets --locked -- -D warnings
   ```
2. **Run Unit & Course Verification Tests:**
   ```sh
   TUTOR_LLM=off cargo test --locked
   ```
   - `cargo test` runs `bundled_challenges_pass_their_reference`, which executes every test of every stage in `challenges/*` against its `reference` command.
   - Note: `challenges/rl` requires Python 3 with `numpy` installed (`python -m pip install numpy`).
3. **Verify a Single Course Quickly:**
   ```sh
   cargo run -- verify challenges/pid
   cargo run -- verify challenges/rl
   ```

---

## 4. Key Technical Conventions for Agents

- **Cross-Platform Parity (Windows + Linux/macOS):**
  - CI runs on both `ubuntu-latest` and `windows-latest`.
  - Be careful with Windows paths containing spaces, `\\?\` verbatim UNC prefixes from `fs::canonicalize`, CRLF line endings, and `/bin/sh` execution inside Git for Windows hooks (`.git/hooks/post-commit`).
- **Determinism in Course Tests:**
  - When authoring or updating numerical/scientific courses (`pid`, `rl`, or new courses), specify exact formatting (e.g., 4 decimal places, `-0.0000` normalized to `0.0000`), avoid inputs that land on floating-point rounding boundaries, and use explicit seeded PRNGs (like `splitmix64` in `challenges/rl`) whenever randomness is involved.
- **State Persistence (`tutor.toml`):**
  - Learner workspaces store progress in `tutor.toml`. Keep schema changes backwards-compatible using `#[serde(default)]` where possible, and write state updates atomically so `tutor serve` and git hooks never read partially written files.
- **Web Dashboard & Docs Parity:**
  - If you change course formulas or behavior shown in [`docs/index.html`](docs/index.html) (such as the interactive PID simulator), keep the JavaScript implementation in sync with the course specification in `challenges/pid/challenge.toml`.
  - Keep [`src/web/index.html`](src/web/index.html) accessible, sanitized (`DOMPurify`), and protected against DNS rebinding (`Host` validation) and cross-site triggers (`X-Tutor` header).

---

## 5. Kanban Priority & Agent Dispatch Protocol

The owner organizes work on the live Mission Control Kanban board ([`docs/dashboard.html`](docs/dashboard.html), published at `https://supragy352.github.io/tutor/dashboard.html`), which syncs directly with GitHub Issue labels on `Supragy352/tutor`:

| Kanban Column | GitHub Label | Meaning for Agents |
|---|---|---|
| **Urgent** | `priority:urgent` | **Pick up first.** Highest-priority work explicitly queued by the owner. |
| **Up Next** | `priority:high` | Pick up after `priority:urgent` is clear (or when requested). |
| **Backlog** | `priority:normal` (or unlabeled) | Lower-priority backlog. |
| **Active / PR** | `status:in-progress` / `status:in-review` | Already claimed by an agent or waiting on an open PR review. Do **not** duplicate without checking existing PRs/branches. |
| **Done** | Closed issue | Completed. |

### Agent Pickup Rules
1. **Check Urgent Queue First:** Before selecting a task, inspect open issues ordered by Kanban lane:
   ```sh
   gh issue list --state open --label "priority:urgent" --json number,title,labels
   gh issue list --state open --label "priority:high" --json number,title,labels
   ```
2. **Claim When Starting:** When starting work on an issue, transition its Kanban label to `status:in-progress` (removing `priority:*` labels so it moves cleanly into the **Active / PR** column):
   ```sh
   gh issue edit <N> --remove-label "priority:urgent,priority:high,priority:normal" --add-label "status:in-progress"
   ```
3. **Move to In Review on PR Open:** Once a PR referencing `Closes #<N>` is opened with passing CI, switch `status:in-progress` to `status:in-review`:
   ```sh
   gh issue edit <N> --remove-label "status:in-progress" --add-label "status:in-review"
   ```

