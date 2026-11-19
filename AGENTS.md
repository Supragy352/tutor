# AGENTS.md — Guide for AI Agents Working on `tutor`

`tutor` is a local, [Codecrafters](https://codecrafters.io)-style hands-on coding tutor for learning deep technical subjects by building them stage by stage.

---

## 1. Project Focus & Strategic Direction

Any agent picking up an issue, designing a feature, or authoring a course in this repository should align with the owner's explicit direction:

1. **Primary Focus — Deep Engineering & ML Courses:**
   - The core value of `tutor` lives in rigorous, from-scratch engineering and machine learning curricula (such as [`challenges/pid`](challenges/pid) and [`challenges/rl`](challenges/rl)) and expanding the catalog with new deep technical courses.
   - Course stages must teach the underlying math, physics, or systems concepts clearly without giving away solution code, while reference solutions under `challenges/<slug>/reference/` must be deterministic, self-contained, and verified against every test.

2. **Architecture & Dependencies — Prioritize Robustness and Features Freely:**
   - Do **not** feel constrained to keep the Rust codebase in a single file (`src/main.rs`) or at near-zero dependencies when implementing fixes or new capabilities.
   - Splitting [`src/`](src) into clean, focused modules (e.g., `cli`, `runner`, `llm`, `server`, `challenge`, `progress`) and adding well-maintained crates (e.g., `clap`, `tiny_http` / `axum`, `shlex`, `tempfile`) is encouraged whenever it improves reliability, cross-platform correctness, or developer velocity.

3. **Test Harness Evolution — Beyond `stdin`/`stdout`:**
   - While black-box `args` + `stdin` -> `stdout` remains the baseline protocol, the test runner is evolving to support richer verification modes, including:
     - Process timeouts, exit-code assertions, and safe concurrent pipe handling.
     - Language-specific test harnesses and interactive protocols (e.g., TCP sockets, HTTP servers, multi-turn REPLs, or file artifacts) for courses that cannot be expressed as one-shot `stdin`/`stdout` filters.

4. **Configurable AI Backends:**
   - `tutor` uses two tiers of AI assistance:
     1. **Fast advisory diff reviewer** after each test run (currently Ollama via `TUTOR_LLM` / `TUTOR_MODEL`).
     2. **Agentic tutor & course designer** for `tutor hint`, automatic escalation after repeated failures, and `tutor new` (currently `claude -p`).
   - Avoid hardcoding assumptions that lock the codebase to a single vendor. Support configurable backends: OpenAI-compatible HTTP endpoints (`/v1/chat/completions` in addition to Ollama `/api/generate`) for reviews, and configurable agent CLI commands / providers for hints and course generation.

---

## 2. Repository Structure

```text
tutor/
├── Cargo.toml                  # Rust crate manifest
├── src/
│   ├── main.rs                 # Core CLI, test runner, LLM integration, and HTTP server
│   └── web/
│       └── index.html          # Local web dashboard served by `tutor serve`
├── docs/
│   └── index.html              # Public landing page with interactive PID step-response plot
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
└── .github/workflows/ci.yml    # GitHub Actions CI (Ubuntu + Windows, Python 3.12 + numpy, Rust stable)
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
