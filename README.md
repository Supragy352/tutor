# tutor

A [codecrafters](https://codecrafters.io)-style coding tutor for learning anything by building it.

A course is a series of small stages. You write code in your own repo, commit, and a git hook runs
the stage's tests. Pass and the next stage unlocks. Along the way:

- **Tests decide.** Each stage has black-box tests: fixed arguments and stdin, exact expected stdout.
  Earlier stages re-run on every check, so nothing regresses.
- **A local LLM reviews.** After each run, a small model served by [Ollama](https://ollama.com) reads
  your latest commit and leaves up to three short notes. It's advisory only and never writes your
  solution.
- **Claude steps in when you're stuck.** After 3 failed runs on one stage (or whenever you run
  `tutor hint`), [Claude Code](https://claude.com/claude-code) reads your folder and explains the
  concept you're missing, without writing the fix.
- **Claude designs new courses.** `tutor new` describes a goal to Claude Code, which writes the stages,
  the tests and a hidden reference solution. tutor only accepts the course once that solution passes
  every test.

There's also a local web dashboard (`tutor serve`) that shows stage instructions, your progress, test
results, reviewer notes and hints, and updates live as you commit.

## Courses included

| Course | Stages | Language | What you build |
|---|---|---|---|
| [`challenges/wc`](challenges/wc) | 3 | any | A tiny `wc`: lines, words, bytes |
| [`challenges/pid`](challenges/pid) | 8 | any (written for C) | A PID controller with clamping, anti-windup, a plant simulator and a gain tuner |
| [`challenges/rl`](challenges/rl) | 24 | Python + numpy | Reinforcement learning from scratch: CartPole and Pendulum physics, tabular Q-learning, an MLP with backprop and Adam, DQN, PPO, and hyperparameter search |

The tests only see your program's stdin and stdout, so any course can be done in any language. The
RL course leans on Python because its later stages train neural networks.

The RL course follows Shawn Hymel's
[reinforcement-learning-demos](https://github.com/ShawnHymel/reinforcement-learning-demos), rebuilding
by hand what those notebooks get from gymnasium and Stable-Baselines3.

## Requirements

- Rust (to build tutor), git, and whatever your course needs (`gcc` for the PID course in C, Python 3
  with numpy for the RL course).
- Python 3, to run reference solutions when verifying or creating courses.
- Optional: [Claude Code](https://claude.com/claude-code), installed and logged in, for hints and
  `tutor new`. tutor calls it as `claude -p`. Without it, everything else still works.
- Optional: [Ollama](https://ollama.com) for the local reviewer. If it isn't reachable, tutor skips
  the review.

## Install

```sh
git clone https://github.com/Supragy352/tutor
cargo install --path tutor
```

Keep the clone around: the courses live in its `challenges/` folder.

## Start a course

```sh
mkdir rl && cd rl && git init
tutor init ../tutor/challenges/rl python rl.py
```

`tutor init <course> [--build "<cmd>"] <run command...>` records how to build and run your program,
installs a `post-commit` hook, and prints stage 1. For a compiled language, pass a build command:

```sh
tutor init ../tutor/challenges/pid --build "gcc -O2 -Wall -o pid main.c -lm" ./pid
```

Then write code and commit. Every commit runs the checks. You can also run them by hand:

```sh
tutor test        # build, test, review; advances on pass
tutor test -v     # same, logging every build, test command and model call
tutor show        # print the current stage
tutor hint        # ask Claude Code for help
tutor serve       # dashboard at http://localhost:4242
```

Your progress lives in `tutor.toml` in your workspace. Add it to your `.gitignore`.

## The local reviewer

By default tutor asks Ollama at `http://localhost:11434` for model `qwen3.6:35b`. Point it anywhere
with environment variables:

| Variable | Meaning |
|---|---|
| `TUTOR_LLM` | Ollama base URL, e.g. `http://my-gpu-box:11434`. Set to `off` to disable reviews. |
| `TUTOR_MODEL` | Any model you've pulled, e.g. `qwen2.5-coder:7b` for smaller machines. |
| `TUTOR_VERBOSE` | Any value turns on `-v` logging, including inside the git hook. |

A remote Ollama box works well over a private network such as Tailscale. Ollama only listens on
localhost by default, so expose it with `OLLAMA_HOST=0.0.0.0` or `tailscale serve`, never directly to
the internet.

## Make your own course

```sh
cd tutor
tutor new lisp "Build a Lisp interpreter with a REPL: tokenizer, parser, eval, lambdas and closures"
```

Claude Code writes `challenges/lisp/challenge.toml` and a reference solution, then tutor verifies the
reference against every test. Failures go back to Claude Code, up to three attempts. You can also
write courses by hand:

```toml
title = "Build your own wc"
reference = "python reference/main.py"   # run from the course folder by `tutor verify`

[[stages]]
name = "Count lines"
instructions = """
Markdown shown to the learner. Explain the concept and spell out the exact stdin/stdout protocol.
"""
tests = [
  { stdin = "a\nb\nc\n", expect = "3" },
  { args = ["-w"], stdin = "one two\n", expect = "2" },   # args are optional
]
```

Output is compared after normalising line endings and trimming trailing whitespace. Check a course
with:

```sh
tutor verify challenges/lisp
```

`cargo test` verifies every bundled course this way.

## Security notes

- `tutor serve` binds to `127.0.0.1` only, rejects requests whose `Host` isn't `localhost` or
  `127.0.0.1` (which blocks DNS rebinding), and requires a custom header on actions (which blocks
  cross-site requests).
- `tutor new` runs Claude Code with `--permission-mode acceptEdits` inside `challenges/<slug>`, so it
  can write files there without asking. Hints run Claude Code in your workspace with its default
  permissions.
- Test commands run your own code on your machine, unsandboxed. Only run courses you trust.

## License

[MIT](LICENSE)
