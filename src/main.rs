mod cli;

use cli::{Action, USAGE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::{env, fs};

static VERBOSE: AtomicBool = AtomicBool::new(false);
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Live progress on stderr, so stdout (and the saved report) stay the same with or without -v.
macro_rules! verbose {
    ($($arg:tt)*) => {
        if VERBOSE.load(Ordering::Relaxed) {
            eprintln!("[tutor] {}", format!($($arg)*));
        }
    };
}

const PROGRESS: &str = "tutor.toml";
const ESCALATE_AFTER: usize = 3;
const INDEX: &str = include_str!("web/index.html");

#[derive(Deserialize)]
struct Challenge {
    title: String,
    reference: Option<String>,
    stages: Vec<Stage>,
}

#[derive(Deserialize)]
struct Stage {
    name: String,
    instructions: String,
    tests: Vec<Test>,
}

#[derive(Deserialize)]
struct Test {
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    stdin: String,
    expect: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
struct LastRun {
    stage: usize,
    ok: bool,
    report: String,
    review: Option<String>,
    hint: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct Progress {
    challenge: PathBuf,
    run: String,
    build: Option<String>,
    stage: usize,
    #[serde(default)]
    fails: usize,
    last: Option<LastRun>,
}

fn main() {
    let raw: Vec<String> = env::args().skip(1).collect();
    let parsed = match cli::parse_args(&raw) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    VERBOSE.store(
        parsed.verbose || env::var_os("TUTOR_VERBOSE").is_some(),
        Ordering::Relaxed,
    );
    let result = match parsed.action {
        Action::Help => {
            println!("{USAGE}");
            Ok(())
        }
        Action::Version => {
            println!("tutor {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Action::New { slug, goal } => new(&slug, &goal),
        Action::Verify { dir } => verify(&dir).and_then(|(report, ok)| {
            println!("{report}");
            if ok {
                Ok(())
            } else {
                Err("reference solution fails".into())
            }
        }),
        Action::Init { dir, build, run } => init(&dir, build, &run),
        Action::Show => load().map(|(p, c)| show(&c, p.stage)),
        Action::Test => test(),
        Action::Hint => current_hint().map(|h| println!("{h}")),
        Action::Serve { port } => serve(port),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn read_challenge(dir: &Path) -> Result<Challenge, String> {
    let path = dir.join("challenge.toml");
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn load_progress_from(path: &Path) -> Result<Progress, String> {
    let text = fs::read_to_string(path)
        .map_err(|_| format!("no {} here; run `tutor init` first", path.display()))?;
    toml::from_str(&text).map_err(|e| e.to_string())
}

fn load() -> Result<(Progress, Challenge), String> {
    let p = load_progress_from(Path::new(PROGRESS))?;
    let c = read_challenge(&p.challenge)?;
    Ok((p, c))
}

fn save_to(path: &Path, p: &Progress) -> Result<(), String> {
    let data = toml::to_string(p).map_err(|e| e.to_string())?;
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(PROGRESS);
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!(".{name}.tmp.{}.{seq}", std::process::id()));

    let write_res = (|| -> std::io::Result<()> {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(data.as_bytes())?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp);
        return Err(e.to_string());
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(e.to_string());
    }
    Ok(())
}

fn save(p: &Progress) -> Result<(), String> {
    save_to(Path::new(PROGRESS), p)
}

fn new(slug: &str, goal: &str) -> Result<(), String> {
    new_in(Path::new("challenges"), slug, goal, claude)
}

fn new_in<F>(root: &Path, slug: &str, goal: &str, mut run_agent: F) -> Result<(), String>
where
    F: FnMut(&str, &Path, &[&str]) -> Result<String, String>,
{
    cli::validate_slug(slug)?;
    let dir = root.join(slug);
    let existed = dir.exists();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        let example = include_str!("../challenges/wc/challenge.toml");
        let mut prompt = format!(
            "You are designing a hands-on coding course in the style of codecrafters.io. The learner's goal:

{goal}

Write `challenge.toml` in the current folder. Break the project into 5-8 small stages, each building on the last, ordered so the learner arrives at a sensible architecture. Each stage has `name`, `instructions` (what to build and why in a few short paragraphs; explain the concept, never give code) and `tests`.

Tests are black-box: the learner's program is started with `args`, fed `stdin`, and its stdout must equal `expect` after trimming trailing whitespace and normalising line endings. Fully specify the stdin/stdout protocol in the instructions. For numeric output, specify exact formatting (e.g. fixed decimal places) and pick inputs whose results don't land on rounding boundaries. Give each stage 2-4 tests including edge cases.

Also write a reference solution in Python 3 (standard library only) under `reference/` that passes every test of every stage. Put the top-level key `reference` (the command that runs it from this folder, e.g. `reference = \"python reference/main.py\"`) at the top of challenge.toml, before any [[stages]].

Format example from a different course:

{example}"
        );
        for attempt in 1..=3 {
            println!("Asking Claude Code to design the course (attempt {attempt}/3)...");
            run_agent(&prompt, &dir, &["--permission-mode", "acceptEdits"])?;
            let (report, ok) = verify(&dir).unwrap_or_else(|e| (e, false));
            println!("{report}");
            if ok {
                println!(
                    "Course ready. Start it with: tutor init {} <your run command>",
                    dir.display()
                );
                return Ok(());
            }
            prompt = format!(
                "This folder holds a coding course for the goal \"{goal}\", but it failed verification. Fix challenge.toml or reference/ so every test passes, without making the course easier.\n\n{report}"
            );
        }
        Err(format!(
            "Claude couldn't produce a passing course; inspect {} by hand",
            dir.display()
        ))
    })();

    if result.is_err() && !existed && !dir.join("challenge.toml").exists() {
        let _ = fs::remove_dir_all(&dir);
    }
    result
}

fn verify(dir: &Path) -> Result<(String, bool), String> {
    let c = read_challenge(dir)?;
    let run = c
        .reference
        .as_deref()
        .ok_or("challenge.toml has no `reference` command")?;
    Ok(run_stages(run, dir, &c.stages))
}

fn init(dir: &str, build: Option<String>, run: &str) -> Result<(), String> {
    let challenge = fs::canonicalize(dir).map_err(|e| format!("{dir}: {e}"))?;
    let c = read_challenge(&challenge)?;
    save(&Progress {
        challenge,
        run: run.into(),
        build,
        stage: 0,
        fails: 0,
        last: None,
    })?;
    install_hook()?;
    println!(
        "Started \"{}\". Commit your work (or run `tutor test`) to check it.\n",
        c.title
    );
    show(&c, 0);
    Ok(())
}

fn install_hook() -> Result<(), String> {
    let hook = Path::new(".git/hooks/post-commit");
    if !Path::new(".git").is_dir() {
        println!("Not a git repo, so there is no commit hook. Run `git init` and `tutor init` again to test on every commit.");
        return Ok(());
    }
    if hook.exists() {
        println!(
            "{} already exists; add `tutor test` to it yourself.",
            hook.display()
        );
        return Ok(());
    }
    // git runs hooks with sh, even on Windows, so use forward slashes.
    let exe = env::current_exe()
        .map_err(|e| e.to_string())?
        .display()
        .to_string()
        .replace('\\', "/");
    fs::write(hook, format!("#!/bin/sh\n\"{exe}\" test\n")).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(hook, fs::Permissions::from_mode(0o755)).map_err(|e| e.to_string())?;
    }
    println!("Installed git post-commit hook: tests run on every commit.");
    Ok(())
}

fn show(c: &Challenge, stage: usize) {
    match c.stages.get(stage) {
        Some(s) => println!(
            "Stage {}/{}: {}\n\n{}",
            stage + 1,
            c.stages.len(),
            s.name,
            s.instructions.trim()
        ),
        None => println!("You finished \"{}\"!", c.title),
    }
}

fn test() -> Result<(), String> {
    let (mut p, c) = load()?;
    if p.stage >= c.stages.len() {
        show(&c, p.stage);
        return Ok(());
    }
    let r = check(&mut p, &c)?;
    println!("{}", r.report);
    if let Some(v) = &r.review {
        println!("Reviewer:\n{v}\n");
    }
    if let Some(h) = &r.hint {
        println!("Claude Code:\n{h}\n");
    }
    if !r.ok {
        return Err("Not yet. Fix the failures above and commit again.".into());
    }
    println!("Stage passed!\n");
    show(&c, p.stage);
    Ok(())
}

/// One full check of the current stage: build, tests, local review, escalation. Shared by the CLI, the hook and the web page.
fn check(p: &mut Progress, c: &Challenge) -> Result<LastRun, String> {
    let stage = c.stages.get(p.stage).ok_or("course already finished")?;
    verbose!(
        "checking stage {}/{}: {} (failed runs so far: {})",
        p.stage + 1,
        c.stages.len(),
        stage.name,
        p.fails
    );
    let (report, ok) = build_and_run(p, c);
    let review = review(stage, &report);
    let mut hint = None;
    let at = p.stage;
    if ok {
        p.stage += 1;
        p.fails = 0;
    } else {
        p.fails += 1;
        if p.fails >= ESCALATE_AFTER {
            p.fails = 0; // ask Claude again only after another run of failures
            println!("Stuck for {ESCALATE_AFTER} attempts; asking Claude Code...\n");
            let h = escalate(stage, &report, review.as_deref())
                .unwrap_or_else(|e| format!("(Claude Code failed: {e})"));
            hint = Some(h);
        }
    }
    let last = LastRun {
        stage: at,
        ok,
        report,
        review,
        hint,
    };
    p.last = Some(last.clone());
    save(p)?;
    Ok(last)
}

fn current_hint() -> Result<String, String> {
    let (p, c) = load()?;
    let stage = c.stages.get(p.stage).ok_or("course already finished")?;
    let (report, _) = build_and_run(&p, &c);
    let review = p
        .last
        .as_ref()
        .filter(|l| l.stage == p.stage)
        .and_then(|l| l.review.as_deref());
    escalate(stage, &report, review)
}

fn build_and_run(p: &Progress, c: &Challenge) -> (String, bool) {
    if let Some(cmd) = &p.build {
        verbose!("build: $ {cmd}");
        let started = Instant::now();
        let parts = match cli::split_command(cmd) {
            Ok(parts) => parts,
            Err(e) => return (format!("  BUILD invalid command `{cmd}`: {e}"), false),
        };
        let out = match parts.split_first() {
            Some((prog, args)) => Command::new(prog).args(args).output(),
            None => return ("  BUILD: empty build command".into(), false),
        };
        match out {
            Err(e) => return (format!("  BUILD could not start `{cmd}`: {e}"), false),
            Ok(o) => {
                let log = format!(
                    "{}{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                );
                verbose!(
                    "build: {} in {:.1?}{}",
                    o.status,
                    started.elapsed(),
                    if log.trim().is_empty() {
                        String::new()
                    } else {
                        format!("\n{}", log.trim_end())
                    }
                );
                if !o.status.success() {
                    return (format!("  BUILD FAILED `{cmd}`\n{}", log.trim_end()), false);
                }
            }
        }
    }
    // Earlier stages are re-run so later work can't silently break them.
    run_stages(&p.run, Path::new("."), &c.stages[..=p.stage])
}

fn run_stages(run: &str, cwd: &Path, stages: &[Stage]) -> (String, bool) {
    let mut report = String::new();
    let mut ok = true;
    for (i, s) in stages.iter().enumerate() {
        for t in &s.tests {
            match run_test(run, cwd, t) {
                Ok(()) => report += &format!("  pass  stage {}: {} {:?}\n", i + 1, s.name, t.args),
                Err(e) => {
                    report += &format!("  FAIL  stage {}: {} {:?}\n{e}\n", i + 1, s.name, t.args);
                    ok = false;
                }
            }
        }
    }
    (report, ok)
}

fn run_test(run: &str, cwd: &Path, t: &Test) -> Result<(), String> {
    let parts = cli::split_command(run).map_err(|e| format!("    invalid run command: {e}"))?;
    let (prog, base_args) = parts
        .split_first()
        .ok_or_else(|| "empty run command".to_string())?;
    verbose!("test: $ {run} {}  stdin={:?}", t.args.join(" "), t.stdin);
    let started = Instant::now();
    let mut child = Command::new(prog)
        .args(base_args)
        .args(&t.args)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("    could not start `{run}`: {e}"))?;
    // Ignore write errors: a program that never reads stdin closes the pipe early.
    let _ = child.stdin.take().unwrap().write_all(t.stdin.as_bytes());
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let got = String::from_utf8_lossy(&out.stdout);
    verbose!(
        "      {} in {:.1?}  stdout={:?}  expected={:?}  stderr={:?}",
        out.status,
        started.elapsed(),
        normalize(&got),
        normalize(&t.expect),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    if out.status.success() && matches(&got, &t.expect) {
        return Ok(());
    }
    let status_line = if out.status.success() {
        String::new()
    } else {
        format!("    status:   {}\n", out.status)
    };
    Err(format!(
        "{status_line}    stdin:    {:?}\n    expected: {:?}\n    got:      {:?}\n    stderr:   {}",
        t.stdin,
        normalize(&t.expect),
        normalize(&got),
        String::from_utf8_lossy(&out.stderr).trim()
    ))
}

fn normalize(s: &str) -> String {
    s.replace("\r\n", "\n").trim_end().to_string()
}

fn matches(got: &str, expect: &str) -> bool {
    normalize(got) == normalize(expect)
}

fn last_commit_diff() -> String {
    let out = Command::new("git")
        .args(["show", "HEAD", "--format="])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .chars()
            .take(12_000)
            .collect(),
        _ => "(no git commit to show)".into(),
    }
}

/// Local small-model review via Ollama. Advisory only: tests alone decide pass/fail.
fn review(stage: &Stage, report: &str) -> Option<String> {
    let url = env::var("TUTOR_LLM").unwrap_or("http://localhost:11434".into());
    if url == "off" {
        return None;
    }
    let model = env::var("TUTOR_MODEL").unwrap_or("qwen3.6:35b".into());
    let diff = last_commit_diff();
    verbose!(
        "review: asking {model} at {url} (last commit diff: {} chars)",
        diff.len()
    );
    let started = Instant::now();
    let prompt = format!(
        "You review code for a learner doing one stage of a coding course.\n\nStage: {}\n{}\n\nTest results:\n{report}\n\nTheir latest commit:\n{}\n\nIn at most 3 short bullet points, point out bugs or design problems relevant to this stage. If tests fail, nudge toward the cause without writing the fix. Never write solution code.",
        stage.name,
        stage.instructions.trim(),
        diff
    );
    #[derive(Deserialize)]
    struct Reply {
        response: String,
    }
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout(Duration::from_secs(180))
        .build();
    let reply: Result<Reply, String> = agent
        .post(&format!("{url}/api/generate"))
        // A small context keeps more of the model on the GPU; the prompt is capped well under 8k tokens.
        .send_json(json!({ "model": model, "prompt": prompt, "stream": false, "think": false, "options": { "num_ctx": 8192 } }))
        .map_err(|e| e.to_string())
        .and_then(|r| r.into_json().map_err(|e| e.to_string()));
    verbose!(
        "review: {} after {:.1?}",
        if reply.is_ok() { "replied" } else { "failed" },
        started.elapsed()
    );
    match reply {
        Ok(r) => Some(r.response.trim().into()),
        Err(e) => {
            println!("(local review skipped: {e})\n");
            None
        }
    }
}

fn escalate(stage: &Stage, report: &str, review: Option<&str>) -> Result<String, String> {
    let prompt = format!(
        "A learner is stuck on a stage of a codecrafters-style course. Their code is in the current folder; read it.\n\nStage: {}\n{}\n\nTest results:\n{report}\n\nA small local reviewer said:\n{}\n\nExplain the concept they are missing and point to where in their code the problem is. Do not write the fix for them; if a snippet helps, keep it under 5 lines and about the concept, not their code. Be brief.",
        stage.name,
        stage.instructions.trim(),
        review.unwrap_or("(no review)")
    );
    claude(&prompt, Path::new("."), &[])
}

/// Runs Claude Code headless with the prompt on stdin (avoids Windows command-line length limits).
fn claude(prompt: &str, cwd: &Path, args: &[&str]) -> Result<String, String> {
    verbose!(
        "claude: $ claude -p {} (in {}, prompt {} chars)",
        args.join(" "),
        cwd.display(),
        prompt.len()
    );
    let started = Instant::now();
    let mut child = Command::new("claude")
        .arg("-p")
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start Claude Code (`claude`): {e}"))?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(prompt.as_bytes())
        .map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    verbose!("claude: {} after {:.1?}", out.status, started.elapsed());
    if !out.status.success() {
        return Err(format!("Claude Code failed: {text}"));
    }
    Ok(text)
}

// shortcut: single-threaded, so a slow Claude hint blocks other requests; fine for one local learner.
fn serve(port: u16) -> Result<(), String> {
    load()?;
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    println!("Dashboard: http://localhost:{port}  (Ctrl+C to stop)");
    for stream in listener.incoming().flatten() {
        if let Err(e) = handle(stream) {
            eprintln!("request failed: {e}");
        }
    }
    Ok(())
}

fn handle(mut s: TcpStream) -> std::io::Result<()> {
    let mut buf = vec![0; 8192];
    let n = s.read(&mut buf)?;
    let req = String::from_utf8_lossy(&buf[..n]);
    let mut lines = req.lines();
    let mut start = lines.next().unwrap_or("").split_whitespace();
    let (method, path) = (start.next().unwrap_or(""), start.next().unwrap_or(""));
    let headers: Vec<String> = lines
        .take_while(|l| !l.is_empty())
        .map(|l| l.to_ascii_lowercase())
        .collect();
    let header = |name: &str| {
        headers
            .iter()
            .find_map(|l| l.strip_prefix(name))
            .map(str::trim)
    };
    // A localhost Host header blocks DNS rebinding; the custom header on POSTs forces a CORS preflight other sites can't pass.
    let host = header("host:").map(|h| h.rsplit_once(':').map_or(h, |(name, _)| name));
    let local = matches!(host, Some("localhost" | "127.0.0.1"));
    let from_page = header("x-tutor:").is_some();
    let (status, ctype, body): (&str, &str, String) = match (local, method, path) {
        (false, ..) => ("403 Forbidden", "text/plain", "forbidden".into()),
        (_, "GET", "/") => ("200 OK", "text/html", INDEX.into()),
        (_, "GET", "/api/state") => api(state()),
        (_, "POST", "/api/test") if from_page => api(web_test()),
        (_, "POST", "/api/hint") if from_page => api(current_hint().map(|h| json!({ "hint": h }))),
        _ => ("404 Not Found", "text/plain", "not found".into()),
    };
    write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn api(r: Result<Value, String>) -> (&'static str, &'static str, String) {
    match r {
        Ok(v) => ("200 OK", "application/json", v.to_string()),
        Err(e) => (
            "500 Internal Server Error",
            "application/json",
            json!({ "error": e }).to_string(),
        ),
    }
}

fn state() -> Result<Value, String> {
    let (p, c) = load()?;
    // Stages ahead of the learner stay locked: name only.
    let stages: Vec<Value> = c
        .stages
        .iter()
        .enumerate()
        .map(|(i, s)| json!({ "name": s.name, "instructions": (i <= p.stage).then_some(&s.instructions) }))
        .collect();
    Ok(
        json!({ "title": c.title, "stage": p.stage, "fails": p.fails, "escalate_after": ESCALATE_AFTER, "stages": stages, "last": p.last }),
    )
}

fn web_test() -> Result<Value, String> {
    let (mut p, c) = load()?;
    check(&mut p, &c)?;
    state()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_compare_ignores_crlf_and_trailing_space() {
        assert!(matches("3\r\n", "3"));
        assert!(matches("a\r\nb\n\n", "a\nb"));
        assert!(!matches("3", "4"));
    }

    #[test]
    fn run_test_fails_on_nonzero_exit_even_when_stdout_matches() {
        let test = Test {
            args: vec!["-c".into(), "import sys; print('ok'); sys.exit(1)".into()],
            stdin: String::new(),
            expect: "ok".into(),
        };
        let err = run_test("python", Path::new("."), &test)
            .expect_err("non-zero exit code must fail even when stdout matches");
        assert!(err.contains("status:"), "missing status in error:\n{err}");

        let empty_expect = Test {
            args: vec!["-c".into(), "import sys; sys.exit(2)".into()],
            stdin: String::new(),
            expect: String::new(),
        };
        let err2 = run_test("python", Path::new("."), &empty_expect)
            .expect_err("crashing program with empty expected output must fail");
        assert!(err2.contains("status:"), "missing status in error:\n{err2}");
    }

    #[test]
    fn run_test_handles_quoted_arguments_with_spaces() {
        let test = Test {
            args: vec!["arg with spaces".into()],
            stdin: String::new(),
            expect: "hello world | arg with spaces".into(),
        };
        run_test(
            r#"python -c "import sys; print('hello world |', sys.argv[1])""#,
            Path::new("."),
            &test,
        )
        .expect("quoted run command with spaces should execute cleanly");
    }

    #[test]
    fn save_to_and_load_progress_round_trip_atomically() {
        let dir = env::temp_dir().join(format!(
            "tutor_atomic_test_{}_{}",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(PROGRESS);

        let initial = Progress {
            challenge: PathBuf::from("challenges/wc"),
            run: "python main.py".into(),
            build: Some("cargo build --release".into()),
            stage: 0,
            fails: 1,
            last: None,
        };
        save_to(&path, &initial).unwrap();
        assert_eq!(load_progress_from(&path).unwrap(), initial);

        let updated = Progress {
            challenge: PathBuf::from("challenges/wc"),
            run: r#""C:\Program Files\Python\python.exe" main.py"#.into(),
            build: None,
            stage: 2,
            fails: 0,
            last: Some(LastRun {
                stage: 1,
                ok: true,
                report: "  pass stage 2\n".into(),
                review: Some("Looks clean.".into()),
                hint: None,
            }),
        };
        save_to(&path, &updated).unwrap();
        assert_eq!(load_progress_from(&path).unwrap(), updated);

        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files should be cleaned up after atomic rename: {leftovers:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn new_in_rejects_traversal_and_cleans_up_empty_dir_on_failure() {
        let root = env::temp_dir().join(format!(
            "tutor_new_cleanup_test_{}_{}",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();

        // Path traversal must be rejected before creating any directory.
        assert!(new_in(&root, "../escape", "goal", |_, _, _| Ok(String::new())).is_err());

        // Failed agent on a fresh slug must remove the empty directory.
        let err = new_in(&root, "broken_course", "goal", |_, _, _| {
            Err("claude failed to start".into())
        })
        .unwrap_err();
        assert!(err.contains("claude failed to start"));
        assert!(
            !root.join("broken_course").exists(),
            "empty directory should be removed when new() fails before creating challenge.toml"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn bundled_challenges_pass_their_reference() {
        for entry in fs::read_dir("challenges").unwrap() {
            let dir = entry.unwrap().path();
            let c = read_challenge(&dir).unwrap();
            assert!(
                c.stages.iter().all(|s| !s.tests.is_empty()),
                "{}",
                dir.display()
            );
            let (report, ok) = verify(&dir).unwrap();
            assert!(ok, "{}:\n{report}", dir.display());
        }
    }
}
