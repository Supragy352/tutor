use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::{env, fs};

static VERBOSE: AtomicBool = AtomicBool::new(false);

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
const USAGE: &str = "usage:
  tutor new <slug> <goal...>                    have Claude Code design a course into challenges/<slug>
  tutor verify <challenge-dir>                  check a course's tests against its reference solution
  tutor init <challenge-dir> [--build \"<cmd>\"] <run command...>
                                                start a course in the current folder
  tutor show                                    print the current stage
  tutor test                                    build, run tests + local review; advance on pass
  tutor hint                                    ask Claude Code for help on the current stage
  tutor serve [port]                            open the web dashboard (default port 4242)

-v / --verbose right after the command (e.g. `tutor test -v`) logs each step live; TUTOR_VERBOSE=1 does the same (e.g. for the git hook).
env: TUTOR_LLM (Ollama URL, default http://localhost:11434, or \"off\"), TUTOR_MODEL (default qwen3.6:35b)";

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

#[derive(Serialize, Deserialize, Clone)]
struct LastRun {
    stage: usize,
    ok: bool,
    report: String,
    review: Option<String>,
    hint: Option<String>,
}

#[derive(Serialize, Deserialize)]
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
    let mut args: Vec<String> = env::args().skip(1).collect();
    // Only right after the command, so a `-v` inside a run command passed to `init` is left alone.
    let flag = args.get(1).is_some_and(|a| a == "-v" || a == "--verbose");
    if flag {
        args.remove(1);
    }
    VERBOSE.store(
        flag || env::var_os("TUTOR_VERBOSE").is_some(),
        Ordering::Relaxed,
    );
    let result = match args.first().map(String::as_str) {
        Some("new") if args.len() >= 3 => new(&args[1], &args[2..].join(" ")),
        Some("verify") if args.len() == 2 => {
            verify(Path::new(&args[1])).and_then(|(report, ok)| {
                println!("{report}");
                if ok {
                    Ok(())
                } else {
                    Err("reference solution fails".into())
                }
            })
        }
        Some("init") if args.len() >= 3 => {
            let (build, run) = match args[2].as_str() {
                "--build" if args.len() >= 5 => (Some(args[3].clone()), &args[4..]),
                _ => (None, &args[2..]),
            };
            init(&args[1], build, &run.join(" "))
        }
        Some("show") => load().map(|(p, c)| show(&c, p.stage)),
        Some("test") => test(),
        Some("hint") => current_hint().map(|h| println!("{h}")),
        Some("serve") => serve(args.get(1).map_or("4242", String::as_str)),
        _ => Err(USAGE.into()),
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

fn load_from(path: &Path) -> Result<(Progress, Challenge), String> {
    let text = fs::read_to_string(path)
        .map_err(|_| format!("no {} here; run `tutor init` first", path.display()))?;
    let p: Progress = toml::from_str(&text).map_err(|e| e.to_string())?;
    let c = read_challenge(&p.challenge)?;
    Ok((p, c))
}

fn load() -> Result<(Progress, Challenge), String> {
    load_from(Path::new(PROGRESS))
}

fn save_to(path: &Path, p: &Progress) -> Result<(), String> {
    fs::write(path, toml::to_string(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn save(p: &Progress) -> Result<(), String> {
    save_to(Path::new(PROGRESS), p)
}

fn new(slug: &str, goal: &str) -> Result<(), String> {
    let dir = Path::new("challenges").join(slug);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
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
        claude(&prompt, &dir, &["--permission-mode", "acceptEdits"])?;
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
    check_with(Path::new(PROGRESS), p, c, build_and_run, review, escalate)
}

fn check_with<R, V, E>(
    progress_path: &Path,
    p: &mut Progress,
    c: &Challenge,
    run_fn: R,
    review_fn: V,
    escalate_fn: E,
) -> Result<LastRun, String>
where
    R: FnOnce(&Progress, &Challenge) -> (String, bool),
    V: FnOnce(&Stage, &str) -> Option<String>,
    E: FnOnce(&Stage, &str, Option<&str>) -> Result<String, String>,
{
    let stage = c.stages.get(p.stage).ok_or("course already finished")?;
    verbose!(
        "checking stage {}/{}: {} (failed runs so far: {})",
        p.stage + 1,
        c.stages.len(),
        stage.name,
        p.fails
    );
    let (report, ok) = run_fn(p, c);
    let review = review_fn(stage, &report);
    let mut hint = None;
    let at = p.stage;
    if ok {
        p.stage += 1;
        p.fails = 0;
    } else {
        p.fails += 1;
        if p.fails >= ESCALATE_AFTER {
            println!("Stuck for {ESCALATE_AFTER} attempts; asking Claude Code...\n");
            match escalate_fn(stage, &report, review.as_deref()) {
                Ok(h) => {
                    p.fails = 0; // ask Claude again only after another run of failures
                    hint = Some(h);
                }
                Err(e) => {
                    println!("(Claude Code skipped: {e})\n");
                }
            }
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
    save_to(progress_path, p)?;
    Ok(last)
}

fn current_hint() -> Result<String, String> {
    current_hint_with(Path::new(PROGRESS), build_and_run, escalate)
}

fn current_hint_with<R, E>(
    progress_path: &Path,
    run_fn: R,
    escalate_fn: E,
) -> Result<String, String>
where
    R: FnOnce(&Progress, &Challenge) -> (String, bool),
    E: FnOnce(&Stage, &str, Option<&str>) -> Result<String, String>,
{
    let (mut p, c) = load_from(progress_path)?;
    let stage = c.stages.get(p.stage).ok_or("course already finished")?;
    let (report, ok) = run_fn(&p, &c);
    let review = p
        .last
        .as_ref()
        .filter(|l| l.stage == p.stage)
        .and_then(|l| l.review.as_deref());
    let h = escalate_fn(stage, &report, review)?;
    match p.last.as_mut().filter(|l| l.stage == p.stage) {
        Some(last) => {
            last.ok = ok;
            last.report = report;
            last.hint = Some(h.clone());
        }
        None => {
            p.last = Some(LastRun {
                stage: p.stage,
                ok,
                report,
                review: None,
                hint: Some(h.clone()),
            });
        }
    }
    save_to(progress_path, &p)?;
    Ok(h)
}

fn build_and_run(p: &Progress, c: &Challenge) -> (String, bool) {
    if let Some(cmd) = &p.build {
        verbose!("build: $ {cmd}");
        let started = Instant::now();
        let mut parts = cmd.split_whitespace();
        let out = match parts.next() {
            Some(prog) => Command::new(prog).args(parts).output(),
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

// shortcut: splits the run command on whitespace (no quoting) and has no timeout; upgrade if challenges need either.
fn run_test(run: &str, cwd: &Path, t: &Test) -> Result<(), String> {
    let mut parts = run.split_whitespace();
    let prog = parts.next().ok_or("empty run command")?;
    verbose!("test: $ {run} {}  stdin={:?}", t.args.join(" "), t.stdin);
    let started = Instant::now();
    let mut child = Command::new(prog)
        .args(parts)
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
    if matches(&got, &t.expect) {
        return Ok(());
    }
    Err(format!(
        "    stdin:    {:?}\n    expected: {:?}\n    got:      {:?}\n    stderr:   {}",
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
fn serve(port: &str) -> Result<(), String> {
    load()?;
    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).map_err(|e| e.to_string())?;
    println!("Dashboard: http://localhost:{port}  (Ctrl+C to stop)");
    for stream in listener.incoming().flatten() {
        if let Err(e) = handle(stream) {
            eprintln!("request failed: {e}");
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum Route {
    Forbidden,
    Index,
    ApiState,
    ApiTest,
    ApiHint,
    NotFound,
}

fn has_header_end(buf: &[u8]) -> bool {
    buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.windows(2).any(|w| w == b"\n\n")
}

fn read_http_head(r: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut buf = [0u8; 8192];
    let mut len = 0;
    while len < buf.len() {
        match r.read(&mut buf[len..]) {
            Ok(0) => break,
            Ok(n) => {
                len += n;
                if has_header_end(&buf[..len]) {
                    break;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(buf[..len].to_vec())
}

fn parse_route(req: &str) -> Route {
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
    // A localhost Host header blocks DNS rebinding; the custom header on API routes forces a CORS preflight other sites can't pass.
    let host = header("host:").map(|h| h.rsplit_once(':').map_or(h, |(name, _)| name));
    let local = matches!(host, Some("localhost" | "127.0.0.1"));
    let from_page = header("x-tutor:").is_some();
    match (local, method, path) {
        (false, ..) => Route::Forbidden,
        (_, "GET", "/") => Route::Index,
        (_, "GET", "/api/state") if from_page => Route::ApiState,
        (_, "POST", "/api/test") if from_page => Route::ApiTest,
        (_, "POST", "/api/hint") if from_page => Route::ApiHint,
        _ => Route::NotFound,
    }
}

fn write_response(
    w: &mut impl Write,
    status: &str,
    ctype: &str,
    body: &str,
) -> std::io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {status}\r\n\
         Content-Type: {ctype}; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         X-Frame-Options: DENY\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    )
}

fn handle(mut s: TcpStream) -> std::io::Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.set_write_timeout(Some(Duration::from_secs(30)))?;
    let buf = read_http_head(&mut s)?;
    if buf.is_empty() {
        return Ok(());
    }
    let req = String::from_utf8_lossy(&buf);
    let (status, ctype, body): (&str, &str, String) = match parse_route(&req) {
        Route::Forbidden => ("403 Forbidden", "text/plain", "forbidden".into()),
        Route::Index => ("200 OK", "text/html", INDEX.into()),
        Route::ApiState => api(state()),
        Route::ApiTest => api(web_test()),
        Route::ApiHint => api(current_hint().map(|h| json!({ "hint": h }))),
        Route::NotFound => ("404 Not Found", "text/plain", "not found".into()),
    };
    write_response(&mut s, status, ctype, &body)
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

fn state_value(p: &Progress, c: &Challenge) -> Value {
    // Stages ahead of the learner stay locked: name only.
    let stages: Vec<Value> = c
        .stages
        .iter()
        .enumerate()
        .map(|(i, s)| json!({ "name": s.name, "instructions": (i <= p.stage).then_some(&s.instructions) }))
        .collect();
    json!({
        "title": c.title,
        "stage": p.stage,
        "fails": p.fails,
        "escalate_after": ESCALATE_AFTER,
        "stages": stages,
        "last": p.last
    })
}

fn state_at(path: &Path) -> Result<Value, String> {
    let (p, c) = load_from(path)?;
    Ok(state_value(&p, &c))
}

fn state() -> Result<Value, String> {
    state_at(Path::new(PROGRESS))
}

fn web_test_at(path: &Path) -> Result<Value, String> {
    let (mut p, c) = load_from(path)?;
    if p.stage < c.stages.len() {
        check_with(path, &mut p, &c, build_and_run, review, escalate)?;
    }
    state_at(path)
}

fn web_test() -> Result<Value, String> {
    web_test_at(Path::new(PROGRESS))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct ChunkedReader<'a> {
        data: &'a [u8],
        pos: usize,
        chunk_size: usize,
    }

    impl Read for ChunkedReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let n = (self.data.len() - self.pos)
                .min(self.chunk_size)
                .min(buf.len());
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    struct TempProgress(PathBuf);

    impl TempProgress {
        fn new(p: &Progress) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!("tutor-test-{}-{id}.toml", std::process::id()));
            save_to(&path, p).unwrap();
            Self(path)
        }
    }

    impl Drop for TempProgress {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn wc_challenge_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("challenges/wc")
    }

    fn sample_progress(stage: usize, fails: usize, last: Option<LastRun>) -> Progress {
        Progress {
            challenge: wc_challenge_dir(),
            run: "echo unused".into(),
            build: None,
            stage,
            fails,
            last,
        }
    }

    fn roundtrip_handle(chunks: &[&[u8]]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle(stream).unwrap();
        });
        let mut client = TcpStream::connect(addr).unwrap();
        for chunk in chunks {
            client.write_all(chunk).unwrap();
            client.flush().unwrap();
        }
        let mut resp = String::new();
        client.read_to_string(&mut resp).unwrap();
        server.join().unwrap();
        resp
    }

    #[test]
    fn output_compare_ignores_crlf_and_trailing_space() {
        assert!(matches("3\r\n", "3"));
        assert!(matches("a\r\nb\n\n", "a\nb"));
        assert!(!matches("3", "4"));
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

    #[test]
    fn http_host_header_validation() {
        assert_eq!(
            parse_route("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"),
            Route::Index
        );
        assert_eq!(
            parse_route("GET / HTTP/1.1\r\nHost: localhost:4242\r\n\r\n"),
            Route::Index
        );
        assert_eq!(
            parse_route("GET / HTTP/1.1\r\nHost: 127.0.0.1:4242\r\n\r\n"),
            Route::Index
        );
        assert_eq!(
            parse_route("GET / HTTP/1.1\r\nHOST: LOCALHOST:4242\r\n\r\n"),
            Route::Index
        );
        assert_eq!(
            parse_route("GET / HTTP/1.1\r\nHost: evil.example\r\n\r\n"),
            Route::Forbidden
        );
        assert_eq!(
            parse_route("GET / HTTP/1.1\r\nHost: localhost.evil.example:4242\r\n\r\n"),
            Route::Forbidden
        );
        assert_eq!(parse_route("GET / HTTP/1.1\r\n\r\n"), Route::Forbidden);
    }

    #[test]
    fn http_api_routes_require_x_tutor_header() {
        assert_eq!(
            parse_route("GET /api/state HTTP/1.1\r\nHost: localhost:4242\r\n\r\n"),
            Route::NotFound
        );
        assert_eq!(
            parse_route("GET /api/state HTTP/1.1\r\nHost: localhost:4242\r\nX-Tutor: 1\r\n\r\n"),
            Route::ApiState
        );
        assert_eq!(
            parse_route("POST /api/test HTTP/1.1\r\nHost: localhost:4242\r\n\r\n"),
            Route::NotFound
        );
        assert_eq!(
            parse_route("POST /api/test HTTP/1.1\r\nHost: localhost:4242\r\nx-tutor: 1\r\n\r\n"),
            Route::ApiTest
        );
        assert_eq!(
            parse_route("POST /api/hint HTTP/1.1\r\nHost: localhost:4242\r\n\r\n"),
            Route::NotFound
        );
        assert_eq!(
            parse_route("POST /api/hint HTTP/1.1\r\nHost: localhost:4242\r\nX-Tutor: 1\r\n\r\n"),
            Route::ApiHint
        );
    }

    #[test]
    fn read_http_head_handles_chunked_lf_eof_and_cap() {
        let mut empty = ChunkedReader {
            data: b"",
            pos: 0,
            chunk_size: 4,
        };
        assert!(read_http_head(&mut empty).unwrap().is_empty());

        let raw = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\ntrailing";
        let mut chunked = ChunkedReader {
            data: raw,
            pos: 0,
            chunk_size: 5,
        };
        let got = read_http_head(&mut chunked).unwrap();
        assert!(has_header_end(&got));
        assert!(got.len() < raw.len());

        let lf_only = b"GET / HTTP/1.1\nHost: localhost\n\nextra";
        let mut lf_reader = ChunkedReader {
            data: lf_only,
            pos: 0,
            chunk_size: 7,
        };
        let got_lf = read_http_head(&mut lf_reader).unwrap();
        assert!(has_header_end(&got_lf));

        let oversized = vec![b'A'; 10_000];
        let mut big_reader = ChunkedReader {
            data: &oversized,
            pos: 0,
            chunk_size: 512,
        };
        assert_eq!(read_http_head(&mut big_reader).unwrap().len(), 8192);
    }

    #[test]
    fn handle_returns_ok_on_closed_preconnect_socket() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle(stream)
        });
        let client = TcpStream::connect(addr).unwrap();
        drop(client);
        server.join().unwrap().unwrap();
    }

    #[test]
    fn handle_includes_security_headers_and_handles_split_request() {
        let resp = roundtrip_handle(&[b"GET / HTTP/1.1\r\n", b"Host: localhost:4242\r\n\r\n"]);
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(resp.contains("Cache-Control: no-store\r\n"));
        assert!(resp.contains("X-Content-Type-Options: nosniff\r\n"));
        assert!(resp.contains("X-Frame-Options: DENY\r\n"));

        let forbidden = roundtrip_handle(&[b"GET / HTTP/1.1\r\nHost: evil.example\r\n\r\n"]);
        assert!(forbidden.starts_with("HTTP/1.1 403 Forbidden\r\n"));
        assert!(forbidden.contains("Cache-Control: no-store\r\n"));

        let missing_header =
            roundtrip_handle(&[b"GET /api/state HTTP/1.1\r\nHost: localhost:4242\r\n\r\n"]);
        assert!(missing_header.starts_with("HTTP/1.1 404 Not Found\r\n"));
    }

    #[test]
    fn web_test_returns_state_for_completed_course() {
        let c = read_challenge(&wc_challenge_dir()).unwrap();
        let tmp = TempProgress::new(&sample_progress(c.stages.len(), 0, None));
        let state = web_test_at(&tmp.0).unwrap();
        assert_eq!(state["stage"], c.stages.len());
        assert_eq!(state["title"], c.title);
    }

    #[test]
    fn current_hint_persists_new_and_updates_matching_last_run() {
        // Case 1: no prior last run -> creates LastRun with hint and persists to tutor.toml
        let tmp = TempProgress::new(&sample_progress(0, 1, None));
        let h1 = current_hint_with(
            &tmp.0,
            |_, _| ("  FAIL stage 1\n".into(), false),
            |_, _, rev| {
                assert_eq!(rev, None);
                Ok("first hint".into())
            },
        )
        .unwrap();
        assert_eq!(h1, "first hint");
        let (p1, _) = load_from(&tmp.0).unwrap();
        let last1 = p1.last.unwrap();
        assert_eq!(last1.stage, 0);
        assert!(!last1.ok);
        assert_eq!(last1.report, "  FAIL stage 1\n");
        assert_eq!(last1.review, None);
        assert_eq!(last1.hint.as_deref(), Some("first hint"));

        // Case 2: matching stage with existing review -> preserves review, updates report/ok/hint
        let existing = LastRun {
            stage: 0,
            ok: false,
            report: "old report".into(),
            review: Some("keep this review".into()),
            hint: Some("old hint".into()),
        };
        let tmp2 = TempProgress::new(&sample_progress(0, 1, Some(existing)));
        let h2 = current_hint_with(
            &tmp2.0,
            |_, _| ("new report".into(), true),
            |_, _, rev| {
                assert_eq!(rev, Some("keep this review"));
                Ok("updated hint".into())
            },
        )
        .unwrap();
        assert_eq!(h2, "updated hint");
        let (p2, _) = load_from(&tmp2.0).unwrap();
        let last2 = p2.last.unwrap();
        assert_eq!(last2.stage, 0);
        assert!(last2.ok);
        assert_eq!(last2.report, "new report");
        assert_eq!(last2.review.as_deref(), Some("keep this review"));
        assert_eq!(last2.hint.as_deref(), Some("updated hint"));

        // Case 3: prior last run from earlier stage -> replaces with fresh LastRun for current stage
        let prev_stage = LastRun {
            stage: 0,
            ok: true,
            report: "stage 0 ok".into(),
            review: Some("old stage review".into()),
            hint: Some("old stage hint".into()),
        };
        let tmp3 = TempProgress::new(&sample_progress(1, 0, Some(prev_stage)));
        current_hint_with(
            &tmp3.0,
            |_, _| ("stage 1 fail".into(), false),
            |_, _, rev| {
                assert_eq!(rev, None);
                Ok("stage 1 hint".into())
            },
        )
        .unwrap();
        let (p3, _) = load_from(&tmp3.0).unwrap();
        let last3 = p3.last.unwrap();
        assert_eq!(last3.stage, 1);
        assert!(!last3.ok);
        assert_eq!(last3.report, "stage 1 fail");
        assert_eq!(last3.review, None);
        assert_eq!(last3.hint.as_deref(), Some("stage 1 hint"));

        // Case 4: escalate failure does not overwrite persisted last run
        let err = current_hint_with(
            &tmp3.0,
            |_, _| ("still failing".into(), false),
            |_, _, _| Err("claude unavailable".into()),
        )
        .unwrap_err();
        assert_eq!(err, "claude unavailable");
        let (p3_after, _) = load_from(&tmp3.0).unwrap();
        assert_eq!(p3_after.last.unwrap().hint.as_deref(), Some("stage 1 hint"));
    }

    #[test]
    fn check_escalation_failure_leaves_hint_none_and_preserves_fail_count() {
        let mut p = sample_progress(0, ESCALATE_AFTER - 1, None);
        let c = read_challenge(&p.challenge).unwrap();
        let tmp = TempProgress::new(&p);

        // Escalation fails: hint stays None and fails is not reset to 0
        let last = check_with(
            &tmp.0,
            &mut p,
            &c,
            |_, _| ("  FAIL stage 1\n".into(), false),
            |_, _| Some("review note".into()),
            |_, _, _| Err("claude not found".into()),
        )
        .unwrap();
        assert!(!last.ok);
        assert_eq!(last.hint, None);
        assert_eq!(p.fails, ESCALATE_AFTER);
        let (saved, _) = load_from(&tmp.0).unwrap();
        assert_eq!(saved.fails, ESCALATE_AFTER);
        assert_eq!(saved.last.unwrap().hint, None);

        // Subsequent escalation succeeds: hint is persisted and fails resets to 0
        let last2 = check_with(
            &tmp.0,
            &mut p,
            &c,
            |_, _| ("  FAIL stage 1\n".into(), false),
            |_, _| None,
            |_, _, _| Ok("actionable hint".into()),
        )
        .unwrap();
        assert_eq!(last2.hint.as_deref(), Some("actionable hint"));
        assert_eq!(p.fails, 0);
        let (saved2, _) = load_from(&tmp.0).unwrap();
        assert_eq!(saved2.fails, 0);
        assert_eq!(
            saved2.last.unwrap().hint.as_deref(),
            Some("actionable hint")
        );
    }
}
