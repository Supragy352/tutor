use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub const DEFAULT_TEST_TIMEOUT_SECS: u64 = 30;
pub const DEFAULT_BUILD_TIMEOUT_SECS: u64 = 300;
const MAX_TIMEOUT_SECS: f64 = 86_400.0;

#[derive(Debug)]
pub enum ExecError {
    Spawn(io::Error),
    Io(io::Error),
    TimedOut(Duration),
}

pub fn parse_timeout(raw: Option<&str>, default_secs: u64) -> Duration {
    match raw.and_then(|s| s.trim().parse::<f64>().ok()) {
        Some(secs) if secs.is_finite() && secs > 0.0 && secs <= MAX_TIMEOUT_SECS => {
            Duration::from_secs_f64(secs)
        }
        _ => Duration::from_secs(default_secs),
    }
}

pub fn test_timeout() -> Duration {
    parse_timeout(
        std::env::var("TUTOR_TIMEOUT").ok().as_deref(),
        DEFAULT_TEST_TIMEOUT_SECS,
    )
}

pub fn build_timeout() -> Duration {
    parse_timeout(
        std::env::var("TUTOR_BUILD_TIMEOUT").ok().as_deref(),
        DEFAULT_BUILD_TIMEOUT_SECS,
    )
}

pub fn format_timeout(timeout: Duration) -> String {
    if timeout.subsec_nanos() == 0 {
        format!("{}s", timeout.as_secs())
    } else {
        format!("{:.1}s", timeout.as_secs_f64())
    }
}

/// Runs a child process with concurrent stdin writing and stdout/stderr draining
/// to prevent OS pipe-buffer deadlocks, plus an optional execution timeout.
pub fn run_command(
    prog: &str,
    args: &[&str],
    cwd: &Path,
    stdin_data: Option<&[u8]>,
    capture_stderr: bool,
    timeout: Option<Duration>,
) -> Result<Output, ExecError> {
    let mut cmd = Command::new(prog);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(if stdin_data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(if capture_stderr {
            Stdio::piped()
        } else {
            Stdio::inherit()
        });

    let mut child = cmd.spawn().map_err(ExecError::Spawn)?;

    // Write stdin on a background thread and drop immediately to deliver EOF while stdout/stderr drain.
    let stdin_thread = match (child.stdin.take(), stdin_data) {
        (Some(mut pipe), Some(bytes)) => {
            let input = bytes.to_vec();
            Some(thread::spawn(move || {
                // Ignore write errors (e.g. BrokenPipe if the child exits without reading stdin).
                let _ = pipe.write_all(&input);
                drop(pipe);
            }))
        }
        _ => None,
    };

    let mut stdout_pipe = child.stdout.take().expect("stdout is piped");
    let stdout_thread = thread::spawn(move || -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        stdout_pipe.read_to_end(&mut buf)?;
        Ok(buf)
    });

    let stderr_thread = child.stderr.take().map(|mut stderr_pipe| {
        thread::spawn(move || -> io::Result<Vec<u8>> {
            let mut buf = Vec::new();
            stderr_pipe.read_to_end(&mut buf)?;
            Ok(buf)
        })
    });

    let status = match timeout {
        None => child.wait().map_err(ExecError::Io)?,
        Some(limit) => {
            let deadline = Instant::now() + limit;
            let mut poll = Duration::from_millis(2);
            loop {
                match child.try_wait().map_err(ExecError::Io)? {
                    Some(status) => break status,
                    None => {
                        let now = Instant::now();
                        if now >= deadline {
                            let _ = child.kill();
                            let _ = child.wait();
                            return Err(ExecError::TimedOut(limit));
                        }
                        let remaining = deadline.saturating_duration_since(now);
                        thread::sleep(poll.min(remaining));
                        poll = (poll * 2).min(Duration::from_millis(20));
                    }
                }
            }
        }
    };

    if let Some(handle) = stdin_thread {
        let _ = handle.join();
    }

    let stdout = stdout_thread
        .join()
        .map_err(|_| ExecError::Io(io::Error::other("stdout reader thread panicked")))?
        .map_err(ExecError::Io)?;

    let stderr = match stderr_thread {
        Some(handle) => handle
            .join()
            .map_err(|_| ExecError::Io(io::Error::other("stderr reader thread panicked")))?
            .map_err(ExecError::Io)?,
        None => Vec::new(),
    };

    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_timeout_handles_valid_and_invalid_values() {
        assert_eq!(parse_timeout(None, 30), Duration::from_secs(30));
        assert_eq!(parse_timeout(Some("45"), 30), Duration::from_secs(45));
        assert_eq!(
            parse_timeout(Some(" 1.5 "), 30),
            Duration::from_millis(1500)
        );
        assert_eq!(parse_timeout(Some("0"), 30), Duration::from_secs(30));
        assert_eq!(parse_timeout(Some("-10"), 30), Duration::from_secs(30));
        assert_eq!(parse_timeout(Some("abc"), 30), Duration::from_secs(30));
        assert_eq!(parse_timeout(Some("NaN"), 30), Duration::from_secs(30));
        assert_eq!(parse_timeout(Some("inf"), 30), Duration::from_secs(30));
        assert_eq!(format_timeout(Duration::from_secs(30)), "30s");
        assert_eq!(format_timeout(Duration::from_millis(250)), "0.2s");
    }

    #[test]
    fn large_stdin_stdout_echo_does_not_deadlock() {
        // 256 KB exceeds both Windows (~4 KB) and Linux (~64 KB) OS pipe buffers.
        let payload = "0123456789abcdef\n".repeat(16_384);
        let script = "import sys\nwhile True:\n    chunk = sys.stdin.buffer.read(1024)\n    if not chunk:\n        break\n    sys.stdout.buffer.write(chunk)\n    sys.stdout.buffer.flush()\n";
        let out = run_command(
            "python",
            &["-c", script],
            Path::new("."),
            Some(payload.as_bytes()),
            true,
            Some(Duration::from_secs(10)),
        )
        .expect("large echo should complete without deadlocking");
        assert!(out.status.success());
        assert_eq!(out.stdout, payload.as_bytes());
    }

    #[test]
    fn large_stdout_and_stderr_concurrent_does_not_deadlock() {
        let script = "import sys\nchunk = b'x' * 4096\nfor _ in range(32):\n    sys.stdout.buffer.write(chunk)\n    sys.stdout.buffer.flush()\n    sys.stderr.buffer.write(chunk)\n    sys.stderr.buffer.flush()\n";
        let out = run_command(
            "python",
            &["-c", script],
            Path::new("."),
            Some(b""),
            true,
            Some(Duration::from_secs(10)),
        )
        .expect("concurrent stdout/stderr should drain without deadlocking");
        assert!(out.status.success());
        assert_eq!(out.stdout.len(), 32 * 4096);
        assert_eq!(out.stderr.len(), 32 * 4096);
    }

    #[test]
    fn early_exit_ignoring_large_stdin_does_not_fail() {
        let payload = vec![b'a'; 256 * 1024];
        let out = run_command(
            "python",
            &["-c", "print('done')"],
            Path::new("."),
            Some(&payload),
            true,
            Some(Duration::from_secs(10)),
        )
        .expect("child exiting early without reading stdin should still succeed");
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "done");
    }

    #[test]
    fn hung_child_process_times_out_and_is_killed() {
        let started = Instant::now();
        let err = run_command(
            "python",
            &["-c", "import time; time.sleep(15)"],
            Path::new("."),
            Some(b""),
            true,
            Some(Duration::from_millis(200)),
        )
        .expect_err("sleeping child should time out");
        match err {
            ExecError::TimedOut(d) => assert_eq!(d, Duration::from_millis(200)),
            other => panic!("expected TimedOut, got {other:?}"),
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timeout took too long: {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn run_test_reports_timeout_message() {
        let test = crate::Test {
            args: vec!["-c".into(), "import time; time.sleep(15)".into()],
            stdin: String::new(),
            expect: String::new(),
        };
        let err =
            crate::run_test_with_timeout("python", Path::new("."), &test, Duration::from_secs(1))
                .expect_err("timed-out test should fail");
        assert_eq!(err, "    timed out after 1s");
    }

    #[test]
    fn run_test_handles_large_stdin_stdout_echo() {
        let payload = "hello world 1234567890\n".repeat(4096);
        let test = crate::Test {
            args: vec![
                "-c".into(),
                "import sys; sys.stdout.write(sys.stdin.read())".into(),
            ],
            stdin: payload.clone(),
            expect: payload,
        };
        crate::run_test_with_timeout("python", Path::new("."), &test, Duration::from_secs(10))
            .expect("large stdin/stdout echo should pass run_test");
    }
}
