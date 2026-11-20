use std::path::PathBuf;

pub const USAGE: &str = "usage:
  tutor new <slug> <goal...>                    have Claude Code design a course into challenges/<slug>
  tutor verify <challenge-dir>                  check a course's tests against its reference solution
  tutor init <challenge-dir> [--build \"<cmd>\"] <run command...>
                                                start a course in the current folder
  tutor show                                    print the current stage
  tutor test                                    build, run tests + local review; advance on pass
  tutor hint                                    ask Claude Code for help on the current stage
  tutor serve [port]                            open the web dashboard (default port 4242)

options:
  -v, --verbose                                 log each step live (before or after the command; TUTOR_VERBOSE=1 does the same)
  -h, --help                                    print this help text and exit
  -V, --version                                 print version and exit

env: TUTOR_LLM (Ollama URL, default http://localhost:11434, or \"off\"), TUTOR_MODEL (default qwen3.6:35b), TUTOR_TIMEOUT (per-test seconds, default 30)";

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    Help,
    Version,
    New {
        slug: String,
        goal: String,
    },
    Verify {
        dir: PathBuf,
    },
    Init {
        dir: String,
        build: Option<String>,
        run: String,
    },
    Show,
    Test,
    Hint,
    Serve {
        port: u16,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParsedArgs {
    pub verbose: bool,
    pub action: Action,
}

fn is_verbose_flag(s: &str) -> bool {
    s == "-v" || s == "--verbose"
}

fn is_help_flag(s: &str) -> bool {
    s == "-h" || s == "--help"
}

fn is_version_flag(s: &str) -> bool {
    s == "-V" || s == "--version"
}

/// Validates a course slug so it cannot traverse directories or look like a flag.
/// Must be non-empty, match `[a-zA-Z0-9_-]+`, and not start with `-`.
pub fn validate_slug(slug: &str) -> Result<(), String> {
    if slug.is_empty() {
        return Err("invalid slug: must not be empty".into());
    }
    if slug.starts_with('-') {
        return Err(format!("invalid slug `{slug}`: must not start with `-`"));
    }
    if !slug
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "invalid slug `{slug}`: use only ASCII letters, digits, `_`, and `-`"
        ));
    }
    Ok(())
}

/// Splits a command string into argv tokens while respecting single (`'...'`) and double (`"..."`) quotes.
/// Backslashes outside quotes are preserved literally (safe for Windows paths); inside double quotes,
/// `\"` unescapes to `"` and `\\` unescapes to `\`, while other `\x` sequences stay verbatim.
pub fn split_command(cmd: &str) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_token = false;
    let mut chars = cmd.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {
                if in_token {
                    args.push(std::mem::take(&mut current));
                    in_token = false;
                }
            }
            '\'' => {
                in_token = true;
                let mut closed = false;
                for sc in chars.by_ref() {
                    if sc == '\'' {
                        closed = true;
                        break;
                    }
                    current.push(sc);
                }
                if !closed {
                    return Err(format!("unterminated single quote in command `{cmd}`"));
                }
            }
            '"' => {
                in_token = true;
                let mut closed = false;
                while let Some(dc) = chars.next() {
                    match dc {
                        '"' => {
                            closed = true;
                            break;
                        }
                        '\\' => match chars.peek().copied() {
                            Some('"') => {
                                chars.next();
                                current.push('"');
                            }
                            Some('\\') => {
                                chars.next();
                                current.push('\\');
                            }
                            _ => current.push('\\'),
                        },
                        other => current.push(other),
                    }
                }
                if !closed {
                    return Err(format!("unterminated double quote in command `{cmd}`"));
                }
            }
            other => {
                in_token = true;
                current.push(other);
            }
        }
    }

    if in_token {
        args.push(current);
    }
    Ok(args)
}

/// Joins argv tokens into a command string that `split_command` will parse back losslessly.
pub fn join_command(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            let needs_quote = arg.is_empty()
                || arg
                    .chars()
                    .any(|c| c.is_whitespace() || c == '"' || c == '\'');
            if !needs_quote {
                return arg.clone();
            }
            let mut quoted = String::with_capacity(arg.len() + 2);
            quoted.push('"');
            for c in arg.chars() {
                match c {
                    '\\' => quoted.push_str("\\\\"),
                    '"' => quoted.push_str("\\\""),
                    other => quoted.push(other),
                }
            }
            quoted.push('"');
            quoted
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parses CLI arguments (excluding argv[0]).
pub fn parse_args(raw: &[String]) -> Result<ParsedArgs, String> {
    let mut verbose = false;
    let mut idx = 0;

    // Leading global flags before the subcommand.
    while idx < raw.len() {
        let arg = raw[idx].as_str();
        if is_verbose_flag(arg) {
            verbose = true;
            idx += 1;
        } else if is_help_flag(arg) {
            return Ok(ParsedArgs {
                verbose,
                action: Action::Help,
            });
        } else if is_version_flag(arg) {
            return Ok(ParsedArgs {
                verbose,
                action: Action::Version,
            });
        } else {
            break;
        }
    }

    let Some(cmd) = raw.get(idx).map(String::as_str) else {
        return Err(USAGE.into());
    };
    let mut rest = &raw[idx + 1..];

    // Subcommand-level `-h` / `--help` or `-v` / `--verbose` right after the subcommand.
    while let Some(first) = rest.first().map(String::as_str) {
        if is_verbose_flag(first) {
            verbose = true;
            rest = &rest[1..];
        } else if is_help_flag(first) {
            return Ok(ParsedArgs {
                verbose,
                action: Action::Help,
            });
        } else {
            break;
        }
    }

    let action = match cmd {
        "new" => {
            if rest.len() < 2 {
                return Err(format!(
                    "`tutor new` requires <slug> and <goal...>\n\n{USAGE}"
                ));
            }
            let slug = rest[0].clone();
            validate_slug(&slug)?;
            let goal = rest[1..].join(" ");
            if goal.trim().is_empty() {
                return Err("`tutor new` requires a non-empty goal".into());
            }
            Action::New { slug, goal }
        }
        "verify" => {
            let mut pos = Vec::new();
            for a in rest {
                if is_verbose_flag(a) {
                    verbose = true;
                } else if is_help_flag(a) {
                    return Ok(ParsedArgs {
                        verbose,
                        action: Action::Help,
                    });
                } else {
                    pos.push(a);
                }
            }
            if pos.len() != 1 {
                return Err(format!(
                    "`tutor verify` expects exactly one <challenge-dir>\n\n{USAGE}"
                ));
            }
            Action::Verify {
                dir: PathBuf::from(pos[0]),
            }
        }
        "init" => {
            if rest.len() < 2 {
                return Err(format!(
                    "`tutor init` requires <challenge-dir> and a run command\n\n{USAGE}"
                ));
            }
            let dir = rest[0].clone();
            let tail = &rest[1..];
            let (build, run_tokens) = if tail[0] == "--build" {
                if tail.len() < 3 {
                    return Err(format!(
                        "`--build` requires both a build command and a run command\n\n{USAGE}"
                    ));
                }
                let build_cmd = tail[1].clone();
                let build_parts = split_command(&build_cmd)?;
                if build_parts.first().is_none_or(|p| p.trim().is_empty()) {
                    return Err("`--build` command must not be empty".into());
                }
                (Some(build_cmd), &tail[2..])
            } else {
                (None, tail)
            };
            if run_tokens.first().is_none_or(|p| p.trim().is_empty()) {
                return Err("run command must not be empty".into());
            }
            let run = join_command(run_tokens);
            Action::Init { dir, build, run }
        }
        "show" | "test" | "hint" => {
            for a in rest {
                if is_verbose_flag(a) {
                    verbose = true;
                } else if is_help_flag(a) {
                    return Ok(ParsedArgs {
                        verbose,
                        action: Action::Help,
                    });
                } else {
                    return Err(format!(
                        "unexpected argument `{a}` for `tutor {cmd}`\n\n{USAGE}"
                    ));
                }
            }
            match cmd {
                "show" => Action::Show,
                "test" => Action::Test,
                _ => Action::Hint,
            }
        }
        "serve" => {
            let mut pos = Vec::new();
            for a in rest {
                if is_verbose_flag(a) {
                    verbose = true;
                } else if is_help_flag(a) {
                    return Ok(ParsedArgs {
                        verbose,
                        action: Action::Help,
                    });
                } else {
                    pos.push(a);
                }
            }
            if pos.len() > 1 {
                return Err(format!(
                    "unexpected argument `{}` for `tutor serve`\n\n{USAGE}",
                    pos[1]
                ));
            }
            let port = match pos.first() {
                None => 4242,
                Some(raw_port) => match raw_port.parse::<u16>() {
                    Ok(p) if p > 0 => p,
                    _ => {
                        return Err(format!(
                            "invalid port `{raw_port}`: expected an integer between 1 and 65535"
                        ));
                    }
                },
            };
            Action::Serve { port }
        }
        other => return Err(format!("unknown command `{other}`\n\n{USAGE}")),
    };

    Ok(ParsedArgs { verbose, action })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_s(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn split_command_handles_quotes_spaces_and_windows_paths() {
        assert_eq!(
            split_command(
                r#""C:\Program Files\Python313\python.exe" "my script.py" --msg 'hello world'"#
            )
            .unwrap(),
            vec![
                r"C:\Program Files\Python313\python.exe",
                "my script.py",
                "--msg",
                "hello world",
            ]
        );
        assert_eq!(
            split_command(r#"python reference\main.py --flag="a \"quoted\" val""#).unwrap(),
            vec!["python", r"reference\main.py", r#"--flag=a "quoted" val"#]
        );
        assert_eq!(
            split_command(r#"python -c """#).unwrap(),
            vec!["python", "-c", ""]
        );
        assert!(split_command(r#"python "unterminated"#).is_err());
        assert!(split_command("python 'unterminated").is_err());
    }

    #[test]
    fn join_command_round_trips_through_split_command() {
        let cases = vec![
            vec_s(&["python", "reference/main.py"]),
            vec_s(&[
                r"C:\Program Files\Python\python.exe",
                "my app.py",
                "--name",
                "a b",
            ]),
            vec_s(&["echo", r#"say "hi""#, "it's", "", r"C:\dir with space\"]),
        ];
        for args in cases {
            let joined = join_command(&args);
            let parsed = split_command(&joined).unwrap();
            assert_eq!(parsed, args, "round-trip failed for joined={joined:?}");
        }
    }

    #[test]
    fn parse_args_handles_help_version_and_verbose_positions() {
        assert_eq!(
            parse_args(&vec_s(&["--help"])).unwrap().action,
            Action::Help
        );
        assert_eq!(parse_args(&vec_s(&["-h"])).unwrap().action, Action::Help);
        assert_eq!(
            parse_args(&vec_s(&["serve", "--help"])).unwrap().action,
            Action::Help
        );
        assert_eq!(parse_args(&vec_s(&["-V"])).unwrap().action, Action::Version);
        assert_eq!(
            parse_args(&vec_s(&["--version"])).unwrap().action,
            Action::Version
        );

        let before = parse_args(&vec_s(&["-v", "test"])).unwrap();
        assert!(before.verbose);
        assert_eq!(before.action, Action::Test);

        let after = parse_args(&vec_s(&["test", "--verbose"])).unwrap();
        assert!(after.verbose);
        assert_eq!(after.action, Action::Test);

        let trailing = parse_args(&vec_s(&["verify", "challenges/wc", "-v"])).unwrap();
        assert!(trailing.verbose);
        assert_eq!(
            trailing.action,
            Action::Verify {
                dir: PathBuf::from("challenges/wc")
            }
        );
    }

    #[test]
    fn parse_args_rejects_build_fallthrough_and_empty_commands() {
        // Missing run command after --build must error instead of saving `--build make` as run.
        assert!(parse_args(&vec_s(&["init", "challenges/wc", "--build", "make"])).is_err());
        assert!(parse_args(&vec_s(&["init", "challenges/wc", "--build"])).is_err());
        assert!(parse_args(&vec_s(&[
            "init",
            "challenges/wc",
            "--build",
            "",
            "python",
            "a.py"
        ]))
        .is_err());
        assert!(parse_args(&vec_s(&[
            "init",
            "challenges/wc",
            "--build",
            "   ",
            "python",
            "a.py"
        ]))
        .is_err());
        assert!(parse_args(&vec_s(&["init", "challenges/wc", "   "])).is_err());

        // Valid init preserves spaces in run command and leaves `-v` inside run command alone.
        let parsed = parse_args(&vec_s(&[
            "init",
            "challenges/wc",
            "--build",
            "cargo build",
            r"C:\Program Files\app.exe",
            "-v",
        ]))
        .unwrap();
        assert!(!parsed.verbose);
        match parsed.action {
            Action::Init { dir, build, run } => {
                assert_eq!(dir, "challenges/wc");
                assert_eq!(build.as_deref(), Some("cargo build"));
                assert_eq!(
                    split_command(&run).unwrap(),
                    vec![r"C:\Program Files\app.exe", "-v"]
                );
            }
            other => panic!("expected Init, got {other:?}"),
        }
    }

    #[test]
    fn parse_args_validates_serve_port_and_extra_positional_args() {
        assert_eq!(
            parse_args(&vec_s(&["serve"])).unwrap().action,
            Action::Serve { port: 4242 }
        );
        assert_eq!(
            parse_args(&vec_s(&["serve", "8080"])).unwrap().action,
            Action::Serve { port: 8080 }
        );
        assert!(parse_args(&vec_s(&["serve", "notaport"])).is_err());
        assert!(parse_args(&vec_s(&["serve", "0"])).is_err());
        assert!(parse_args(&vec_s(&["serve", "70000"])).is_err());
        assert!(parse_args(&vec_s(&["serve", "8080", "extra"])).is_err());
        assert!(parse_args(&vec_s(&["show", "extra"])).is_err());
        assert!(parse_args(&vec_s(&["test", "extra"])).is_err());
        assert!(parse_args(&vec_s(&["hint", "extra"])).is_err());
        assert!(parse_args(&vec_s(&["verify", "a", "b"])).is_err());
    }

    #[test]
    fn validate_slug_blocks_traversal_and_invalid_names() {
        assert!(validate_slug("pid").is_ok());
        assert!(validate_slug("my_course-2").is_ok());
        assert!(validate_slug("").is_err());
        assert!(validate_slug("-bad").is_err());
        assert!(validate_slug(".hidden").is_err());
        assert!(validate_slug("../escape").is_err());
        assert!(validate_slug("a/b").is_err());
        assert!(validate_slug(r"a\b").is_err());
        assert!(validate_slug("/etc/passwd").is_err());
        assert!(parse_args(&vec_s(&["new", "../evil", "build a shell"])).is_err());
    }
}
