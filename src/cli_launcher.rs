//! Stable native entry point for the Python command-line interface.
//!
//! Maturin's `bin` binding installs Rust executables from both wheels and
//! source builds. The launcher finds Python beside itself in a virtual
//! environment, then delegates all command behavior to `python -m lplearn`.

use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{self, Command};

fn python_candidates() -> Vec<OsString> {
    if let Some(python) = env::var_os("LPLEARN_PYTHON") {
        return vec![python];
    }

    let mut candidates = Vec::new();
    if let Ok(executable) = env::current_exe() {
        if let Some(directory) = executable.parent() {
            #[cfg(windows)]
            {
                candidates.push(directory.join("python.exe").into_os_string());
                if let Some(root) = directory.parent() {
                    candidates.push(root.join("python.exe").into_os_string());
                }
            }
            #[cfg(not(windows))]
            {
                candidates.push(directory.join("python").into_os_string());
                candidates.push(directory.join("python3").into_os_string());
            }
        }
    }

    #[cfg(windows)]
    candidates.push(OsString::from("python.exe"));
    #[cfg(not(windows))]
    {
        candidates.push(OsString::from("python3"));
        candidates.push(OsString::from("python"));
    }
    candidates
}

fn main() {
    let arguments: Vec<OsString> = env::args_os().skip(1).collect();
    let mut failures = Vec::new();

    for python in python_candidates() {
        if PathBuf::from(&python).components().count() > 1 && !PathBuf::from(&python).is_file() {
            continue;
        }
        match Command::new(&python)
            .arg("-m")
            .arg("lplearn")
            .args(&arguments)
            .status()
        {
            Ok(status) => process::exit(status.code().unwrap_or(1)),
            Err(error) => failures.push(format!("{}: {error}", PathBuf::from(&python).display())),
        }
    }

    eprintln!(
        "lplearn: could not find or start Python; set LPLEARN_PYTHON to the interpreter for this installation"
    );
    for failure in failures {
        eprintln!("  {failure}");
    }
    process::exit(1);
}
