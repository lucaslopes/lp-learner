use serde::Deserialize;
use serde_json::json;
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    command: String,
}

fn execute() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let request: Request = serde_json::from_str(&input)?;
    if request.schema != "lplearn-bridge/1" {
        return Err("unsupported schema; expected lplearn-bridge/1".into());
    }
    if request.command != "ping" && request.command != "info" {
        return Err("unsupported command; expected ping or info".into());
    }
    println!(
        "{}",
        json!({
            "schema": "lplearn-bridge/1",
            "command": request.command,
            "backend": "rust",
            "version": env!("CARGO_PKG_VERSION"),
            "capabilities": ["gda", "sgda"],
        })
    );
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("lplearn-native: {error}");
        std::process::exit(1);
    }
}
