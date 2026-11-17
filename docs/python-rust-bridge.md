# lplearn bridge bootstrap

This bootstrap proves that a Python CLI installed from a wheel can invoke a
packaged Rust executable. It reports backend identity only; no solver or policy
calculation is available yet.

With Python 3.12, Rust, and Maturin available, build and install in a fresh venv:

```sh
export DATABASE_DIR=/path/to/lplearn-build
export CARGO_TARGET_DIR="$DATABASE_DIR/target"
mkdir -p "$DATABASE_DIR/wheels"
maturin build --release --out "$DATABASE_DIR/wheels"
python3.12 -m venv "$DATABASE_DIR/venv"
"$DATABASE_DIR/venv/bin/python" -m pip install "$DATABASE_DIR"/wheels/lplearn-*.whl
"$DATABASE_DIR/venv/bin/lplearn" backend ping
```

Prebuilt wheels bundle the worker and need no Rust installation at runtime.
Building from source requires a Rust toolchain. The Python interface is also
available through `import lplearn as lpl; lpl.backend_info()`.

On Windows, the wheel includes an `lplearn.cmd` launcher for virtual
environments and standard Python installations. For other installation
layouts, run `python -m lplearn backend ping` using the Python interpreter
into which the wheel was installed. Windows and Linux remain unverified until
their CI jobs complete.

The worker accepts a JSON object on stdin with schema `lplearn-bridge/1` and
command `ping` or `info`. Its JSON reply identifies Rust, its version, and an
empty capability list. Diagnostics go to stderr and errors return nonzero.
