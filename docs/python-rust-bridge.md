# lplearn Python and Rust bridge

The Python package calls a packaged Rust executable for GDA and SGDA. The
`backend` diagnostic reports the available capabilities separately from the
solver interface.

After the `0.1.0` release is published, install and check the diagnostic
backend with:

```sh
python -m pip install lplearn==0.1.0
lplearn --help
lplearn backend ping
```

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

The wheel includes a native `lplearn` launcher that locates Python beside
itself in a virtual environment and delegates to `python -m lplearn`. Set
`LPLEARN_PYTHON` to select the interpreter for nonstandard installations. A
Windows `lplearn.cmd` compatibility wrapper remains available. The release
matrix checks the supported x64 and ARM64 Linux, macOS, and Windows targets.

The worker accepts a JSON object on stdin with schema `lplearn-bridge/1` and
command `ping` or `info`. Its JSON reply identifies Rust, its version, and its
capability list. Diagnostics go to stderr and errors return nonzero. The Rust
solver worker uses the separate `lpl-core/1` request/response protocol.

Versioned releases use matching `vX.Y.Z` Git tags, Python package metadata,
and Rust crate metadata. The tag workflow checks the supported platform wheels
and sdist, then attaches those artifacts to a GitHub Release. TestPyPI upload
and clean-install checks run separately through the `testpypi` environment;
production PyPI publishing is not part of this release plan. Releases before
`1.0.0` are experimental.
