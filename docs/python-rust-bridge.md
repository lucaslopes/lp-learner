# lplearn Python and Rust bridge

The Python package calls a packaged Rust executable for GDA and SGDA. The
`backend` diagnostic reports the available capabilities separately from the
solver interface.

Download the wheel for your platform from the
[versioned GitHub Releases](https://github.com/lucaslopes/lplearn/releases).
With Python 3.12 and an activated virtual environment:

```sh
python -m pip install /path/to/lplearn-wheel.whl
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

On Windows, the wheel includes an `lplearn.cmd` launcher for virtual
environments and standard Python installations. For other installation
layouts, run `python -m lplearn backend ping` using the Python interpreter
into which the wheel was installed.

The worker accepts a JSON object on stdin with schema `lplearn-bridge/1` and
command `ping` or `info`. Its JSON reply identifies Rust, its version, and its
capability list. Diagnostics go to stderr and errors return nonzero. The Rust
solver worker uses the separate `lpl-core/1` request/response protocol.

Versioned releases use matching `vX.Y.Z` Git tags and Python and Rust package
metadata. The tag workflow builds and installs the platform wheels and source
distribution before it attaches them to the GitHub Release.
