# Web explainer provenance

The `web/` application and bundled model/reference snapshots are copied from the `lp-learner-private` snapshot `37cda1c8515c4e464ab1e1269c0ae79c6fadbac6` (2026-11-04). The application is a standalone TypeScript implementation of the teaching model; it does not call the Python package or Rust solver. The JSON data files are precomputed outputs from the Python reference project. Their regeneration workflow is not part of this website PR.

The source files are licensed under GPL-3.0-or-later; see `LICENSE`. The path-and-content SHA-256 of the copied `web/` files is `b246e209b4c5385fea3f2fdf385b2ca4060f2864173d72d734ddda569124b036`; paths are sorted relative to `web/`, and the digest is computed by hashing each UTF-8 path, a NUL byte and the binary SHA-256 of its contents. Build output, dependencies, caches, and `.DS_Store` are excluded.
