# Reproducible dependencies and security backports

The Rust build requires Rust 1.95+ and uses sibling Burn and CubeCL checkouts. Burn is deliberately
pinned to `=0.22.0-pre.2`: a caret prerelease constraint can select a later stable
release and silently leave the local patches unused. `Cargo.lock` is tracked for
reproducible CLI builds. Use `--locked`; library consumers need their own lockfile.

## Prepare a fresh checkout

From the YOLO repository, with the sibling destination directories absent:

```sh
git clone https://github.com/mahogny/burn.git ../burn
git -C ../burn checkout e20ada6837eaadf59e2e78efd572788c178c9bd5
git -C ../burn apply "$PWD/patches/burn-security.patch"
git clone https://github.com/tracel-ai/cubecl.git ../cubecl-security
git -C ../cubecl-security checkout 48348968468e433235be3db6e7c344c34bc446c0
git -C ../cubecl-security apply "$PWD/patches/cubecl-security.patch"
cargo check --locked --all-targets
cargo audit --file Cargo.lock
```

For existing checkouts, inspect their status and use separate worktrees rather
than replacing local changes. The CubeCL patch targets Burn's exact prerelease;
the newer sibling `../cubecl` checkout is intentionally preserved. The patched
`cubecl-environment` source is unchanged apart from its SQLite dependency.

## Changes

- Burn: `rusqlite` 0.41, `r2d2_sqlite` 0.36, `serde_rusqlite` 0.44. CubeCL's
  environment cache also uses `rusqlite` 0.41. Both must agree on the SQLite ABI;
  patched bundled SQLite requires `libsqlite3-sys >= 0.39`.
- Burn: `tch` / `torch-sys` 0.27 replace the vulnerable ZIP 0.6 compression stack
  with ZIP 9.5 and Zstandard 0.14 / `zstd-safe` 8.2. Burn's own ZIP dependency is
  also updated to 9.5. LibTorch remains available; its required native PyTorch
  version is now **2.15.0**, not 2.9.0. Never skip the native version check to use
  an older installation.
- The lockfile retains `pliron` and `pliron-derive` 0.17 together. Their old
  `version = "0"` dependency constraints otherwise select incompatible 0.20
  components during a broad update. If regenerating the lockfile, downgrade
  `pliron@0.20.0` to 0.17.0 first, then `pliron-derive` to 0.17.0.

These changes address RUSTSEC-2026-0459, RUSTSEC-2026-0429, RUSTSEC-2026-0368,
RUSTSEC-2026-0369, RUSTSEC-2026-0367 and RUSTSEC-2026-0403. The local resolved
lockfile audits with zero vulnerabilities and four unmaintained-package warnings
on 2026-11-11. No advisories or backends were suppressed or removed.

## Applications consuming YOLO

Cargo ignores dependency-level `[patch]` tables. The application must repeat the
Burn, Burn Store and CubeCL environment patches from this manifest in its own
workspace root. Dependency path patches resolve source identity, not a promise
that arbitrary sibling checkout contents are compatible: use the revisions and
patches above. An application lockfile alone cannot supply these source changes.

Hamilton's committed tool manifest still pins the previously published YOLO and
Burn revisions. These local changes do not update those remote pins or clear
GitHub alerts by themselves. Publish the dependency fixes (or reproduce the
patches in a managed source distribution), update the application's patches/pins,
and regenerate and audit its lockfile before claiming the application is fixed.

## Validation on 2026-11-11

- `cargo check --locked --all-targets` passed for the default CPU/WGPU build.
- `cargo test --locked --lib model::model::tests` passed all four existing
  feature extraction, training-forward, fusion and parameter-freezing tests.
- A temporary consumer enabled `libtorch` and `burn-tch/doc`; Rust API checking
  passed against tch 0.27. This upstream doc-only mode skips native C++ linking
  and does not establish LibTorch runtime compatibility. Native LibTorch and
  CUDA execution were not tested.
- Both this lockfile and an isolated Hamilton detector/training dependency
  graph using these sources audit with zero vulnerability records. Four
  maintenance warnings remain: bincode, number_prefix, paste and unsafe-libyaml.
