# Changelog

## 0.2.2 (unreleased)

### Fixed

- Align runtime `__version__` with distribution metadata; add an installed-package version regression.
- Preserve cosine direction across finite input scales and reject KL-divergence inputs outside the supported numeric range (#18).
- Bound CLI input reads and nesting; reject nonregular files, duplicate JSON keys, and nonfinite numbers (#19).
- Use the shared canonical JSON implementation when available while retaining the local fallback and existing valid-JSON bytes (#12).

### Packaging

- Sync the canonical release gate, including isolated installed-wheel probes and dependency checks, and add its wheel-isolation regressions.

The implementation fixes above are changes since 0.2.1 already present on main. This release preparation changes version identity, packaging validation, and tests. Earlier releases are documented in the repository's GitHub Releases.
