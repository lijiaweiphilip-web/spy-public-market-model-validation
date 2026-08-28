# Changelog

## Unreleased

- Split `spy-validate validate` into a generic self-describing run validator and
  a strict `validate-reference` SPY contract validator.
- Add JSON Schemas for public mathematical claims and the reference manifest,
  including explicit raw-byte and Git-LF hash modes.
- Cache synthetic test runs so contract-failure cases remain complete without
  rebuilding the full pipeline for every assertion. Canonical scientific
  artifacts are unchanged.

## [0.1.0] - release candidate

- Added purged expanding walk-forward validation for a five-day realised-variance proxy.
- Added fixed RiskMetrics-style EWMA (`lambda=0.94`) alongside historical mean, Ridge and Random Forest baselines.
- Added train-only temporal calibration, artifact-hash verification, negative-result reporting and a deterministic public synthetic CLI demo.
- Added Python 3.10-3.12 CI with coverage, compile and Ruff checks.
- Kept vendor raw bytes, point-level predictions, exposure paths and canonical private evidence outside the public-safe derived bundle.
- Clarified public documentation boundaries and refreshed the overview figure, provenance wording and method references.

The release date is intentionally omitted until a GitHub `v0.1.0` release is explicitly approved and created.
