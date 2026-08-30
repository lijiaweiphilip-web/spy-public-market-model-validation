# Changelog

## 0.2.1 - 2026-08-30

- Correctness-only schema packaging patch: the root and packaged
  `math_claims.schema.json` now have identical bytes and the complete
  provenance `required` set.
- No canonical predictions, metrics, figures, seeds, or scientific conclusions
  changed.

## 0.2.0 - 2026-08-30

- `spy-validate validate` now recomputes fold, aggregate, regime, calibration
  and decision-cost tables from predictions instead of trusting manifest check
  labels alone.
- `spy-validate validate-reference` retains the strict SPY reference contract;
  the generic validator remains scoped to this repository's flat run schema.
- JSON Schema validation is available after a base `pip install .`.
- Split `spy-validate validate` into a generic self-describing run validator and
  a strict `validate-reference` SPY contract validator.
- Add JSON Schemas for public mathematical claims and the reference manifest,
  including explicit raw-byte and Git-LF hash modes.
- Cache synthetic test runs so contract-failure cases remain complete without
  rebuilding the full pipeline for every assertion. Canonical scientific
  artifacts are unchanged.

## [0.1.0] - 2026-08-26

- Added purged expanding walk-forward validation for a five-day realised-variance proxy.
- Added fixed RiskMetrics-style EWMA (`lambda=0.94`) alongside historical mean, Ridge and Random Forest baselines.
- Added train-only temporal calibration, artifact-hash verification, negative-result reporting and a deterministic public synthetic CLI demo.
- Added Python 3.10-3.12 CI with coverage, compile and Ruff checks.
- Kept vendor raw bytes, point-level predictions, exposure paths and canonical private evidence outside the public-safe derived bundle.
- Clarified public documentation boundaries and refreshed the overview figure, provenance wording and method references.
