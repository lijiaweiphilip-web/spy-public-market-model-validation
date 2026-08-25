# Reference-run provenance

Supplied handoff and local reruns are kept separate from the canonical
reference run. The public reference bundle is generated from one canonical run
only; it is never assembled by mixing nearby metrics from different
environments.

The handoff and earlier local results differ by small floating-point amounts and
were produced under different Python/scientific-library versions. That is not
treated as a scientific contradiction, but it is not an exact reproducibility
receipt. The canonical run records Python, platform, package versions, code
commit, artifact/repository commit, configuration hash, source-data hash, and
every public artifact hash in one manifest. In `environment.json`,
`code_commit` identifies the source state that produced predictions and metrics;
`artifact_repository_commit` identifies the repository commit that first
introduced the derived reference bundle. The raw vendor snapshot and
point-level predictions stay private. See `docs/DATA_AND_REPRODUCIBILITY.md`
for the public/private tiers.

## Canonical receipt

- Canonical run ID: `20260825T112527Z`.
- `code_commit`: `69dbbab8d6ea5d461c799416c0017e8960ca0afc`.
- `artifact_repository_commit`: `9b732b1c448d436e59808b816dfec2b2567996e0`.
  A subsequent metadata-only commit records the final provenance fields.
- Python 3.12.10; NumPy 2.5.2; pandas 3.0.5; scikit-learn 1.9.0;
  matplotlib 3.11.1; config SHA-256 `50714ae7f90ad53f8d025dfa4855f88c91d03a299598d320df123d231fe22905`;
  source SHA-256 `eae913fd646ed32c3904d9b6e61beb2216c103bacaf16a0a36eab7528970710a`.
- Strict validation: `PASS`; 27 folds; four models; 1,620 OOF rows per model;
  21 private artifact hashes verified.
- Public extraction/byte comparison: `PASS`; 19 derived files; raw vendor
  bytes, point predictions, exposure paths and private run manifest excluded.
