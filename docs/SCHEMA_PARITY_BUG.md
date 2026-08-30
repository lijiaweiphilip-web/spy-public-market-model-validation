# Schema parity correctness note

The v0.2.0 public release exposed a packaging inconsistency: the checked-in
root `schemas/math_claims.schema.json` required six provenance fields that were
not present in the packaged schema loaded through `importlib.resources`.

This v0.2.1 candidate copies the packaged schema source of truth to the public
root and adds regression tests for byte equality and the complete provenance
required set. The patch does not regenerate or modify canonical predictions,
metrics, figures, seeds, or conclusions. It is a packaging/schema correctness
fix, not a scientific-result update.
