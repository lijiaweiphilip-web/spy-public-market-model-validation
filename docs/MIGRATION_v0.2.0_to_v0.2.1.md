# Migration: v0.2.0 to v0.2.1

No runtime command changes are required. Install the v0.2.1 wheel or source
distribution as usual. The packaged `math_claims.schema.json` is now
byte-identical to the public root copy and validates the same provenance
required fields in both locations.

This is a correctness-only patch. Canonical scientific artifacts remain
unchanged; consumers should not expect new metrics or refreshed predictions.
