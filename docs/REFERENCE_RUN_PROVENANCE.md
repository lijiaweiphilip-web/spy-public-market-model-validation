# Reference-run provenance

V4.5 keeps the supplied handoff reference, the V4.4 local rerun, and the final
canonical reference run separate. The public reference bundle is generated
from the canonical run only; it is never assembled by mixing nearby metrics
from different environments.

The handoff and V4.4 local results differ by small floating-point amounts and
were produced under different Python/scientific-library versions. That is not
treated as a scientific contradiction, but it is not an exact reproducibility
receipt. The V4.5 canonical run records Python, platform, package versions,
code commit, configuration hash, source-data hash, and every public artifact
hash in one manifest. The raw vendor snapshot and point-level predictions stay
private. See `docs/DATA_AND_REPRODUCIBILITY.md` for the public/private tiers.
