"""Compare allowlisted public derived artifacts with one private canonical run."""

from __future__ import annotations

import argparse
from pathlib import Path

from refresh_public_reference import PUBLIC_FILES, sha256


def compare(canonical_run: Path, public_dir: Path) -> dict:
    checked = []
    for name in PUBLIC_FILES:
        private = canonical_run / name
        public = public_dir / name
        if not private.is_file() or not public.is_file():
            raise AssertionError(f"Missing comparison artifact: {name}")
        if private.read_bytes() != public.read_bytes():
            raise AssertionError(
                f"Artifact mismatch: {name} ({sha256(private)} != {sha256(public)})"
            )
        checked.append(name)
    for forbidden in ("predictions_oof.csv", "illustrative_exposure_path.csv", "run_manifest.json"):
        if (public_dir / forbidden).exists():
            raise AssertionError(f"Private artifact in public directory: {forbidden}")
    return {"status": "PASS", "checked": len(checked), "public_dir": str(public_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-run", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    args = parser.parse_args()
    import json

    print(json.dumps(compare(args.canonical_run.resolve(), args.public_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
