from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from spy_validation.validation import _validate_json_schema, validate_reference_contract

ROOT = Path(__file__).resolve().parents[1]


def test_math_claims_schema_rejects_missing_required_field(tmp_path: Path) -> None:
    claims_path = ROOT / "results" / "reference_run" / "MATH_CLAIMS.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims.pop("canonical_run_id")
    with pytest.raises(RuntimeError, match="schema validation failed"):
        _validate_json_schema(claims, ROOT / "schemas" / "math_claims.schema.json", "claims")


def test_reference_schema_rejects_unsupported_manifest_version(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    shutil.copytree(ROOT / "results" / "reference_run", reference)
    manifest_path = reference / "PUBLIC_REFERENCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_version"] = 2
    # The public validator reports this before reading any scientific values.
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest_version 3"):
        validate_reference_contract(reference)


def test_reference_validator_requires_math_claims_in_artifact_map(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    shutil.copytree(ROOT / "results" / "reference_run", reference)
    manifest_path = reference / "PUBLIC_REFERENCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["public_artifact_sha256"].pop("MATH_CLAIMS.json")
    manifest["public_artifact_hash_details"].pop("MATH_CLAIMS.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError):
        validate_reference_contract(reference)
