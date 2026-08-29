from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from spy_validation.provenance import artifact_hash_detail, canonical_bytes, sha256_bytes
from spy_validation.validation import validate_reference_contract

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference_run"


def test_text_canonical_hash_accepts_crlf_and_cr_only(tmp_path: Path) -> None:
    path = tmp_path / "artifact.csv"
    path.write_bytes(b"a,b\n1,2\n")
    expected = artifact_hash_detail(path)["canonical_sha256"]
    path.write_bytes(b"a,b\r\n1,2\r\n")
    assert artifact_hash_detail(path)["canonical_sha256"] == expected
    path.write_bytes(b"a,b\r1,2\r")
    assert artifact_hash_detail(path)["canonical_sha256"] == expected


def test_text_value_or_whitespace_change_fails_canonical_hash(tmp_path: Path) -> None:
    path = tmp_path / "artifact.csv"
    path.write_bytes(b"a,b\n1,2\n")
    expected = artifact_hash_detail(path)["canonical_sha256"]
    path.write_bytes(b"a,b\n1,3\n")
    assert artifact_hash_detail(path)["canonical_sha256"] != expected
    path.write_bytes(b"a,b \n1,2\n")
    assert artifact_hash_detail(path)["canonical_sha256"] != expected


def test_binary_hash_is_raw_bytes(tmp_path: Path) -> None:
    path = tmp_path / "plot.png"
    path.write_bytes(b"PNG\x00\x01")
    detail = artifact_hash_detail(path)
    assert detail["content_type"] == "binary"
    assert detail["hash_mode"] == "raw-bytes"
    original = detail["canonical_sha256"]
    path.write_bytes(b"PNG\x00\x02")
    assert artifact_hash_detail(path)["canonical_sha256"] != original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hash_mode", "raw-bytes"),
        ("canonical_sha256", None),
        ("public_artifact_sha256", {"../escape.csv": "0" * 64}),
    ],
)
def test_reference_validator_rejects_invalid_hash_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    reference = tmp_path / "reference"
    shutil.copytree(REFERENCE, reference)
    manifest_path = reference / "PUBLIC_REFERENCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "public_artifact_sha256":
        manifest[field] = value
    else:
        detail = manifest["public_artifact_hash_details"]["aggregate_metrics.csv"]
        detail[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError):
        validate_reference_contract(reference)


def test_reference_validator_rejects_duplicate_or_extra_manifest_file(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    shutil.copytree(REFERENCE, reference)
    (reference / "extra.txt").write_text("not allow-listed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file set differs"):
        validate_reference_contract(reference)


def test_manifest_does_not_hash_itself() -> None:
    manifest = json.loads((REFERENCE / "PUBLIC_REFERENCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert "PUBLIC_REFERENCE_MANIFEST.json" not in manifest["public_artifact_sha256"]
    assert "PUBLIC_REFERENCE_MANIFEST.json" not in manifest["public_artifact_hash_details"]


def test_canonical_bytes_is_explicitly_modeled() -> None:
    assert sha256_bytes(canonical_bytes(REFERENCE / "aggregate_metrics.csv", "git-lf-v1")) == (
        json.loads((REFERENCE / "PUBLIC_REFERENCE_MANIFEST.json").read_text(encoding="utf-8"))
        ["public_artifact_sha256"]["aggregate_metrics.csv"]
    )
