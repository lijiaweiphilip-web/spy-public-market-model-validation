"""Portable, explicit hashing for public SPY artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".parquet"}
TEXT_HASH_MODE = "git-lf-v1"
RAW_HASH_MODE = "raw-bytes"


def is_safe_flat_path(name: str) -> bool:
    """Return whether *name* is a flat, relative POSIX artifact name."""

    if not isinstance(name, str) or not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    return (
        len(path.parts) == 1
        and path.parts[0] not in {".", ".."}
        and not path.is_absolute()
        and ":" not in path.parts[0]
    )


def canonical_bytes(path: Path, hash_mode: str) -> bytes:
    raw = path.read_bytes()
    if hash_mode == RAW_HASH_MODE:
        return raw
    if hash_mode == TEXT_HASH_MODE:
        return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    raise ValueError(f"unsupported artifact hash mode: {hash_mode}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def raw_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def artifact_hash_detail(path: Path) -> dict[str, str]:
    """Return a typed hash record; text records retain source raw provenance."""

    if path.suffix.lower() in BINARY_SUFFIXES:
        return {
            "content_type": "binary",
            "hash_algorithm": "sha256",
            "hash_mode": RAW_HASH_MODE,
            "canonical_sha256": raw_sha256(path),
        }
    return {
        "content_type": "text",
        "hash_algorithm": "sha256",
        "hash_mode": TEXT_HASH_MODE,
        "canonical_sha256": sha256_bytes(canonical_bytes(path, TEXT_HASH_MODE)),
        "source_raw_sha256": raw_sha256(path),
    }
