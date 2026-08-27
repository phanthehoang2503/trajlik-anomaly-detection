import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path


SHA256_HEX_LENGTH = 64


def sha256_file(path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a checkpoint or other artifact."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path) -> dict[str, object]:
    """Build the portable identity stored in cache metadata."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def checkpoint_identity_errors(identity) -> list[str]:
    """Return structural errors without requiring the original checkpoint path."""

    if not isinstance(identity, Mapping):
        return ["invad_checkpoint must be an object"]

    errors = []
    filename = identity.get("filename")
    if not isinstance(filename, str) or not filename:
        errors.append("invad_checkpoint.filename must be a non-empty string")

    size_bytes = identity.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        errors.append("invad_checkpoint.size_bytes must be a non-negative integer")

    digest = identity.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        errors.append("invad_checkpoint.sha256 must be a 64-character hex digest")
    return errors


def normalized_timestep_map(values: Sequence[object]) -> list[int]:
    """Normalize and validate the ordered diffusion timestep identity."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("timestep_map must be a sequence")
    normalized = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("timestep_map entries must be integers")
        try:
            integer = int(value)
        except (TypeError, ValueError) as error:
            raise TypeError("timestep_map entries must be integers") from error
        if integer < 0:
            raise ValueError("timestep_map entries must be non-negative")
        normalized.append(integer)
    if not normalized:
        raise ValueError("timestep_map cannot be empty")
    if normalized != sorted(set(normalized)):
        raise ValueError("timestep_map must be strictly increasing")
    return normalized
