import tempfile
import unittest
from pathlib import Path

from trajlik.cache_identity import (
    checkpoint_identity,
    checkpoint_identity_errors,
    normalized_timestep_map,
    sha256_file,
)


class CacheIdentityTest(unittest.TestCase):
    def test_checkpoint_identity_is_content_based(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "renamable.pth"
            checkpoint.write_bytes(b"stable checkpoint bytes")

            identity = checkpoint_identity(checkpoint)

            self.assertEqual(identity["filename"], checkpoint.name)
            self.assertEqual(identity["size_bytes"], checkpoint.stat().st_size)
            self.assertEqual(identity["sha256"], sha256_file(checkpoint))
            self.assertEqual(checkpoint_identity_errors(identity), [])

    def test_invalid_checkpoint_identity_is_reported(self):
        errors = checkpoint_identity_errors(
            {"filename": "", "size_bytes": -1, "sha256": "not-a-digest"}
        )

        self.assertEqual(len(errors), 3)

    def test_timestep_map_must_be_strictly_increasing(self):
        self.assertEqual(normalized_timestep_map([0, 499, 999]), [0, 499, 999])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            normalized_timestep_map([0, 0, 999])


if __name__ == "__main__":
    unittest.main()
