import struct
import tempfile
import unittest
from pathlib import Path

from gguf_model_bench.gguf import inspect
from gguf_model_bench.planner import expand
from gguf_model_bench.llama_bench import command


def s(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


class Tests(unittest.TestCase):
    def test_minimal_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.gguf"
            data = (
                b"GGUF" +
                struct.pack("<IQQ", 3, 0, 1) +
                s("general.architecture") +
                struct.pack("<I", 8) +
                s("example")
            )
            path.write_bytes(data)
            info = inspect(str(path))
            self.assertEqual(info.version, 3)
            self.assertEqual(info.metadata["general.architecture"], "example")

    def test_plan_and_command(self):
        rows = expand({"models":["a.gguf"],"prompt_tokens":[512],"generation_tokens":[64]})
        cmd = command(rows[0])
        self.assertIn("-o", cmd)
        self.assertIn("json", cmd)


if __name__ == "__main__":
    unittest.main()
