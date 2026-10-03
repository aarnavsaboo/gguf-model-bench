from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO, Any


GGUF_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

SCALARS = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}


@dataclass(frozen=True)
class GGUFInfo:
    path: str
    version: int
    tensor_count: int
    metadata_count: int
    size_bytes: int
    metadata: dict[str, Any]


def _read_exact(handle: BinaryIO, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise EOFError("unexpected end of GGUF file")
    return data


def _u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _string(handle: BinaryIO) -> str:
    length = _u64(handle)
    if length > 16 * 1024 * 1024:
        raise ValueError("refusing unreasonable GGUF string length")
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _value(handle: BinaryIO, value_type: int, array_limit: int = 256):
    if value_type in SCALARS:
        fmt, size = SCALARS[value_type]
        return struct.unpack(fmt, _read_exact(handle, size))[0]
    if value_type == 8:
        return _string(handle)
    if value_type == 9:
        element_type = _u32(handle)
        length = _u64(handle)
        values = []
        keep = min(length, array_limit)
        for i in range(length):
            value = _value(handle, element_type, array_limit)
            if i < keep:
                values.append(value)
        if length > array_limit:
            return {"items": values, "truncated": True, "length": length}
        return values
    raise ValueError(f"unsupported GGUF metadata type {value_type}")


def inspect(path: str, array_limit: int = 256) -> GGUFInfo:
    target = Path(path)
    with target.open("rb") as handle:
        magic = _read_exact(handle, 4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file")
        version = _u32(handle)
        tensor_count = _u64(handle)
        metadata_count = _u64(handle)
        metadata = {}
        for _ in range(metadata_count):
            key = _string(handle)
            value_type = _u32(handle)
            metadata[key] = _value(handle, value_type, array_limit)
    return GGUFInfo(
        path=str(target),
        version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        size_bytes=target.stat().st_size,
        metadata=metadata,
    )
