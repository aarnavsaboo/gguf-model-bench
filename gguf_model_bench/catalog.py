from pathlib import Path

from .gguf import inspect


def discover(path: str) -> list[dict]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob("*.gguf"))
    rows = []
    for item in files:
        try:
            info = inspect(str(item))
            meta = info.metadata
            rows.append({
                "path": info.path,
                "size_gb": info.size_bytes / (1024 ** 3),
                "version": info.version,
                "tensor_count": info.tensor_count,
                "metadata_count": info.metadata_count,
                "architecture": meta.get("general.architecture"),
                "name": meta.get("general.name"),
                "file_type": meta.get("general.file_type"),
                "quantization_version": meta.get("general.quantization_version"),
            })
        except Exception as exc:
            rows.append({
                "path": str(item),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    return rows
