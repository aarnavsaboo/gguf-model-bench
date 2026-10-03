# GGUF metadata notes

The metadata reader only walks the header and key/value metadata area. It does not parse tensor payloads or allocate model weights.

GGUF metadata can contain scalars, strings and arrays. Large arrays are truncated in the returned Python representation after a configurable item limit so tokenizer tables do not turn a quick inspection into a large in-memory copy.

The catalog extracts a small set of common `general.*` fields when present while preserving the full metadata reader for deeper inspection.
