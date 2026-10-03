# Benchmarking notes

Prompt processing and text generation are different workloads and should be compared separately.

Large RAG prompts can spend most of their time ingesting tokens before generation starts. Interactive chat with short turns may care more about decode throughput and first-response latency.

The wrapper stores llama-bench's own backend/build metadata instead of replacing it with local assumptions. This makes it possible to compare results across runtime builds later while keeping the original benchmark output available.
