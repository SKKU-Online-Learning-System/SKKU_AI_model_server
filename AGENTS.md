# AGENTS.md

## Repository boundary

This repository is a stateless GPU inference server for SKKU Course Agent.

Allowed responsibilities:
- Text LLM serving
- Voice LLM serving
- ASR serving
- TTS serving
- HTTP APIs that expose those inference backends

Do not add application/business logic here. In particular, do not add RAG, databases, pgvector,
course logic, tool implementations, web search, visualization, chat history, VAD, turn-taking,
barge-in logic, or WebSocket session orchestration. Those belong in `SKKU_AI_agent`.

## Model policy

- Prefer official Qwen models and official Qwen packages.
- Do not silently replace official BF16 models with community quantized checkpoints.
- Do not hard-code application system prompts.
- Keep all model IDs, GPU mappings, ports, and context limits configurable through environment variables.
- Do not commit model weights, Hugging Face cache files, API keys, or generated user audio.

## Runtime and dependency policy

Package and process management uses **uv only**.

The school development session is already provided inside a GPU-enabled container, so this repository must
not require Docker, Docker Compose, Docker-in-Docker, or `/var/run/docker.sock`. Do not reintroduce Docker
runtime files unless the deployment environment is intentionally redesigned.

LLM and Speech runtimes are intentionally separate uv projects because Qwen ASR/TTS and vLLM have different
dependency constraints. Do not merge them into one Python environment without re-validating the official
package constraints and GPU runtime.

Service lifecycle is managed by `scripts/start_all.sh` and `scripts/stop_all.sh`. They launch the three
process groups directly in the existing school GPU environment and store PID files under `.run/` and logs
under `logs/` by default.

## Testing policy

Unit/API tests must not download or load large models. Use mocked ASR/TTS backends. Real model verification
belongs in `scripts/smoke_test.sh` and `scripts/benchmark.py` on the GPU server.
