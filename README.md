# AILS AI Endpoint Platform

Internal FastAPI endpoint for centralized vision-language inference. Phase 1 exposes a single `/asset-parts/label` API so users can send a segmented image plus color metadata and receive structured labels without running local LM Studio setups.

## Scope

Implemented now:

- `POST /asset-parts/label`
- `GET /health`
- server-owned prompt in `prompts/label_v1.txt`
- OpenAI-compatible model-server client
- Docker Compose setup for API plus optional GPU llama.cpp service
- smoke script using `tmp/combined_prompt.png`

Not implemented in phase 1:

- authentication
- persistence
- feedback collection
- training/fine-tuning
- multiple inference endpoints

## Architecture

```text
Client
  -> FastAPI /asset-parts/label
  -> llama.cpp /v1/chat/completions
  -> quantized Gemma 4 GGUF
```

The API and model server run as separate services. The API owns request validation and prompt construction; llama.cpp owns quantized GGUF model serving.

## Dependencies

This repo uses `uv`.

```bash
uv sync
uv run uvicorn services.api.main:app --reload
uv run pytest
```

## Configuration

The repo includes a default `.env` for the current deployment machine. Keep
`.env.example` as a reference if this is deployed elsewhere later.

| Variable | Default | Description |
| --- | --- | --- |
| `API_PORT` | `8080` | Host port for the API container. |
| `MODEL_SERVER_BASE_URL` | `http://llama-cpp:8000/v1` | OpenAI-compatible model API base URL. Use `http://localhost:8000/v1` when running the API directly on the host. |
| `MODEL_SERVER_MODEL` | `gemma-4-26b-a4b-it-gguf` | Model name sent in chat completion requests. Must match `/models`. |
| `MODEL_SERVER_TIMEOUT_SECONDS` | `120` | HTTP timeout for model calls. |
| `LABEL_TEMPERATURE` | `0.1` | Generation temperature. |
| `LABEL_MAX_TOKENS` | `2048` | Max generated tokens. Gemma 4 may spend some budget in hidden reasoning before final content. |
| `PROMPT_VERSION` | `label_v1` | Prompt template version. |
| `DEBUG_MODEL_IO` | `false` | Logs redacted model request payloads and raw model text outputs for debugging. Never logs base64 image data. |
| `MAX_DECODED_IMAGE_BYTES` | `4194304` | Max decoded image size. |
| `LLAMA_CPP_PORT` | `8000` | Host port for llama.cpp. |
| `LLAMA_CPP_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | llama.cpp CUDA server image. |
| `LLAMA_CPP_CUDA_VISIBLE_DEVICES` | `1` | Host GPU index exposed to the llama.cpp container. Use `1` to reserve GPU 0 for other work. |
| `LLAMA_CPP_MODEL_URL` | Unsloth `UD-Q4_K_XL` GGUF URL | Exact GGUF file URL to download and serve. |
| `LLAMA_CPP_MMPROJ_URL` | Unsloth `mmproj-BF16.gguf` URL | Multimodal projector URL required for image input. |
| `LLAMA_CPP_MODEL_ALIAS` | `gemma-4-26b-a4b-it-gguf` | Served model alias returned by `/models`. |
| `LLAMA_CPP_CTX_SIZE` | `3200` | Context size. Keep conservative on 48 GB GPUs. |
| `LLAMA_CPP_N_GPU_LAYERS` | `999` | Try to offload all layers to GPU. Lower if startup fails. |
| `LLAMA_CPP_IMAGE_MAX_TOKENS` | `560` | Vision encoder image-token budget. Lower uses less memory. |
| `LLAMA_CPP_REASONING` | `off` | Disables Gemma 4 reasoning mode so responses go to `message.content` instead of spending the token budget in `reasoning_content`. |

`MODEL_SERVER_MODEL` must match the model name returned by `{MODEL_SERVER_BASE_URL}/models`.

## API

### `GET /health`

Returns API status, prompt version, model-server reachability, and whether `MODEL_SERVER_MODEL` appears in `/models`.

### `POST /asset-parts/label`

Request:

```json
{
  "image": "data:image/png;base64,...",
  "segments": [
    {"id": 0, "color_name": "red", "rgb": [216, 38, 38]}
  ]
}
```

Optional header:

```text
X-Request-ID: client-request-id
```

Success response:

```json
[
  {"id": 0, "name": "yacht hull", "material": "painted fiberglass"}
]
```

Success headers:

```text
X-Request-ID: <request_id>
X-Prompt-Version: label_v1
```

The API accepts `image/png`, `image/jpeg`, and `image/webp` data URLs. It decodes the image once to verify it before calling the model server.

`POST /label` is kept as a deprecated compatibility alias during the initial transition.

Frontend integration details are documented in [docs/frontend_integration.md](docs/frontend_integration.md).

## Docker Operations

The API image is built from this repo's `Dockerfile`. The llama.cpp model server
uses the external CUDA image configured by `LLAMA_CPP_IMAGE`, defaulting to:

```text
ghcr.io/ggml-org/llama.cpp:server-cuda
```

Both Compose services use `restart: unless-stopped`.

Start the API and managed GPU llama.cpp server in the background:

```bash
sudo docker compose --profile gpu up -d --build
```

Use the same command without `--build` when code and Dockerfile have not changed:

```bash
sudo docker compose --profile gpu up -d
```

Check status and readiness:

```bash
sudo docker compose ps
curl http://localhost:8080/health
```

View logs:

```bash
sudo docker compose logs -f api
sudo docker compose logs -f llama-cpp
```

Stop both services:

```bash
sudo docker compose down
```

Run only the API, pointing to an external OpenAI-compatible model server:

```bash
sudo docker compose up -d api
```

Run only the managed GPU llama.cpp service:

```bash
sudo docker compose --profile gpu up -d llama-cpp
```

The GPU profile starts llama.cpp with `${LLAMA_CPP_MODEL_URL}` and `${LLAMA_CPP_MMPROJ_URL}`, then serves it under `${LLAMA_CPP_MODEL_ALIAS}`. The downloaded model is cached in the `llama-cpp-cache` Docker volume so restarts do not redownload the GGUF. If the model is gated or download-limited, export `HF_TOKEN` in the shell before starting Compose.

## Smoke Test

```bash
MODEL_SERVER_BASE_URL=http://localhost:8000/v1 uv run python tools/smoke_label.py
```

The script verifies `MODEL_SERVER_MODEL` against `/models`, sends the sample image and six segment colors, then prints the response body plus `X-Request-ID` and `X-Prompt-Version`.

Optional live integration tests are also available when llama.cpp and the API are already running:

```bash
RUN_INTEGRATION=1 API_URL=http://localhost:8080 uv run pytest -m integration
```

Normal `uv run pytest` keeps these tests skipped.

## Load Test

Use the load test to check several concurrent `/asset-parts/label` requests against the running API:

```bash
uv run python tools/load_label.py --api-url http://localhost:8080 --requests 5 --concurrency 2
```

For a slightly heavier local check:

```bash
uv run python tools/load_label.py --api-url http://localhost:8080 --requests 10 --concurrency 4
```

The load test exits nonzero if any threshold fails. Defaults are based on the
initial local baseline:

```text
--min-success-rate 1.0
--max-p50-latency 10
--max-latency 20
```

Watch llama.cpp memory and GPU usage while this runs:

```bash
nvidia-smi
sudo docker stats ails_ai_endpoint_platform-llama-cpp-1
```

## Notes

- Clients do not send prompts or OpenAI-style messages.
- Base64 image contents are not logged or persisted.
- Feedback, storage, evaluation, and model improvement loops are future work.
