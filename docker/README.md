# Docker Deployment

The API and llama.cpp are deployed as separate services.

- `api` is built from the repo `Dockerfile`.
- `llama-cpp` uses the external image configured by `LLAMA_CPP_IMAGE`, defaulting to `ghcr.io/ggml-org/llama.cpp:server-cuda`.
- Both services use `restart: unless-stopped`.

Run both services detached:

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

Run only the API, pointing to an existing OpenAI-compatible model server:

```bash
sudo docker compose up -d api
```

Run only the managed GPU llama.cpp service:

```bash
sudo docker compose --profile gpu up -d llama-cpp
```

Before using the GPU profile, check `.env` and make sure `LLAMA_CPP_MODEL_URL`, `LLAMA_CPP_MMPROJ_URL`, `LLAMA_CPP_MODEL_ALIAS`, and `MODEL_SERVER_MODEL` agree. `.env.example` is kept as a reference for future deployments.
