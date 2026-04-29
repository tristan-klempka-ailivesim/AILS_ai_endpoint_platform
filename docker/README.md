# Docker Deployment

The API and llama.cpp are deployed as separate services.

Run only the API, pointing to an existing OpenAI-compatible model server:

```bash
docker compose up api
```

Run only the managed GPU llama.cpp service:

```bash
sudo docker compose --profile gpu up llama-cpp
```

Run the API and the managed GPU llama.cpp service:

```bash
sudo docker compose --profile gpu up
```

Before using the GPU profile, check `.env` and make sure `LLAMA_CPP_MODEL_URL`, `LLAMA_CPP_MMPROJ_URL`, `LLAMA_CPP_MODEL_ALIAS`, and `MODEL_SERVER_MODEL` agree. `.env.example` is kept as a reference for future deployments.
