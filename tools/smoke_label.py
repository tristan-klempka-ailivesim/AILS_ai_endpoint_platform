#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
from pathlib import Path

import httpx

SEGMENTS = [
    {"id": 0, "color_name": "red", "rgb": [216, 38, 38]},
    {"id": 1, "color_name": "green", "rgb": [48, 232, 101]},
    {"id": 2, "color_name": "purple", "rgb": [168, 61, 244]},
    {"id": 3, "color_name": "gold/yellow", "rgb": [228, 212, 103]},
    {"id": 4, "color_name": "sky blue", "rgb": [25, 195, 229]},
    {"id": 5, "color_name": "hot pink", "rgb": [243, 36, 148]},
]


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def check_model_server(model_server_base_url: str, model_server_model: str, timeout: float) -> None:
    response = httpx.get(f"{model_server_base_url.rstrip('/')}/models", timeout=timeout)
    response.raise_for_status()
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("model server /models response was not valid JSON") from exc
    if not isinstance(body, dict):
        raise RuntimeError("model server /models response must be an object")
    models = [item.get("id") for item in body.get("data", []) if isinstance(item, dict)]
    if model_server_model not in models:
        raise RuntimeError(
            f"MODEL_SERVER_MODEL '{model_server_model}' not found in /models response: {models}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the /asset-parts/label endpoint.")
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8080"))
    parser.add_argument("--image", default="tmp/combined_prompt.png")
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args()

    model_server_base_url = os.getenv("MODEL_SERVER_BASE_URL", "http://localhost:8000/v1")
    model_server_model = os.getenv("MODEL_SERVER_MODEL", "gemma-4-26b-a4b-it-gguf")
    timeout = float(os.getenv("MODEL_SERVER_TIMEOUT_SECONDS", "120"))

    if not args.skip_model_check:
        check_model_server(model_server_base_url, model_server_model, timeout)

    payload = {"image": image_to_data_url(Path(args.image)), "segments": SEGMENTS}
    response = httpx.post(
        f"{args.api_url.rstrip('/')}/asset-parts/label",
        json=payload,
        timeout=timeout,
    )

    print("X-Request-ID:", response.headers.get("X-Request-ID"))
    print("X-Prompt-Version:", response.headers.get("X-Prompt-Version"))
    print(json.dumps(response.json(), indent=2))

    if response.status_code >= 400:
        return 1
    body = response.json()
    if not isinstance(body, list) or len(body) != len(SEGMENTS):
        print("invalid response shape", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
