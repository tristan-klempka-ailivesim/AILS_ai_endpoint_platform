import base64
import os
import uuid
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

SEGMENTS = [
    {"id": 0, "color_name": "red", "rgb": [216, 38, 38]},
    {"id": 1, "color_name": "green", "rgb": [48, 232, 101]},
    {"id": 2, "color_name": "purple", "rgb": [168, 61, 244]},
    {"id": 3, "color_name": "gold/yellow", "rgb": [228, 212, 103]},
    {"id": 4, "color_name": "sky blue", "rgb": [25, 195, 229]},
    {"id": 5, "color_name": "hot pink", "rgb": [243, 36, 148]},
]


def integration_enabled() -> bool:
    return os.getenv("RUN_INTEGRATION") == "1"


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@pytest.mark.skipif(not integration_enabled(), reason="set RUN_INTEGRATION=1 to run")
def test_live_health() -> None:
    api_url = os.getenv("API_URL", "http://localhost:8080").rstrip("/")

    response = httpx.get(f"{api_url}/health", timeout=30)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["metadata_ready"] is False
    assert body["label_ready"] is True


@pytest.mark.skipif(not integration_enabled(), reason="set RUN_INTEGRATION=1 to run")
def test_live_asset_parts_label() -> None:
    api_url = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
    image_path = Path(os.getenv("INTEGRATION_IMAGE", "tmp/combined_prompt.png"))
    request_id = f"pytest-integration-label-{uuid.uuid4()}"
    payload = {"image": image_to_data_url(image_path), "segments": SEGMENTS}

    response = httpx.post(
        f"{api_url}/asset-parts/label",
        json=payload,
        headers={"X-Request-ID": request_id},
        timeout=180,
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id
    assert response.headers["X-Prompt-Version"] == "label_v1"
    body = response.json()
    assert isinstance(body, list)
    assert [item["id"] for item in body] == [segment["id"] for segment in SEGMENTS]
    assert all(item["name"] and item["material"] for item in body)
