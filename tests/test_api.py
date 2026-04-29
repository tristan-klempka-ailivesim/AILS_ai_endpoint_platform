import base64
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services.api.main import create_app
from services.inference.openai_client import OpenAIModelClient
from shared.settings import get_settings


def png_data_url() -> str:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def label_payload() -> dict:
    return {
        "image": png_data_url(),
        "segments": [{"id": 0, "color_name": "red", "rgb": [216, 38, 38]}],
    }


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_SERVER_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("MODEL_SERVER_MODEL", "gemma-4-26b-a4b-it-gguf")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_health_reports_model_match(monkeypatch: pytest.MonkeyPatch) -> None:
    async def list_models(self: OpenAIModelClient) -> list[str]:
        return ["gemma-4-26b-a4b-it-gguf"]

    monkeypatch.setattr(OpenAIModelClient, "list_models", list_models)
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_server_model_match"] is True


def test_label_returns_array_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def chat_completion(self: OpenAIModelClient, **kwargs) -> str:
        return '[{"id":0,"name":"hull","material":"painted fiberglass"}]'

    monkeypatch.setattr(OpenAIModelClient, "chat_completion", chat_completion)
    client = TestClient(create_app())
    response = client.post(
        "/asset-parts/label",
        json=label_payload(),
        headers={"X-Request-ID": "test-rid"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-rid"
    assert response.headers["X-Prompt-Version"] == "label_v1"
    assert response.json() == [
        {"id": 0, "name": "hull", "material": "painted fiberglass"}
    ]


def test_label_malformed_model_output_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def chat_completion(self: OpenAIModelClient, **kwargs) -> str:
        return "not json"

    monkeypatch.setattr(OpenAIModelClient, "chat_completion", chat_completion)
    client = TestClient(create_app())
    response = client.post(
        "/asset-parts/label",
        json=label_payload(),
        headers={"X-Request-ID": "test-rid"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "model output was not valid JSON"
