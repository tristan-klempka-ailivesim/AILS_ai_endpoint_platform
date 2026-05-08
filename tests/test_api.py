import base64
import logging
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services.api.main import create_app
from services.inference.openai_client import OpenAIModelClient
from shared.settings import get_settings


def png_data_url(width: int = 1, height: int = 1) -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def label_payload(image: str | None = None) -> dict:
    return {
        "image": image or png_data_url(),
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


def test_label_generates_request_id_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def chat_completion(self: OpenAIModelClient, **kwargs) -> str:
        return '[{"id":0,"name":"hull","material":"painted fiberglass"}]'

    monkeypatch.setattr(OpenAIModelClient, "chat_completion", chat_completion)
    client = TestClient(create_app())
    response = client.post("/asset-parts/label", json=label_payload())

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_request_log_includes_request_id_status_and_latency(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def chat_completion(self: OpenAIModelClient, **kwargs) -> str:
        return '[{"id":0,"name":"hull","material":"painted fiberglass"}]'

    caplog.set_level(logging.INFO, logger="uvicorn.error")
    monkeypatch.setattr(OpenAIModelClient, "chat_completion", chat_completion)
    client = TestClient(create_app())
    response = client.post(
        "/asset-parts/label",
        json=label_payload(),
        headers={"X-Request-ID": "test-log-rid"},
    )

    assert response.status_code == 200
    assert "api request completed request_id=test-log-rid" in caplog.text
    assert "path=/asset-parts/label" in caplog.text
    assert "status=200" in caplog.text
    assert "latency_ms=" in caplog.text


@pytest.mark.parametrize(
    ("label", "width", "height"),
    [
        ("small", 64, 64),
        ("medium", 512, 512),
        ("large", 1136, 1135),
    ],
)
def test_label_accepts_valid_image_sizes(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    width: int,
    height: int,
) -> None:
    async def chat_completion(self: OpenAIModelClient, **kwargs) -> str:
        return '[{"id":0,"name":"hull","material":"painted fiberglass"}]'

    monkeypatch.setattr(OpenAIModelClient, "chat_completion", chat_completion)
    client = TestClient(create_app())
    response = client.post(
        "/asset-parts/label",
        json=label_payload(image=png_data_url(width, height)),
        headers={"X-Request-ID": f"test-{label}-image"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == f"test-{label}-image"


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


def test_label_retries_once_after_malformed_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def chat_completion(self: OpenAIModelClient, **kwargs) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "not json"
        return '[{"id":0,"name":"hull","material":"painted fiberglass"}]'

    monkeypatch.setenv("LABEL_MODEL_MAX_RETRIES", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(OpenAIModelClient, "chat_completion", chat_completion)
    client = TestClient(create_app())

    response = client.post(
        "/asset-parts/label",
        json=label_payload(),
        headers={"X-Request-ID": "test-rid"},
    )

    assert response.status_code == 200
    assert calls == 2
    assert response.json() == [
        {"id": 0, "name": "hull", "material": "painted fiberglass"}
    ]


def test_label_returns_502_after_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def chat_completion(self: OpenAIModelClient, **kwargs) -> str:
        nonlocal calls
        calls += 1
        return "not json"

    monkeypatch.setenv("LABEL_MODEL_MAX_RETRIES", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(OpenAIModelClient, "chat_completion", chat_completion)
    client = TestClient(create_app())

    response = client.post(
        "/asset-parts/label",
        json=label_payload(),
        headers={"X-Request-ID": "test-rid"},
    )

    assert response.status_code == 502
    assert calls == 2
    assert response.json()["detail"] == "model output was not valid JSON"


def test_label_oversized_image_dimensions_return_413(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def chat_completion(self: OpenAIModelClient, **kwargs) -> str:
        nonlocal calls
        calls += 1
        return '[{"id":0,"name":"hull","material":"painted fiberglass"}]'

    monkeypatch.setenv("MAX_IMAGE_PIXELS", "0")
    get_settings.cache_clear()
    monkeypatch.setattr(OpenAIModelClient, "chat_completion", chat_completion)
    client = TestClient(create_app())

    response = client.post(
        "/asset-parts/label",
        json=label_payload(),
        headers={"X-Request-ID": "test-rid"},
    )

    assert response.status_code == 413
    assert calls == 0
    assert response.json()["detail"] == "image dimensions too large"


def test_request_log_includes_failure_reason_for_http_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "0")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.post(
        "/asset-parts/label",
        json=label_payload(),
        headers={"X-Request-ID": "test-failure-log-rid"},
    )

    assert response.status_code == 413
    assert "api request completed request_id=test-failure-log-rid" in caplog.text
    assert "status=413" in caplog.text
    assert "failure_reason=image dimensions too large" in caplog.text


def test_label_rejects_image_above_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "1290240")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.post(
        "/asset-parts/label",
        json=label_payload(image=png_data_url(1137, 1135)),
        headers={"X-Request-ID": "test-over-limit-image"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "image dimensions too large"
