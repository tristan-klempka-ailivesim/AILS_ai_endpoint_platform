import asyncio

import respx
from httpx import Response

from services.inference.openai_client import OpenAIModelClient, redact_model_payload
from shared.settings import Settings


@respx.mock
def test_list_models() -> None:
    settings = Settings(
        model_server_base_url="http://model.test/v1",
        model_server_model="gemma-4-26b-a4b-it-gguf",
    )
    respx.get("http://model.test/v1/models").mock(
        return_value=Response(200, json={"data": [{"id": "gemma-4-26b-a4b-it-gguf"}]})
    )
    assert asyncio.run(OpenAIModelClient(settings).list_models()) == [
        "gemma-4-26b-a4b-it-gguf"
    ]


@respx.mock
def test_chat_completion_returns_content() -> None:
    settings = Settings(
        model_server_base_url="http://model.test/v1",
        model_server_model="gemma-4-26b-a4b-it-gguf",
    )
    respx.post("http://model.test/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '[{"id":0,"name":"hull","material":"metal"}]'
                        }
                    }
                ]
            },
        )
    )
    content = asyncio.run(
        OpenAIModelClient(settings).chat_completion(
            system_prompt="system",
            user_prompt="user",
            image_data_url="data:image/png;base64,AAAA",
            request_id="rid",
        )
    )
    assert "hull" in content


def test_redact_model_payload_removes_image_data_url() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "prompt"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,SECRET"},
                    },
                ],
            }
        ]
    }

    redacted = redact_model_payload(payload)

    assert redacted["messages"][0]["content"][1]["image_url"]["url"] == (
        "<redacted image data url>"
    )
    assert payload["messages"][0]["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,SECRET"
    )
