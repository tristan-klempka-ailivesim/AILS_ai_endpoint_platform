import asyncio
import json

import pytest
import respx
from fastapi import HTTPException
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


@respx.mock
def test_chat_completion_forwards_request_id_and_token_budget() -> None:
    settings = Settings(
        model_server_base_url="http://model.test/v1",
        model_server_model="gemma-4-26b-a4b-it-gguf",
        label_max_tokens=8192,
    )
    route = respx.post("http://model.test/v1/chat/completions").mock(
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

    asyncio.run(
        OpenAIModelClient(settings).chat_completion(
            system_prompt="system",
            user_prompt="user",
            image_data_url="data:image/png;base64,AAAA",
            request_id="rid-budget-test",
        )
    )

    request = route.calls.last.request
    assert request.headers["X-Request-ID"] == "rid-budget-test"
    assert json.loads(request.content)["max_tokens"] == 8192


@respx.mock
def test_chat_completion_rejects_reasoning_only_response() -> None:
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
                            "content": "",
                            "reasoning_content": "thinking without final answer",
                        }
                    }
                ]
            },
        )
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            OpenAIModelClient(settings).chat_completion(
                system_prompt="system",
                user_prompt="user",
                image_data_url="data:image/png;base64,AAAA",
                request_id="rid",
            )
        )

    assert exc.value.status_code == 502
    assert exc.value.detail == (
        "model returned only reasoning_content; increase max tokens or disable reasoning"
    )


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
