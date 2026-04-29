import copy
import logging

import httpx
from fastapi import HTTPException

from shared.settings import Settings

logger = logging.getLogger(__name__)


class OpenAIModelClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.model_server_base_url.rstrip("/")

    async def list_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        try:
            timeout = self.settings.model_server_timeout_seconds
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="model server timeout") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="model server connection failure") from exc

        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"model server /models returned {response.status_code}",
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="model server /models response was not valid JSON",
            ) from exc
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=502,
                detail="model server /models response must be an object",
            )
        return [
            item.get("id")
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]

    async def chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        request_id: str,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.settings.model_server_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            "temperature": self.settings.label_temperature,
            "max_tokens": self.settings.label_max_tokens,
        }

        if self.settings.debug_model_io:
            logger.warning(
                "model request request_id=%s payload=%s",
                request_id,
                redact_model_payload(payload),
            )

        try:
            timeout = self.settings.model_server_timeout_seconds
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"X-Request-ID": request_id},
                )
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="model server timeout") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="model server connection failure") from exc

        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"model server returned {response.status_code}",
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="model server response was not valid JSON",
            ) from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="model server response must be an object")

        if self.settings.debug_model_io:
            logger.warning("model raw response request_id=%s body=%s", request_id, data)

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(
                status_code=502,
                detail="model server response missing message content",
            ) from exc

        if not isinstance(content, str):
            raise HTTPException(status_code=502, detail="model server message content was not text")

        if not content and data["choices"][0]["message"].get("reasoning_content"):
            raise HTTPException(
                status_code=502,
                detail=(
                    "model returned only reasoning_content; "
                    "restart llama.cpp with reasoning disabled"
                ),
            )

        if self.settings.debug_model_io:
            logger.warning("model response request_id=%s content=%r", request_id, content)
        return content


def redact_model_payload(payload: dict) -> dict:
    redacted = copy.deepcopy(payload)
    for message in redacted.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, dict) and "url" in image_url:
                image_url["url"] = "<redacted image data url>"
    return redacted
