from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    model_server_base_url: str = "http://llama-cpp:8000/v1"
    model_server_model: str = "gemma-4-26b-a4b-it-gguf"
    model_server_timeout_seconds: float = 120.0

    label_temperature: float = 0.1
    label_max_tokens: int = 512
    label_model_max_retries: int = 1
    prompt_version: str = "label_v1"
    debug_model_io: bool = False

    max_decoded_image_bytes: int = 4_194_304
    max_image_pixels: int = 1_290_240

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def label_prompt_template(self) -> str:
        path = Path("prompts") / f"{self.prompt_version}.txt"
        return path.read_text(encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
