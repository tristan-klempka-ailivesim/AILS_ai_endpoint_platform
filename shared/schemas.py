import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
from typing import Annotated

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

ALLOWED_IMAGE_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp"}


class Segment(BaseModel):
    id: int
    color_name: str = Field(min_length=1)
    rgb: Annotated[list[int], Field(min_length=3, max_length=3)]

    @field_validator("rgb")
    @classmethod
    def validate_rgb(cls, rgb: list[int]) -> list[int]:
        if any(value < 0 or value > 255 for value in rgb):
            raise ValueError("rgb values must be between 0 and 255")
        return rgb


class LabelRequest(BaseModel):
    image: str
    segments: Annotated[list[Segment], Field(min_length=1)]

    @field_validator("image")
    @classmethod
    def validate_image_prefix(cls, image: str) -> str:
        if not image.startswith("data:image/"):
            raise ValueError("image must be a data URL")
        return image

    @model_validator(mode="after")
    def validate_unique_segment_ids(self) -> "LabelRequest":
        ids = [segment.id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment IDs must be unique")
        return self


class LabelOutput(BaseModel):
    id: int
    name: str = Field(min_length=1, validation_alias=AliasChoices("name", "new_name"))
    material: str = Field(min_length=1)


@dataclass(frozen=True)
class ImageMetadata:
    media_type: str
    encoded_size: int
    decoded_size: int
    width: int
    height: int


def _split_data_url(data_url: str) -> tuple[str, str]:
    header, separator, encoded = data_url.partition(",")
    if separator != "," or not encoded:
        raise HTTPException(status_code=422, detail="image must be a valid data URL")

    if not header.endswith(";base64"):
        raise HTTPException(status_code=422, detail="image data URL must be base64 encoded")

    media_type = header.removeprefix("data:").removesuffix(";base64")
    if media_type not in ALLOWED_IMAGE_MEDIA_TYPES:
        raise HTTPException(status_code=422, detail="unsupported image media type")

    return media_type, encoded


def validate_image_data_url(data_url: str, *, max_decoded_bytes: int) -> ImageMetadata:
    media_type, encoded = _split_data_url(data_url)
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="image base64 data was invalid") from exc

    if len(decoded) > max_decoded_bytes:
        raise HTTPException(status_code=413, detail="decoded image too large")

    try:
        with Image.open(BytesIO(decoded)) as image:
            image.verify()
        with Image.open(BytesIO(decoded)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="image data was not a valid image") from exc

    return ImageMetadata(
        media_type=media_type,
        encoded_size=len(encoded),
        decoded_size=len(decoded),
        width=width,
        height=height,
    )
