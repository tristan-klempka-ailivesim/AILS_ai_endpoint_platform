import base64
from io import BytesIO

import pytest
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from shared.schemas import LabelOutput, LabelRequest, validate_image_data_url


def png_data_url() -> str:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_valid_label_request_accepts_segments() -> None:
    request = LabelRequest(
        image=png_data_url(),
        segments=[{"id": 0, "color_name": "red", "rgb": [216, 38, 38]}],
    )
    assert request.segments[0].id == 0


def test_duplicate_segment_ids_fail() -> None:
    with pytest.raises(ValidationError):
        LabelRequest(
            image=png_data_url(),
            segments=[
                {"id": 0, "color_name": "red", "rgb": [216, 38, 38]},
                {"id": 0, "color_name": "green", "rgb": [48, 232, 101]},
            ],
        )


def test_invalid_rgb_fails() -> None:
    with pytest.raises(ValidationError):
        LabelRequest(
            image=png_data_url(),
            segments=[{"id": 0, "color_name": "red", "rgb": [999, 38, 38]}],
        )


def test_label_output_accepts_legacy_new_name_but_serializes_name() -> None:
    output = LabelOutput.model_validate(
        {"id": 0, "new_name": "hull", "material": "fiberglass"}
    )
    assert output.name == "hull"
    assert output.model_dump() == {
        "id": 0,
        "name": "hull",
        "material": "fiberglass",
    }


def test_validate_image_data_url_returns_metadata() -> None:
    metadata = validate_image_data_url(png_data_url(), max_decoded_bytes=10_000)
    assert metadata.media_type == "image/png"
    assert metadata.width == 1
    assert metadata.height == 1


def test_oversized_decoded_image_fails() -> None:
    with pytest.raises(HTTPException) as exc:
        validate_image_data_url(png_data_url(), max_decoded_bytes=1)
    assert exc.value.status_code == 413


def test_invalid_media_type_fails() -> None:
    bad = png_data_url().replace("data:image/png", "data:text/plain")
    with pytest.raises(HTTPException) as exc:
        validate_image_data_url(bad, max_decoded_bytes=10_000)
    assert exc.value.status_code == 422
