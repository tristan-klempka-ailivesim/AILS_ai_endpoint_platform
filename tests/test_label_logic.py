import pytest
from fastapi import HTTPException

from services.api.routes.label import build_label_prompt, parse_label_output
from shared.schemas import LabelRequest


def label_request() -> LabelRequest:
    return LabelRequest(
        image="data:image/png;base64,AAAA",
        segments=[
            {"id": 0, "color_name": "red", "rgb": [216, 38, 38]},
            {"id": 1, "color_name": "green", "rgb": [48, 232, 101]},
        ],
    )


def test_prompt_builder_preserves_order() -> None:
    prompt = build_label_prompt(label_request(), "Segments: {segments}. Count: {segment_count}")
    assert prompt.index("ID 0") < prompt.index("ID 1")
    assert "RGB=216,38,38" in prompt
    assert "Count: 2" in prompt


def test_parse_label_output_orders_by_request() -> None:
    labels = parse_label_output(
        '[{"id": 1, "name": "window", "material": "glass"},'
        '{"id": 0, "name": "hull", "material": "fiberglass"}]',
        [0, 1],
    )
    assert [label.id for label in labels] == [0, 1]


def test_parse_label_output_accepts_markdown_json_array() -> None:
    labels = parse_label_output(
        '```json\n[{"id":0,"name":"hull","material":"fiberglass"}]\n```',
        [0],
    )
    assert labels[0].name == "hull"


def test_parse_label_output_accepts_embedded_json_array() -> None:
    labels = parse_label_output(
        'Here is the result:\n[{"id":0,"name":"hull","material":"fiberglass"}]',
        [0],
    )
    assert labels[0].material == "fiberglass"


def test_parse_label_output_logs_request_id_on_invalid_json(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(HTTPException):
        parse_label_output("not json", [0], request_id="rid-123")

    assert "request_id=rid-123" in caplog.text


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"id": 0}',
        '[{"id": 0, "name": "hull"}]',
        '[{"id": 0, "name": "hull", "material": "fiberglass"},'
        '{"id": 0, "name": "duplicate", "material": "metal"}]',
        '[{"id": 2, "name": "extra", "material": "metal"}]',
    ],
)
def test_parse_label_output_rejects_invalid_shapes(raw: str) -> None:
    with pytest.raises(HTTPException):
        parse_label_output(raw, [0])
